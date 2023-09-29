import os
import transformers
import argparse
import torch
import logging
from torch.utils.data import ConcatDataset, DataLoader
from datasets import load_dataset, concatenate_datasets
import re

from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    AutoTokenizer,
    LlamaForCausalLM
)

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from utils import utils
from utils import indexing



def main():
    parser = argparse.ArgumentParser(description='OpenP5')
    parser = utils.parse_args(parser)
    args, extras = parser.parse_known_args()
    
    # setup
    utils.setup_logging(args)
    utils.setup_model_path(args)
    utils.set_seed(args.seed)
    
    # determine whether distributed
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        args.gradient_accumulation_steps = args.gradient_accumulation_steps // world_size
        
    # use wandb
    wandb_project = ""
    wandb_run_name = ""
    wandb_watch = ""  # options: false | gradients | all
    wandb_log_model = ""
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
        
    # load model, tokenizer
    if 't5' in args.backbone.lower():
        config = T5Config.from_pretrained(args.backbone)
        model = T5ForConditionalGeneration.from_pretrained(args.backbone, config=config)
        tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    elif 'llama' in args.backbone.lower():
        model = LlamaForCausalLM.from_pretrained(
            'meta-llama/' + args.backbone,
            load_in_8bit=True,
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/' + args.backbone)
    else:
        raise NotImplementError
        
        
    datasets = args.datasets.split(',')
    print("datasets: ", datasets)
    
    for dataset in datasets:
        train_data_file = os.path.join(args.data_path, dataset, f'{dataset}_{args.tasks}_{args.item_indexing}_train.json')
        valid_data_file = os.path.join(args.data_path, dataset, f'{dataset}_{args.tasks}_{args.item_indexing}_validation_{args.valid_prompt}.json')
        train_data = load_dataset("json", data_files=train_data_file, field='data')
        valid_data = load_dataset("json", data_files=valid_data_file, field='data')
    
    
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt, truncation=True, max_length=args.cutoff, padding=False, return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result
    
    def generate_prompt(data_point):
        return f'{data_point["input"]} {data_point["output"]}'
    
    def process_func(datapoint):
        if 't5' in args.backbone.lower():
            print("---  T5 Model ---")
            encoding = tokenize(datapoint['input'], add_eos_token=True)
            labels = tokenize(datapoint['output'], add_eos_token=True)
            encoding['labels'] = labels['input_ids']
        elif 'llama' in args.backbone.lower():
            print("--- LLAMA Model ---")
            user_prompt = generate_prompt({**datapoint, "output": ""})
            encoding_input = tokenize(user_prompt, add_eos_token=False)
            input_len = len(encoding_input["input_ids"])
            full_prompt = generate_prompt(datapoint)
            encoding = tokenize(full_prompt)
            
            encoding["labels"] = (
                [-100] * input_len
                + encoding["labels"][input_len:]
            )

        return encoding
        # return {**datapoint, **encoding}
    
    TrainSet = train_data['train'].shuffle().map(process_func, batched=True)
    ValidSet = valid_data['train'].shuffle().map(process_func, batched=True)
        
    # add token and resize embedding for collaborative indexing
    if args.item_indexing == 'collaborative':
        new_tokens = []
        for dataset in datasets:
            item_index_file = os.path.join(args.data_path, args.dataset, f'item_collaborative_indexing_{args.collaborative_token_size}_{args.collaborative_cluster}_{args.collaborative_last_token}.txt')
            item_info = utils.ReadLineFromFile(item_index_file)
            item_map = indexing.get_dict_from_lines(item_info)
            for idx in list(item_map.values()):
                new_token += re.findall(r'\<.*?\>', idx)
        tokenizer.add_tokens(ds.new_token)
        model.resize_token_embeddings(len(tokenizer))
    
    # randomly initialize number related tokens
    if args.random_initialize == 1:
        # logging.info("Random initialize number related tokens")
        utils.random_initialization(model, tokenizer, args.backbone)
        
    # apply lora
    if args.lora > 0:
        model = prepare_model_for_int8_training(model)

        config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.lora_target_modules.split(','),
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left" 
    
    # decide output dir
    if len(args.datasets.split(',')) > 1:
        folder_name = 'SP5'
    else:
        folder_name = args.datasets
    output_dir = os.path.join(args.model_dir, folder_name, args.item_indexing, args.backbone)
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=TrainSet,
        eval_dataset=ValidSet if args.valid_select > 0 else None,
        args= transformers.TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=True,
            logging_steps=args.logging_steps,
            optim=args.optim,
            evaluation_strategy="steps" if args.valid_select > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if args.valid_select > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if args.valid_select > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=False,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    
    trainer.train()
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    


if __name__ == "__main__":
    
    main()