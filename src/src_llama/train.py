import os
import transformers
import argparse
import torch
import logging
from torch.utils.data import ConcatDataset, DataLoader
from datasets import load_dataset, concatenate_datasets
from TaskAlternateTrainer import TaskAlternateTrainer
import re
import bitsandbytes as bnb
from collections import defaultdict
from tqdm import tqdm
# import pdb


from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
)

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
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
        
    hf_key = '' # Add your huggingface token if necessary
    # load model, tokenizer
    if 'open' in args.backbone.lower():
        model = LlamaForCausalLM.from_pretrained(
            'openlm-research/' + args.backbone,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=device_map,
            use_auth_token=hf_key,
            cache_dir=args.cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained('openlm-research/' + args.backbone)
    elif 'llama' in args.backbone.lower():
        model = LlamaForCausalLM.from_pretrained(
            'meta-llama/' + args.backbone,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=device_map,
            cache_dir=args.cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/' + args.backbone)
    else:
        raise NotImplementError
        
    
        
    datasets = args.datasets.split(',')
    if len(datasets) == 1:
        dataset = datasets[0]
        if args.item_indexing == 'collaborative':
            train_data_file = os.path.join(args.data_path, dataset, f'{dataset}_{args.tasks}_{args.item_indexing}_{args.collaborative_token_size}_{args.collaborative_cluster}_{args.collaborative_last_token}_train.json')
            valid_data_file = os.path.join(args.data_path, dataset, f'{dataset}_{args.tasks}_{args.item_indexing}_{args.collaborative_token_size}_{args.collaborative_cluster}_{args.collaborative_last_token}_validation_{args.valid_prompt}.json')
        else:
            train_data_file = os.path.join(args.data_path, dataset, f'{dataset}_{args.tasks}_{args.item_indexing}_train.json')
            valid_data_file = os.path.join(args.data_path, dataset, f'{dataset}_{args.tasks}_{args.item_indexing}_validation_{args.valid_prompt}.json')
        train_data = load_dataset("json", data_files=train_data_file, field='data', cache_dir=args.cache_dir)
        valid_data = load_dataset("json", data_files=valid_data_file, field='data', cache_dir=args.cache_dir)
    else:
        train_data_list, valid_data_list = [], []
        for dataset in datasets:
            print(dataset)
            if args.item_indexing == 'collaborative':
                train_data_file = os.path.join(args.data_path, dataset, f'{dataset}_{args.tasks}_{args.item_indexing}_{args.collaborative_token_size}_{args.collaborative_cluster}_{args.collaborative_last_token}_train.json')
                valid_data_file = os.path.join(args.data_path, dataset, f'{dataset}_{args.tasks}_{args.item_indexing}_{args.collaborative_token_size}_{args.collaborative_cluster}_{args.collaborative_last_token}_validation_{args.valid_prompt}.json')
            else:
                train_data_file = os.path.join(args.data_path, dataset, f'{dataset}_{args.tasks}_{args.item_indexing}_train.json')
                valid_data_file = os.path.join(args.data_path, dataset, f'{dataset}_{args.tasks}_{args.item_indexing}_validation_{args.valid_prompt}.json')
            t_data = load_dataset("json", data_files=train_data_file, field='data', cache_dir=args.cache_dir)
            v_data = load_dataset("json", data_files=valid_data_file, field='data', cache_dir=args.cache_dir)
            train_data_list.append(t_data['train'])
            valid_data_list.append(v_data['train'])
        train_data = concatenate_datasets(train_data_list)
        valid_data = concatenate_datasets(valid_data_list)
    
    
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt, truncation=True, max_length=args.cutoff, padding=False, return_tensors=None,
        )
        if (isinstance(result["input_ids"][-1], int) and result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff
            and add_eos_token
           ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        elif isinstance(result["input_ids"][-1], list) and add_eos_token:
            for i in range(len(result['input_ids'])):
                if result["input_ids"][i][-1] != tokenizer.eos_token_id and len(result["input_ids"][i]) < args.cutoff:
                    result["input_ids"][i].append(tokenizer.eos_token_id)
                    result["attention_mask"][i].append(1)

        result["labels"] = result["input_ids"].copy()

        return result
    
    def generate_prompt(data_point, output=True):
        if isinstance(data_point['input'], list):
            if output:
                return [f'{data_point["input"][i]} Response: {data_point["output"][i]}' for i in range(len(data_point['input']))]
            else:
                return [f'{data_point["input"][i]} Response: ' for i in range(len(data_point['input']))]
        if output:
            return f'{data_point["input"]} Response: {data_point["output"]}'
        else:
            return f'{data_point["input"]} Response: '
    
    def process_func(datapoint):
        if 't5' in args.backbone.lower():
            encoding = tokenize(datapoint['input'], add_eos_token=True)
            labels = tokenize(datapoint['output'], add_eos_token=True)
            encoding['labels'] = labels['input_ids'].copy()
            encoding['output_attn'] = labels['attention_mask'].copy()
        elif 'llama' in args.backbone.lower():
            user_prompt = generate_prompt(datapoint, output=False)
            # print(len(user_prompt))
            # print(user_prompt[:10])
            encoding_input = tokenize(user_prompt, add_eos_token=False)
            if isinstance(user_prompt, list):
                input_len = [len(encoding_input["input_ids"][i]) for i in range(len(encoding_input["input_ids"]))]
            else:
                input_len = len(encoding_input["input_ids"])
            full_prompt = generate_prompt(datapoint)
            # print(full_prompt)
            encoding = tokenize(full_prompt)

            if isinstance(user_prompt, list):
                encoding["labels"] = [
                    [-100] * input_len[i]
                    + encoding["labels"][i][input_len[i]:] for i in range(len(encoding["labels"]))
                ]
            else:
                encoding["labels"] = (
                    [-100] * input_len
                    + encoding["labels"][input_len:]
                )

        # return encoding
        return {**datapoint,**encoding}
    
    # add token and resize embedding for collaborative indexing
    if args.item_indexing == 'collaborative':
        new_token = []
        for dataset in datasets:
            item_index_file = os.path.join(args.data_path, dataset, f'item_collaborative_indexing_{args.collaborative_token_size}_{args.collaborative_cluster}_{args.collaborative_last_token}.txt')
            item_info = utils.ReadLineFromFile(item_index_file)
            item_map = indexing.get_dict_from_lines(item_info)
            for idx in list(item_map.values()):
                new_token += re.findall(r'\<.*?\>', idx)
        tokenizer.add_tokens(new_token)
        model.resize_token_embeddings(len(tokenizer))
        
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left" 
        
    # no task alternating optimization if only one task in the data
    if len(datasets) == 1 and len(set(train_data['train']['task'])) == 1:
        args.task_alternating_optim = 0
    
    if args.task_alternating_optim == 1:
        TrainSet = dict()
        if len(datasets) == 1:
            for task in set(train_data['train']['task']):
                TrainSet[task] = train_data['train'].filter(lambda example: example["task"]==task)
        else:
            task_idx = defaultdict(list)
            task_list = train_data['task']
            for idx, element in enumerate(tqdm(task_list)):
                task_idx[element].append(idx)
            for task in set(train_data['task']):
                TrainSet[task] = train_data.select(task_idx[task])
        for task in TrainSet:
            TrainSet[task] = TrainSet[task].shuffle().select(range(int(len(TrainSet[task]) * args.sample_ratio))).map(process_func, batched=True, batch_size=1000)
    else:
        if len(datasets) == 1:
            TrainSet = train_data['train'].shuffle().select(range(int(len(TrainSet[task]) * args.sample_ratio))).map(process_func, batched=True, batch_size=1000)
        else:
            TrainSet = train_data.shuffle().select(range(int(len(TrainSet[task]) * args.sample_ratio))).map(process_func, batched=True, batch_size=1000)

    # valid_data['train'] = valid_data['train'].remove_columns(['task', 'instruction'])
    if args.valid_select > 0:
        if len(datasets) == 1:
            ValidSet = valid_data['train'].shuffle().map(process_func, batched=True, batch_size=1000)
        else:
            ValidSet = valid_data.shuffle().map(process_func, batched=True, batch_size=1000)
        
    
    
    # randomly initialize number related tokens
    if args.random_initialize == 1:
        # logging.info("Random initialize number related tokens")
        utils.random_initialization(model, tokenizer, args.backbone)
        
    # apply lora
    if args.lora > 0:
        model = prepare_model_for_kbit_training(model)

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
        
    
    
    # decide output dir
    if len(args.datasets.split(',')) > 1:
        folder_name = 'SP5'
    else:
        folder_name = args.datasets
    output_dir = os.path.join(args.model_dir, folder_name, args.item_indexing, args.backbone)
    
    if args.task_alternating_optim == 1:
        trainer = TaskAlternateTrainer(model=model,
            train_dataset=TrainSet,
            eval_dataset=ValidSet if args.valid_select > 0 else None,
            args= transformers.Seq2SeqTrainingArguments(
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                warmup_steps=args.warmup_steps,
                num_train_epochs=args.epochs,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                adam_epsilon=args.adam_eps,
                fp16=True,
                logging_dir=args.log_dir,
                logging_steps=args.logging_steps,
                optim=args.optim,
                evaluation_strategy="steps" if args.valid_select > 0 else "no",
                save_strategy="steps",
                eval_steps=100 if args.valid_select > 0 else None,
                save_steps=100,
                output_dir=output_dir,
                save_total_limit=3,
                load_best_model_at_end=True if args.valid_select > 0 else False,
                ddp_find_unused_parameters=False,
                group_by_length=False,
                report_to="wandb" if use_wandb else None,
                run_name=wandb_run_name if use_wandb else None,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
    else:
        trainer = transformers.Trainer(
            model=model,
            train_dataset=TrainSet,
            eval_dataset=ValidSet if args.valid_select > 0 else None,
            args= transformers.Seq2SeqTrainingArguments(
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                warmup_steps=args.warmup_steps,
                num_train_epochs=args.epochs,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                adam_epsilon=args.adam_eps,
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
                ddp_find_unused_parameters=False,
                group_by_length=False,
                report_to="wandb" if use_wandb else None,
                run_name=wandb_run_name if use_wandb else None,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
    # pdb.set_trace()
    trainer.train()
    # if args.lora > 0:
    #     model = model.merge_and_unload()
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    


if __name__ == "__main__":
    
    main()