from transformers import LlamaForCausalLM
from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
import torch

if __name__ == '__main__':
    model = LlamaForCausalLM.from_pretrained(
            'meta-llama/Llama-2-7b-hf',
            load_in_8bit=True,
            torch_dtype=torch.float16
        )
    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
            "q_proj",
            "v_proj",
        ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()