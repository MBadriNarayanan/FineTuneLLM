import json
import os
import torch

import numpy as np

from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
    TextStreamer,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
set_seed(random_seed)


def create_directory(directory):
    try:
        os.mkdir(directory)
        print("Created directory: {}!".format(directory))
    except:
        print("Directory: {} already exists!".format(directory))


def create_bnb_config_tokenizer(
    base_model_name, four_bit_load, four_bit_quant, double_quant, trust_code
):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=four_bit_load,
        bnb_4bit_quant_type=four_bit_quant,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=double_quant,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=trust_code
    )
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    return bnb_config, tokenizer


def create_peft_config(
    rank_value, lora_alpha, lora_dropout, bias_value, task_type, target_modules
):
    peft_config = LoraConfig(
        r=rank_value,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias_value,
        task_type=task_type,
        target_modules=target_modules,
    )
    return peft_config


def get_model_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print("Number of Parameters: {}".format(all_param))
    print(
        "Number of Trainable Parameters: {} -> {:.3f}%".format(
            trainable_params, (trainable_params / all_param) * 100
        )
    )
    print("--------------------")


def get_model_prediction(model, tokenizer, user_prompt, sequence_length):
    model.config.use_cache = True
    model.eval()
    runtimeFlag = "cuda:0"
    data_string = "The conversation is between Human and AI assisatant named Samantha\n"
    prompt = "{}{}{}\n{}".format(data_string, "[INST]", user_prompt.strip(), "[/INST]")
    inputs = tokenizer([prompt], return_tensors="pt").to(runtimeFlag)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    _ = model.generate(
        **inputs, streamer=streamer, max_new_tokens=sequence_length
    )


def initialise_training_arguments(
    output_directory,
    logging_directory,
    epochs,
    device_batch_size,
    gradient_accumulation,
    optimizer_function,
    save_steps,
    logging_steps,
    learning_rate,
    weight_decay,
    grad_norm,
    max_steps,
    warmup_ratio,
    use_reentrant,
    run_name,
):
    training_arguments = TrainingArguments(
        output_dir=output_directory,
        logging_dir=logging_directory,
        num_train_epochs=epochs,
        per_device_train_batch_size=device_batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        optim=optimizer_function,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=False,
        bf16=False,
        max_grad_norm=grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="wandb",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": use_reentrant},
        run_name="{}".format(run_name),
    )
    return training_arguments


def load_data(data_path):
    with open(data_path, "r") as json_file:
        json_data = json.load(json_file)

    idx_list = []
    conversation_list = []
    for idx, data_dict in tqdm(enumerate(json_data)):
        data_string = (
            "The conversation is between Human and AI assisatant named Samantha "
        )
        for item in data_dict["conversations"]:
            if item["from"]:
                data_string += "[INST] "
            elif item["gpt"]:
                data_string += "[/INST] "
            data_string += item["value"]
        idx_list.append(idx)
        conversation_list.append(data_string)
    dataset_data = {"id": idx_list, "conversations": conversation_list}
    dataset = Dataset.from_dict(dataset_data)
    return dataset


def load_base_model(base_model_name, bnb_config, trust_code):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=trust_code,
    )
    return model


def prepare_model_for_training(model, peft_config, use_cache):
    model.config.use_cache = use_cache
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    get_model_parameters(model=model)
    return model
