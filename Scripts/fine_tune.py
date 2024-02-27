import json
import os
import sys
import torch
import wandb

from datetime import datetime
from trl import SFTTrainer


from general_utils import (
    create_bnb_config_tokenizer,
    create_directory,
    create_peft_config,
    initialise_training_arguments,
    load_base_model,
    load_data,
    prepare_model_for_training,
)


if __name__ == "__main__":
    print(
        "\n--------------------\nStarting the fine-tuning process!\n--------------------\n"
    )
    if len(sys.argv) != 2:
        print("Pass config file as argument!")
        sys.exit()

    filename = sys.argv[1]
    with open(filename, "rt") as fjson:
        config = json.load(fjson)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU available!")
    else:
        device = torch.device("cpu")
        print("GPU not available!")

    authentication_key = config["Authentication"]["wandbSecurityKey"]

    wandb.login(key=authentication_key)
    run = wandb.init(
        project="Fine tuning Mistral 7B", job_type="training", anonymous="allow"
    )

    data_path = config["Data"]["dataPath"]
    text_field = config["Data"]["textField"]

    base_model_name = config["BaseModel"]["modelName"]
    four_bit_load = config["BaseModel"]["fourBitLoad"]
    four_bit_quant = config["BaseModel"]["fourBitQuant"]
    double_quant = config["BaseModel"]["doubleQuant"]
    trust_code = config["BaseModel"]["trustCode"]

    new_model_name = config["FineTune"]["newModelName"]
    use_cache = config["FineTune"]["useCache"]
    target_modules = config["FineTune"]["targetModules"].split(",")
    rank_value = config["FineTune"]["rankValue"]
    lora_alpha = config["FineTune"]["loraAlpha"]
    lora_dropout = config["FineTune"]["loraDropout"]
    bias_value = config["FineTune"]["biasValue"]
    task_type = config["FineTune"]["taskType"]

    output_directory = config["Train"]["outputDirectory"]
    logging_directory = config["Train"]["loggingDirectory"]
    epochs = config["Train"]["numberOfEpochs"]
    device_batch_size = config["Train"]["deviceBatchSize"]
    gradient_accumulation = config["Train"]["gradientAccumulation"]
    optimizer_function = config["Train"]["optimizerFunction"]
    save_steps = config["Train"]["saveSteps"]
    logging_steps = config["Train"]["loggingSteps"]
    learning_rate = config["Train"]["learningRate"]
    weight_decay = config["Train"]["weightDecay"]
    grad_norm = config["Train"]["maxGradNorm"]
    max_steps = config["Train"]["maxSteps"]
    warmup_ratio = config["Train"]["warmupRatio"]
    use_reentrant = config["Train"]["useReentrant"]
    sequence_length = config["Train"]["sequenceLength"]

    user_prompt = config["Eval"]["userPrompt"]

    create_directory(directory=output_directory)
    output_directory = os.path.join(output_directory, new_model_name)
    create_directory(directory=output_directory)

    create_directory(directory=logging_directory)
    run_string = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = "{}_{}".format(new_model_name, run_string)

    dataset = load_data(data_path=data_path)

    bnb_config, tokenizer = create_bnb_config_tokenizer(
        base_model_name=base_model_name,
        four_bit_load=four_bit_load,
        four_bit_quant=four_bit_quant,
        double_quant=double_quant,
        trust_code=trust_code,
    )

    model = load_base_model(
        base_model_name=base_model_name, bnb_config=bnb_config, trust_code=trust_code
    )

    peft_config = create_peft_config(
        rank_value=rank_value,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias_value=bias_value,
        task_type=task_type,
        target_modules=target_modules,
    )

    model = prepare_model_for_training(
        model=model, peft_config=peft_config, use_cache=use_cache
    )

    training_arguments = initialise_training_arguments(
        output_directory=output_directory,
        logging_directory=logging_directory,
        epochs=epochs,
        device_batch_size=device_batch_size,
        gradient_accumulation=gradient_accumulation,
        optimizer_function=optimizer_function,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_norm=grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        use_reentrant=use_reentrant,
        run_name=run_name,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=sequence_length,
        dataset_text_field=text_field,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    trainer.train()
    trainer.model.save_pretrained(output_directory)
    wandb.finish()

    del model, trainer
    torch.cuda.empty_cache()

    print(
        "\n--------------------\nFine-tuning has been successfully completed!\n--------------------\n"
    )
