import json
import sys
import torch

from peft import PeftModel


from general_utils import (
    create_bnb_config_tokenizer,
    get_model_prediction,
    load_base_model,
)


if __name__ == "__main__":
    print(
        "\n--------------------\nStarting response generation!\n--------------------\n"
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

    text_field = config["Data"]["textField"]

    base_model_name = config["BaseModel"]["modelName"]
    four_bit_load = config["BaseModel"]["fourBitLoad"]
    four_bit_quant = config["BaseModel"]["fourBitQuant"]
    double_quant = config["BaseModel"]["doubleQuant"]
    trust_code = config["BaseModel"]["trustCode"]

    sequence_length = config["Train"]["sequenceLength"]

    user_prompt = config["Eval"]["userPrompt"]
    checkpoint_path = config["Eval"]["checkpointPath"]

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

    model = PeftModel.from_pretrained(model, checkpoint_path)

    generated_response = get_model_prediction(
        model=model,
        tokenizer=tokenizer,
        user_prompt=user_prompt,
        sequence_length=sequence_length,
    )

    del model
    torch.cuda.empty_cache()

    print(
        "\n--------------------\nResponse has been successfully generated from the model!\n--------------------\n"
    )
