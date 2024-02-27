import json
import os

from tqdm import tqdm


from general_utils import create_directory


data_dir = "data"
data_path = os.path.join(data_dir, "data.json")

output_dir = "fine_tune_data"


def write_jsonl(file_path, data):
    with open(file_path, "w") as json_file:
        for item in data:
            json_file.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    print("\n--------------------\nCreating data for fine-tuning!\n--------------------\n")
    
    create_directory(directory=output_dir)
    fine_tune_data_path = os.path.join(output_dir, "data.jsonl")

    with open(data_path, "r") as json_file:
        json_data = json.load(json_file)

    text_list = []
    for idx, data_dict in tqdm(enumerate(json_data)):
        data_string = (
            "The conversation is between Human and AI assisatant named Samantha "
        )
        for item in data_dict["conversations"]:
            if item["from"] == "human":
                data_string += "[INST] "
            elif item["from"] == "gpt":
                data_string += "[/INST] "
            data_string += item["value"]
        text_list.append({"text": data_string})

    write_jsonl(file_path=fine_tune_data_path, data=text_list)
    print(
        "\n--------------------\nData for fine-tuning has been created!\n--------------------\n"
    )
