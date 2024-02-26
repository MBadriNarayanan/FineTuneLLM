import json
import os

from sklearn.model_selection import train_test_split
from tqdm import tqdm

random_seed = 42
val_size = 0.2
test_size = 0.5

data_dir = "data"
data_path = os.path.join(data_dir, "data.json")

output_dir = "fine_tune_data"


def write_jsonl(file_path, data):
    with open(file_path, "w") as json_file:
        for item in data:
            json_file.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    print("\n--------------------\nStarting data processing!\n--------------------\n")
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "valid.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")

    with open(data_path, "r") as json_file:
        json_data = json.load(json_file)

    text_list = []
    for idx, data_dict in tqdm(enumerate(json_data)):
        data_string = ""
        for item in data_dict["conversations"]:
            if item["from"]:
                data_string += "[INST] "
            elif item["gpt"]:
                data_string += "[/INST] "
            data_string += item["value"]
        text_list.append({"text": data_string})
    train_data, val_data = train_test_split(
        text_list, test_size=val_size, random_state=random_seed
    )
    val_data, test_data = train_test_split(
        val_data, test_size=test_size, random_state=random_seed
    )

    write_jsonl(file_path=train_path, data=train_data)
    write_jsonl(file_path=val_path, data=val_data)
    write_jsonl(file_path=test_path, data=test_data)
    print(
        "\n--------------------\nData processing has been completed!\n--------------------\n"
    )
