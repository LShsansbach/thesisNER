################################ Aufbereitung der Ursprungsdaten ################################
import input

start = input.start_end_file(file=__file__.split("/")[-1])
################################ Start

# import libraries
import pandas as pd
from datasets import load_dataset
import os

# input
dataset = input.dataset
language_filt = input.language
entities_to_keep = input.entities_to_keep
label = input.label
tokens = input.text
random_state = input.random_state
number_data = input.number_of_data
dict_intins = input.dict_intins
dict_sinint = [{v: k for k, v in d.items()} for d in dict_intins]


# function
def get_all_data(dataset: str):
    try:
        data = load_dataset(dataset)
    except:
        data = load_dataset(dataset, language_filt)
    data_train = pd.DataFrame(data["train"], columns=[label, tokens])
    try:
        data_test = pd.DataFrame(data["test"], columns=[label, tokens])
        data_val = pd.DataFrame(data["validation"], columns=[label, tokens])
    except:
        data_train, data_test, data_val = create_test_val(data_train)
    if number_data:
        data_train = data_train[:number_data]
        data_test = data_test[:number_data]
        data_val = data_val[:number_data]
    return (
        translate_to_int_and_keep_certain_entities(data_train),
        translate_to_int_and_keep_certain_entities(data_test),
        translate_to_int_and_keep_certain_entities(data_val),
    )


def keep_only_certain_entities(data: pd.DataFrame, entities: list):
    "keep the entities in the list entities"
    list_outer = []
    for sentence in data[label]:
        list_inner = []
        for entity in sentence:
            if entity in entities:
                if "B-PER" in intins[entity]:
                    list_inner.append(sinint["B-PER"])
                elif "I-PER" in intins[entity]:
                    list_inner.append(sinint["I-PER"])
                elif "B-LOC" in intins[entity]:
                    list_inner.append(sinint["B-LOC"])
                elif "I-LOC" in intins[entity]:
                    list_inner.append(sinint["I-LOC"])
                elif "B-ORG" in intins[entity]:
                    list_inner.append(sinint["B-ORG"])
                elif "B-Org" in intins[entity]:
                    list_inner.append(sinint["B-Org"])
                elif "I-ORG" in intins[entity]:
                    list_inner.append(sinint["I-ORG"])
                elif "I-Org" in intins[entity]:
                    list_inner.append(sinint["I-Org"])
            else:
                list_inner.append(sinint["O"])
        list_outer.append(list_inner)
    data[label] = list_outer
    return data


def save_data(data: pd.DataFrame, save_path: str, part: str = "train"):
    os.makedirs(f"../dataset/{save_path}", exist_ok=True)
    data.to_csv(f"../dataset/{save_path}/{part}.csv", sep="|")
    data.to_pickle(f"../dataset/{save_path}/{part}.pkl")


def create_test_val(data: pd.DataFrame):
    # get randomly train, test, val from data
    data_train_cut = data.sample(frac=0.7, random_state=random_state)
    data_test = data.drop(data_train_cut.index)
    data_val = data_test.sample(frac=0.5, random_state=random_state)
    data_test = data_test.drop(data_val.index)
    return data_train_cut, data_test, data_val


def translate_to_int_and_keep_certain_entities(data: pd.DataFrame):
    """get a list of entities in int format"""
    if type(data[label].iloc[0][0]) == str:
        list_outer = []
        list_inner = []
        print("translate to int")
        for sentence in data[label]:
            for entity in sentence:
                list_inner.append(dict_sinint[i][entity])
            list_outer.append(list_inner)
        data[label] = list_outer
    return keep_only_certain_entities(data, int_entities_to_keep)


def entities_in_int(entities: list):
    new_entities = []
    for key in sinint:
        for entity in entities:
            if entity.lower() in key.lower():
                new_entities.append(sinint[key])
    return new_entities


def save_data_to_txt(data: pd.DataFrame, save_path: str):
    with open(f"../dataset/{save_path}/train.txt", "w", encoding="utf-8") as f:
        for sentence in data:
            count_help = 0
            for word in sentence:
                if (
                    word in (".", ",", "!", "?", ":", ";", "-", ")", "]")
                    or count_help == 0
                ):
                    f.write(word)
                    count_help += 1
                else:
                    f.write(" " + word)
            f.write("\n")
            # python -m prodigy ner.manual train blank:de ../dataset/germaner/train.txt --label I-PER,I-ORG,I-LOC,B-PER,B-ORG,B-LOC
            # python -m prodigy terms.to-patterns train --label I-PER,I-ORG,I-LOC,B-PER,B-ORG,B-LOC --spacy-model blank:de > ./train.jsonl


# Ablauf
for i in range(len(dataset)):
    intins = dict_intins[i]
    sinint = dict_sinint[i]
    int_entities_to_keep = entities_in_int(entities_to_keep)
    data_train, data_test, data_val = get_all_data(dataset[i])
    save_data(data_train, dataset[i], part="train")
    save_data(data_test, dataset[i], part="test")
    save_data(data_val, dataset[i], part="val")
    save_data_to_txt(data_train[tokens], dataset[i])
    input.write_json({"dataset": dataset[i]})
    input.write_json({"dataset": f"snorkel_{dataset[i]}"})
    input.write_json({"dataset": f"active_{dataset[i]}"})
    input.write_json({"dataset": f"augmentation_{dataset[i]}"})
    input.write_json({"dataset": f"transfer_{dataset[i]}"})
    input.write_json({"dataset": f"combination_{dataset[i]}"})
    print(len(data_train), len(data_test), len(data_val))
del data_train, data_test, data_val, intins, sinint, int_entities_to_keep

################################ End
input.start_end_file("end", start, file=__file__.split("/")[-1])
