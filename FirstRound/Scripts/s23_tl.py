###################################### Einsatz von Transfer Learning ######################################
import input

start = input.start_end_file(file=__file__.split("/")[-1])
################################ Start

# import libraries
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from tqdm import tqdm
import torch

# input
dataset = input.dataset
label_spec = input.label
execute_transfer_learning = input.transfer_learning_active
language_model = input.transfer_language_model
text_sentences = input.text
transfer_label = input.transfer_label
dict_intins = input.dict_intins
dict_sinint = [{v: k for k, v in d.items()} for d in dict_intins]
tokenizer = AutoTokenizer.from_pretrained(language_model, is_split_into_words=True)
model = AutoModelForTokenClassification.from_pretrained(language_model)
list_for_json = []
sinint_model = model.config.label2id


# function
def init_transfer():
    tokenizer = AutoTokenizer.from_pretrained(
        language_model,
        is_split_into_words=True,
        padding=True,
        truncation=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(language_model)
    return tokenizer, model


def transfer_learning(df: pd.DataFrame):
    tokenizer, model = init_transfer()
    result = []
    for i in tqdm(df[text_sentences]):
        tokenized_input = tokenizer(
            i,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
        )
        tokenized_tensor = tokenizer(
            i,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            logits = model(**tokenized_tensor).logits
        predicted_token_class_ids = logits.argmax(-1)
        predicted_tokens_classes = [
            model.config.id2label[t.item()] for t in predicted_token_class_ids[0]
        ]
        int_help = 0
        list_help = []
        list_label = []
        for word in tokenized_input.word_ids():
            if word == None or word in list_help:
                int_help += 1
            else:
                list_help.append(word)
                list_label.append(predicted_tokens_classes[int_help])
                int_help += 1
        result.append(list_label)
    df[transfer_label] = result
    return df


def save_results(df: pd.DataFrame, choice: str, list_for_json: list):
    df.to_pickle(f"../output/df_{choice}.pkl")
    df.to_csv(f"../output/df_{choice}.csv")
    list_for_json.append([None])
    list_for_json.append([None])
    list_for_json.append([None])
    list_for_json.append([None])
    list_for_json.append(
        {
            "Language_Model": language_model,
            "Labels_capability": model.config.label2id,
        }
    )
    list_for_json.append([None])
    return list_for_json


# Ablauf
if execute_transfer_learning:
    for num, choice in enumerate(dataset):
        sinint = dict_sinint[num]
        df = pd.read_pickle(f"../output/df_{choice}.pkl")
        print(f"Dataset: {choice}, Rows: {len(df)}")
        df_trained = transfer_learning(df)

        list_help = []
        for i in df_trained[transfer_label]:
            list_help_2 = []
            for j in i:
                list_help_2.append(sinint[j])
            list_help.append(list_help_2)
        df[transfer_label] = list_help

        list_for_json = save_results(df, choice, list_for_json)
    del df, df_trained, list_help, list_help_2
    input.update_json(list_for_json, "Transfer_Data")

################################ End
input.start_end_file("end", start, file=__file__.split("/")[-1])
