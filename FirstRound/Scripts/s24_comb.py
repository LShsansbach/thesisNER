###################################### Einsatz von Data Augmentation ######################################
import input

start = input.start_end_file(file=__file__.split("/")[-1])
################################ Start

# import
import pandas as pd
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel
import os
import random
from nltk.corpus import stopwords
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

##################### input
comb_active = input.comb_active
dataset = input.dataset
label = input.label  # label_spec
text_sentence = input.text
dict_intins = input.dict_intins
synonym_path = input.synonym_path
random_state = input.random_state
dict_sinint = [{v: k for k, v in d.items()} for d in dict_intins]
random.seed(random_state)
list_for_json = []

# snorkel
n_epochs = input.snorkel_epochs
log_freq = input.snorkel_log_freq
seed = input.snorkel_seed
snorkel_label = input.snorkel_label
list_cardinality = [len(list(i.keys())) for i in dict_intins]
count_lf = 0
transfer_list = pd.DataFrame()

# data augmentation
da_tokens = input.da_tokens
da_count = input.da_count
signs = input.da_noSynonyms
number_of_synReplacementCycles = input.da_number_of_synReplacementCycles
# https://www.openthesaurus.de/about/api
stop_words = stopwords.words("german")
# warnings.filterwarnings("ignore")

# transfer learning
language_model = input.transfer_language_model
tokenizer = AutoTokenizer.from_pretrained(language_model, is_split_into_words=True)
model = AutoModelForTokenClassification.from_pretrained(language_model)


##################### functions
########### snorkel
# labeling functions
@labeling_function()
def lf_transfer_name(x):
    global count_lf
    global transfer_list
    count_lf += 1
    return (
        PERSON
        if transfer_list[label][count_lf - 1] == ("B-PER" or "I-PER")
        else ABSTAIN
    )


@labeling_function()
def lf_transfer_location(x):
    global count_lf
    global transfer_list
    # transfer_entity = transfer_list[label][count_lf - 1]
    # return LOCATION if transfer_entity == ("B-LOC" or "I-LOC") else ABSTAIN
    return (
        LOCATION
        if transfer_list[label][count_lf - 1] == ("B-LOC" or "I-LOC")
        else ABSTAIN
    )


@labeling_function()
def lf_transfer_organization(x):
    global count_lf
    global transfer_list
    return (
        ORGANIZATION
        if transfer_list[label][count_lf - 1]
        == ("B-ORG" or "B-Org" or "I-ORG" or "I-Org")
        else ABSTAIN
    )


@labeling_function()
def lf_transfer_nonEntity(x):
    global count_lf
    global transfer_list
    return NONENTITY if transfer_list[label][count_lf - 1] == "O" else ABSTAIN


@labeling_function()
def lf_lookup_nonEntity(x):
    return NONENTITY if x.text in file[4] else ABSTAIN


@labeling_function()
def lf_lookup_firstname(x):
    return PERSON if x.text in file[7] else ABSTAIN


@labeling_function()
def lf_lookup_lastname(x):
    return PERSON if x.text in file[5] else ABSTAIN


@labeling_function()
def lf_lookup_city(x):
    return LOCATION if x.text in file[6] else ABSTAIN


@labeling_function()
def lf_lookup_country(x):
    return LOCATION if x.text in file[3] else ABSTAIN


@labeling_function()
def lf_lookup_street(x):
    return LOCATION if x.text in file[0] else ABSTAIN


@labeling_function()
def lf_lookup_building(x):
    return LOCATION if x.text in file[1] else ABSTAIN


@labeling_function()
def lf_lookup_organization(x):
    return ORGANIZATION if x.text in file[2] else ABSTAIN


lfs = [
    lf_transfer_name,
    lf_transfer_location,
    lf_transfer_organization,
    lf_transfer_nonEntity,
    lf_lookup_nonEntity,
    lf_lookup_firstname,
    lf_lookup_lastname,
    lf_lookup_city,
    lf_lookup_country,
    lf_lookup_street,
    lf_lookup_building,
    lf_lookup_organization,
]


# Snorkel functions
def init_lf(sinint):
    ABSTAIN = -1  # sinint["O"]
    PERSON = sinint["B-PER"]
    LOCATION = sinint["B-LOC"]
    NONENTITY = sinint["O"]
    try:
        ORGANIZATION = sinint["B-ORG"]
    except:
        ORGANIZATION = sinint["B-Org"]
    return ABSTAIN, PERSON, LOCATION, ORGANIZATION, NONENTITY


def get_flat_df(df: pd.DataFrame, tokens: str = "text", label: str = label):
    df_flat_tokens = df[tokens].explode(ignore_index=True).reset_index()
    df_flat_label = df[label].explode(ignore_index=True).reset_index()
    return df_flat_tokens.merge(df_flat_label).drop(columns=["index"])


def bilou_from_list(df: pd.DataFrame, text_name="text", label_name=label):
    for num, i in enumerate(df[label_name]):
        assert len(df[text_name][num]) == len(
            df[label_name][num]
        ), f"df text: {len(df[text_name][num])}; df label: {len(df[label_name][num])}"
    return get_flat_df(df, text_name, label_name)


def list_from_bilou(
    df_bilou, df_original, text_name=text_sentence, label_name=snorkel_label
):
    df_list = []
    num_temp = 0
    for list_word in df_original[text_name]:
        df_list.append(
            [
                df_bilou[label_name].iloc[i]
                for i in range(num_temp, num_temp + len(list_word))
            ]
        )
        num_temp += len(list_word)
    df_original[label_name] = df_list
    return transform_labels(df_original)


def train_snorkel(df, cardinality, lfs, text_name=text_sentence, label_name=label):
    print("Weak Supervision")
    df = pd.DataFrame(df).rename(columns={text_name: "text"})
    df_bilou = bilou_from_list(df, label_name=label_name)
    assert len(df_bilou) == len(  ###############################
        transfer_list
    ), f"df bilou: {len(df_bilou)}; Transfer list: {len(transfer_list)}"
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_bilou)
    analysis = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
    label_model = LabelModel(cardinality=cardinality, verbose=True)
    label_model.fit(L_train, n_epochs=n_epochs, log_freq=log_freq, seed=seed)
    return predict_snorkel(df_bilou, label_model, L_train, bool_outer=False), analysis


def predict_snorkel(df, label_model, L_train, text_name=text_sentence, bool_outer=True):
    if bool_outer:
        df = pd.DataFrame(df[text_name]).rename(columns={text_name: "text"})
        for num, i in enumerate(df["text"]):
            df["text"][num] = str(df["text"][num])
    df[snorkel_label] = label_model.predict(L=L_train, tie_break_policy="abstain")
    return df


def transform_labels(df):
    # df[label_name] = df[label_name].apply(lambda x: [y.replace("B-", "").replace("I-", "").replace("L-", "") for y in x])
    for num, row in enumerate(df[snorkel_label]):
        ent_prev = sinint["O"]
        for num2, i in enumerate(row):
            if i == ent_prev and i != sinint["O"]:
                df[snorkel_label][num][num2] = sinint["I-" + intins[i].split("-")[1]]
            ent_prev = i
    return df


########### Active Learning
# Nothing


########### Data Augmentation
def switch_entities_from_external_list(df: pd.DataFrame, label: str, iterations: int):
    print("Switch Entities")
    df_new = df[[label, text_sentence]]
    df_new[da_tokens] = df_new[text_sentence]
    for i in tqdm(range(iterations)):
        int_help = random.choice(df.index)
        label_row = df.loc[int_help][label]
        list_help = []
        for num, int_label in enumerate(label_row):
            if int_label in label_list_PER:
                list_help.append(
                    random.choice(
                        file[
                            random.choice(
                                [i for i in range(0, 8) if i not in [0, 1, 2, 3, 4, 6]]
                            )
                        ]
                    )
                )
            elif int_label in label_list_LOC:
                list_help.append(
                    random.choice(
                        file[
                            random.choice(
                                [i for i in range(0, 8) if i not in [2, 4, 5, 7]]
                            )
                        ]
                    )
                )
            elif int_label in label_list_ORG:
                list_help.append(random.choice(file[2]))
            else:
                list_help.append(df.loc[[int_help]][text_sentence].iloc[0][num])
        df_new = pd.concat(
            [
                df_new,
                pd.DataFrame(
                    [
                        [
                            list_help,
                            df.loc[[int_help]][label].iloc[0],
                            df.loc[[int_help]][text_sentence].iloc[0],
                        ]
                    ],
                    columns=[da_tokens, label, text_sentence],
                ),
            ],
            ignore_index=True,
        )
    return synonym_replacement(df_new)


def get_synonyms(word):
    list_help = []
    for sublist in synonym_file:
        if word in sublist:
            list_help.append(sublist)
    list_synonym = []
    list_synonym = list(
        set(
            [
                item
                for sublist in list_help
                for item in sublist
                if item != word and len(item.split(" ")) == 1
            ]
        )
    )
    return list_synonym


def check_conditions(word_to_replace, count, stop_words, signs):
    if (
        count < 20
        and (word_to_replace not in stop_words)
        and (word_to_replace not in signs)
        and not any(char.isdigit() for char in word_to_replace)
    ):
        return False
    else:
        return True


def synonym_replacement(df: pd.DataFrame):
    print("Synonym Replacement")
    df.reset_index(inplace=True, drop=True)
    for row in tqdm(df.iterrows(), total=len(df)):
        sentence = row[1][text_sentence]
        word_to_replace = "."
        count = 0
        while True:
            random_number = random.randrange(len(sentence))
            word_to_replace = sentence[random_number]
            count += 1
            if not check_conditions(word_to_replace, count, stop_words, signs):
                break
            if count >= 20:
                break
        synonym = get_synonyms(word_to_replace)
        if synonym and len(synonym) != 0:
            list_row = {text_sentence: sentence, snorkel_label: row[1][snorkel_label]}
            sentence_add = sentence.copy()
            sentence_add[random_number] = random.choice(synonym).replace("\n", "")
            list_row[da_tokens] = sentence_add
            df = pd.concat([df, pd.DataFrame([list_row])], ignore_index=True)
    return df


def init_entityLists():
    label_list_PER = [key for key in intins if "per" in intins[key].lower()]
    label_list_LOC = [key for key in intins if "loc" in intins[key].lower()]
    label_list_ORG = [key for key in intins if "org" in intins[key].lower()]
    return label_list_PER, label_list_LOC, label_list_ORG


########### Transfer Learning
def init_transfer_word():
    tokenizer = AutoTokenizer.from_pretrained(language_model)
    model = AutoModelForTokenClassification.from_pretrained(language_model)
    return tokenizer, model


def init_transfer_sentence():
    tokenizer = AutoTokenizer.from_pretrained(language_model, is_split_into_words=True)
    model = AutoModelForTokenClassification.from_pretrained(language_model)
    return tokenizer, model


def transfer_learning_sentence(df: pd.DataFrame):
    print("Transfer Learning")
    tokenizer, model = init_transfer_sentence()
    result = []
    for i in tqdm(df[text_sentence]):
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
    df.reset_index(drop=True, inplace=True)
    for num, i in enumerate(result):
        if len(df[label][num]) != len(i):
            result[num] += ["O"] * (len(df[label][num]) - len(i))
    global transfer_list
    transfer_list[label] = make_one_list(result)
    transfer_list.reset_index(drop=True, inplace=True)


def make_one_list(list_of_lists: list):
    one_list = []
    for i in list_of_lists:
        for j in i:
            one_list.append(j)
    return one_list


########### Allgemein
def get_filenames():
    files = os.listdir("../dataset/")
    file = []
    for i in files:
        if ".txt" in i and i != "z_openthesaurus.txt":
            var = pd.read_csv(f"../dataset/{i}", header=None)[0].tolist()
            var = [str(z).replace(".", "").replace("-", "") for z in var]
            file.append(var)
            print(f"Lookup-Table: {i.replace('.txt', '')}, {len(var)}")
    return file


def get_synonym_file():
    with open(synonym_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        syn_list = [line.split(";") for line in lines]
    print(f"Länge z_openthesaurus.txt: {len(syn_list)}")
    # delete between brackets
    for num, i in enumerate(syn_list):
        for num2, j in enumerate(i):
            if "(" in j:
                word = j[j.find("(") : j.find(")") + 1]
                syn_list[num][num2] = j.replace(word, "").strip()
    return syn_list


def save_results(df: pd.DataFrame, dataset_choice: str, list_for_json: list):
    df.to_pickle(f"../output/df_comb_{dataset_choice}.pkl")
    df.to_csv(f"../output/df__comb_{dataset_choice}.csv")
    list_for_json.append([None])
    list_for_json.append([None])
    list_for_json.append([None])
    list_for_json.append([None])
    list_for_json.append([None])
    list_for_json.append(
        {
            "Dataset": dataset_choice,
            "Snorkel": {
                "LabelingFunctions": len(lfs),
                "Epochs": n_epochs,
                "Log_Freq": log_freq,
                "Seed": seed,
                "Applier": "PandasLFApplier",
                "Format": "Bilou",
                "Analysis": str(lf_analysis),
            },
            "Active_Learning": None,
            "Data Augmentation": {
                "Additional Data": additional_data,
                "Deleted Duplicates": del_duplicates,
                "Synonym Replacement Cycles": number_of_synReplacementCycles,
                "Entity Swapping numeber": da_count,
                "Total": len(df),
            },
            "Transfer Learning": {
                "Language Model": language_model,
                "Labels_capability": model.config.label2id,
            },
        }
    )
    return list_for_json


def determine_data(dataset_choice, cardinality):
    df_train = pd.read_pickle(f"../dataset/{dataset_choice}/train.pkl").reset_index(
        drop=True
    )
    print(
        f"Dataset: {dataset_choice}, Cardinality: {cardinality}, Rows: {len(df_train)}"
    )
    return df_train, cardinality


def run_snorkel(dataset_choice, choice):
    print("Run Snorkel and Transfer Learning")
    df_train, cardinality = determine_data(
        dataset_choice[choice],
        list_cardinality[choice],
    )
    transfer_learning_sentence(df_train)
    df_trained, lf_analys = train_snorkel(df_train, cardinality, lfs, label_name=label)
    df_trained[snorkel_label] = df_trained[snorkel_label].replace(ABSTAIN, sinint["O"])
    df_final = list_from_bilou(df_trained, df_train)
    print(df_final[snorkel_label].explode(ignore_index=True).map(intins).value_counts())
    return df_final, list_for_json, lf_analys


def run_al(df: pd.DataFrame):
    print("Run Active Learning")
    return df


def run_da(df: pd.DataFrame):
    print("Run Data Augmentation")
    print(f"Cycle 1/{number_of_synReplacementCycles}")
    df_aug = switch_entities_from_external_list(df, snorkel_label, da_count)
    for cycle in range(number_of_synReplacementCycles - 1):
        print(f"Cycle {cycle + 2}/{number_of_synReplacementCycles}")
        df_aug = switch_entities_from_external_list(
            df_aug.drop(columns=text_sentence).rename(
                columns={da_tokens: text_sentence}
            ),
            snorkel_label,
            da_count,
        )
    len_prev = len(df_aug)
    print("start deleting duplicates")
    df_aug.drop_duplicates(subset=[da_tokens], inplace=True)
    print("shuffle")
    df_aug = df_aug.sample(frac=1, random_state=random_state).reset_index(drop=True)
    del_duplicates = len_prev - len(df_aug)
    print(f"Deleted Duplicates: {del_duplicates}")
    return df_aug, list_for_json, del_duplicates


##################### ablauf
if comb_active:
    file = get_filenames()
    synonym_file = get_synonym_file()
    for num, choice in enumerate(dataset):
        intins = dict_intins[num]
        sinint = dict_sinint[num]
        # snorkel with transfer learning
        tokenizer, model = init_transfer_word()
        ABSTAIN, PERSON, LOCATION, ORGANIZATION, NONENTITY = init_lf(sinint)
        df_snorkel, list_for_json, lf_analysis = run_snorkel(dataset, num)
        # active learning
        df_al = run_al(df_snorkel)
        # data augmentation
        label_list_PER, label_list_LOC, label_list_ORG = init_entityLists()
        df_da, list_for_json, del_duplicates = run_da(df_al)
        additional_data = len(df_da) - len(df_snorkel)
        list_for_json = save_results(df_da, choice, list_for_json)
    del df_snorkel, df_al, df_da
    input.update_json(list_for_json, "Combination")

################################ End
input.start_end_file("end", start, file=__file__.split("/")[-1])
