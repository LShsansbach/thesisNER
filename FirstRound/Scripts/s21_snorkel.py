###################################### Einsatz von Snorkel ######################################
import input

start = input.start_end_file(file=__file__.split("/")[-1])
################################ Start

# import libraries
import pandas as pd
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel
import os

# input
snorkel_active = input.snorkel_active
dataset_choice = input.dataset
label_spec = input.label
intins = input.dict_intins
n_epochs = input.snorkel_epochs
log_freq = input.snorkel_log_freq
seed = input.snorkel_seed
text_sentence = input.text
snorkel_label = input.snorkel_label
list_for_json = []
sinint = [{v: k for k, v in d.items()} for d in intins]
list_cardinality = [len(list(i.keys())) for i in intins]


# labeling functions
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
    lf_lookup_nonEntity,
    lf_lookup_firstname,
    lf_lookup_lastname,
    lf_lookup_city,
    lf_lookup_country,
    lf_lookup_street,
    lf_lookup_building,
    lf_lookup_organization,
]


# functions
def determine_data(dataset_choice, cardinality):
    df_train = pd.read_pickle(f"../dataset/{dataset_choice}/train.pkl").reset_index(
        drop=True
    )
    print(
        f"Dataset: {dataset_choice}, Cardinality: {cardinality}, Rows: {len(df_train)}"
    )
    return df_train, cardinality


def init_lf(sinint):
    ABSTAIN = -1
    PERSON = sinint["B-PER"]
    LOCATION = sinint["B-LOC"]
    NONENTITY = sinint["O"]
    try:
        ORGANIZATION = sinint["B-ORG"]
    except:
        ORGANIZATION = sinint["B-Org"]
    return ABSTAIN, PERSON, LOCATION, ORGANIZATION, NONENTITY


def get_flat_df(df: pd.DataFrame, tokens: str = "text", label: str = label_spec):
    df_flat_tokens = df[tokens].explode(ignore_index=True).reset_index()
    df_flat_label = df[label].explode(ignore_index=True).reset_index()
    return df_flat_tokens.merge(df_flat_label).drop(columns=["index"])


def bilou_from_list(df: pd.DataFrame, text_name="text", label_name=label_spec):
    assert len(df[text_name]) == len(
        df[label_name]
    ), f"df text: {len(df[text_name])}; df label: {len(df[label_name])}"
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


def train_snorkel(df, cardinality, lfs, text_name=text_sentence, label_name=label_spec):
    df = pd.DataFrame(df).rename(columns={text_name: "text"})
    df_bilou = bilou_from_list(df, label_name=label_name)
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
    """Transforms the B- to I-Labels"""
    for num, row in enumerate(df[snorkel_label]):
        ent_prev = sinint[choice]["O"]
        for num2, i in enumerate(row):
            if i == ent_prev and i != sinint[choice]["O"]:
                df[snorkel_label][num][num2] = sinint[choice][
                    "I-" + intins[choice][i].split("-")[1]
                ]
            ent_prev = i
    return df


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


def save_results(df: pd.DataFrame, dataset_choice: str, list_for_json: list):
    df.to_pickle(f"../output/df_{dataset_choice}.pkl")
    df.to_csv(f"../output/df_{dataset_choice}.csv")
    list_for_json.append([None])
    list_for_json.append(
        {
            "LabelingFunctions": len(lfs),
            "Epochs": n_epochs,
            "Log_Freq": log_freq,
            "Seed": seed,
            "Applier": "PandasLFApplier",
            "Format": "Bilou",
            "Analysis": str(lf_analysis),
        }
    )
    list_for_json.append([None])
    list_for_json.append([None])
    list_for_json.append([None])
    list_for_json.append([None])
    return list_for_json


# ausführen
if snorkel_active:
    file = get_filenames()
    for choice in range(len(dataset_choice)):
        df_train, cardinality = determine_data(
            dataset_choice[choice],
            list_cardinality[choice],
        )
        ABSTAIN, PERSON, LOCATION, ORGANIZATION, NONENTITY = init_lf(sinint[choice])
        df_trained, lf_analysis = train_snorkel(
            df_train, cardinality, lfs, label_name=label_spec
        )
        df_trained[snorkel_label] = df_trained[snorkel_label].replace(
            ABSTAIN, sinint[choice]["O"]
        )
        df_final = list_from_bilou(df_trained, df_train)
        print(
            df_final[snorkel_label]
            .explode(ignore_index=True)
            .map(intins[choice])
            .value_counts()
        )
        list_for_json = save_results(df_final, dataset_choice[choice], list_for_json)
    del df_train, df_trained, df_final
    input.update_json(list_for_json, "Snorkel_Data")

################################ End
input.start_end_file("end", start, file=__file__.split("/")[-1])
