###################################### Einsatz von Data Augmentation ######################################
import input

start = input.start_end_file(file=__file__.split("/")[-1])
################################ Start

# import
import pandas as pd
import random
import os
from nltk.corpus import stopwords
from tqdm import tqdm
import warnings

# input
dataset = input.dataset
sentences = input.text
label_sentences = input.label
da_tokens = input.da_tokens
da_label = input.da_label
da_active = input.da_active
dict_intins = input.dict_intins
da_count = input.da_count
signs = input.da_noSynonyms
random_state = input.random_state
sample_size_percent = input.da_sample_size_percent
number_of_synReplacementCycles = input.da_number_of_synReplacementCycles
synonym_path = input.synonym_path
random.seed(random_state)
list_for_json = []
# https://www.openthesaurus.de/about/api
stop_words = stopwords.words("german")
warnings.filterwarnings("ignore")


# function
def get_filenames():
    files = os.listdir("../dataset/")
    file = []
    for i in files:
        if ".txt" in i and "z_" not in i:
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
        sentence = row[1][sentences]
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
            list_row = {sentences: sentence, label_sentences: row[1][label_sentences]}
            sentence_add = sentence.copy()
            sentence_add[random_number] = random.choice(synonym).replace("\n", "")
            list_row[da_tokens] = sentence_add
            # df = df.append(list_row, ignore_index=True)
            df = pd.concat([df, pd.DataFrame([list_row])], ignore_index=True)
    return df


def switch_entities_from_external_list(df: pd.DataFrame, label: str, iterations: int):
    print("Switch Entities")
    # int_init = random.choice(df.index)
    df_new = df[[label, sentences]]
    df_new[da_tokens] = df_new[sentences]
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
                list_help.append(df.loc[[int_help]][sentences].iloc[0][num])
        df_new = pd.concat(
            [
                df_new,
                pd.DataFrame(
                    [
                        [
                            list_help,
                            df.loc[[int_help]][label].iloc[0],
                            df.loc[[int_help]][sentences].iloc[0],
                        ]
                    ],
                    columns=[da_tokens, label, sentences],
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


def save_results(df: pd.DataFrame, choice: str, list_for_json: list):
    df.to_pickle(f"../output/df_augmented_{choice}.pkl")
    df.to_csv(f"../output/df_augmented_{choice}.csv")
    list_for_json.append([None])
    list_for_json.append([None])
    list_for_json.append([None])
    list_for_json.append(
        {
            "Additional Data": additional_data,
            "Deleted Duplicates": len_prev - len(df),
            "Synonym Replacement Cycles": number_of_synReplacementCycles,
            "Entity Swapping numeber": da_count,
            "Total": len_prev,
            "Percentage of Original Data": f"{sample_size_percent}%",
        }
    )
    list_for_json.append([None])
    list_for_json.append([None])
    return list_for_json


# ablauf
if da_active:
    file = get_filenames()
    synonym_file = get_synonym_file()
    for num, choice in enumerate(dataset):
        intins = dict_intins[num]
        df = pd.read_pickle(f"../output/df_{choice}.pkl")
        print("Data Original: ", len(df))
        df = df.sample(
            random_state=random_state, n=int(len(df) * (sample_size_percent / 100))
        )
        print(f"Dataset: {choice}, Percentage: {sample_size_percent}%, Rows: {len(df)}")

        label_list_PER = [key for key in intins if "per" in intins[key].lower()]
        label_list_LOC = [key for key in intins if "loc" in intins[key].lower()]
        label_list_ORG = [key for key in intins if "org" in intins[key].lower()]
        print(f"Cycle 1/{number_of_synReplacementCycles}")
        df_aug = switch_entities_from_external_list(df, label_sentences, da_count)
        for cycle in range(number_of_synReplacementCycles - 1):
            print(f"Cycle {cycle + 2}/{number_of_synReplacementCycles}")
            df_aug = switch_entities_from_external_list(
                df_aug.drop(columns=sentences).rename(columns={da_tokens: sentences}),
                label_sentences,
                da_count,
            )
        len_prev = len(df_aug)
        print("start deleting duplicates")
        df_aug.drop_duplicates(subset=[da_tokens], inplace=True)
        print(f"Deleted Duplicates: {len_prev - len(df_aug)}")
        df_aug = df_aug.sample(frac=1, random_state=random_state).reset_index(drop=True)
        additional_data = len(df_aug) - len(df)
        list_for_json = save_results(df_aug, choice, list_for_json)
    del df_aug, df
    input.update_json(list_for_json, "Aug_Data")

################################ End
input.start_end_file("end", start, file=__file__.split("/")[-1])
