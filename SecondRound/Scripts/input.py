################################ Adjust relevant data ################################


test_index = 0
test_sentence = "Hallo ich heiße Andi Max aus Spanien"

snorkel_active = True
da_active = True
transfer_learning_active = True
comb_active = True

model = "model-best"  # "model-last" # "model-final"
crf_on = False
transformer_on = True

entities_to_keep = ["PER", "LOC", "ORG", "Org"]
random_state = 42
number_of_data = None  # None means all data


################################ Add relevant data ################################


dataset = ["germaner"]
dataset2 = ["germeval_14"]
dataset3 = ["Babelscape/wikineural"]
label = "ner_tags"
text = "tokens"
dict_intins = [
    {
        0: "B-LOC",
        1: "B-ORG",
        2: "B-OTH",
        3: "B-PER",
        4: "I-LOC",
        5: "I-ORG",
        6: "I-OTH",
        7: "I-PER",
        8: "O",
    },
]
dict_intins2 = [
    {
        0: "O",
        1: "B-LOC",
        2: "I-LOC",
        3: "B-LOCderiv",
        4: "I-LOCderiv",
        5: "B-LOCpart",
        6: "I-LOCpart",
        7: "B-ORG",
        8: "I-ORG",
        9: "B-ORGderiv",
        10: "I-ORGderiv",
        11: "B-ORGpart",
        12: "I-ORGpart",
        13: "B-PER",
        14: "I-PER",
        15: "B-PERderiv",
        16: "I-PERderiv",
        17: "B-PERpart",
        18: "I-PERpart",
        19: "B-OTH",
        20: "I-OTH",
        21: "B-OTHderiv",
        22: "I-OTHderiv",
        23: "B-OTHpart",
        24: "I-OTHpart",
    },
]
dict_intins3 = [
    {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-ORG": 3,
        "I-ORG": 4,
        "B-LOC": 5,
        "I-LOC": 6,
        "B-MISC": 7,
        "I-MISC": 8,
    }
]


################################ unchanged data ################################


trf = "_trf"
json_file = "../reporting/data.json"
log = "../reporting/hist.log"
synonym_path = "../dataset/z_openthesaurus.txt"
language = "de"


################################ snorkel data ################################


snorkel_epochs = 1000
snorkel_log_freq = 100
snorkel_seed = random_state
snorkel_label = "label_snorkel"


################################ data augmentation data ################################


da_label = "label_augmentation"
da_tokens = "tokens_augmentation"
da_count = 10000
da_sample_size_percent = 100
da_number_of_synReplacementCycles = 8
da_noSynonyms = [
    ".",
    ",",
    "*",
    "'",
    "!",
    "?",
    "",
    '"',
    "-",
    "(",
    ")",
    ":",
    "_",
    ";",
    "%",
    "*",
    "**",
    "''",
    "/",
    "\\",
    "=",
    "“",
    "”",
    "«",
    "»",
    "–",
    "­",
    "в",
    "§",
    "‘",
    "’",
    "άποψη",
    "£",
    "&",
    "+",
    "„",
    "<",
    ">",
]


################################ transfer learning data ################################


# https://huggingface.co/mschiesser/ner-bert-german
transfer_language_model = "mschiesser/ner-bert-german"
transfer_label = "label_transfer"


################################ relevant function ################################


def load_model(
    data: str,
    satz: str,
    type_of_model: str = "crf",
    pre_direction: str = "../result/model_",
    display: bool = True,
    return_nlp: bool = False,
    return_result: bool = False,
):
    import spacy
    from spacy import displacy

    nlp = spacy.load(f"{pre_direction}{type_of_model}/{data}/{model}")
    doc = nlp(satz)
    if display:
        print(f"Modell: {type_of_model} {data}")
        print(f"Satz: {satz}")
        print(f"Alle Entitäten: {doc.ents}")
        for num, ent in enumerate(doc.ents):
            print(f"Entität: {str(num + 1)}: {ent.text} {ent.label_}")
        displacy.render(
            doc,
            style="ent",
        )
    if return_result:
        dict_for_json = {
            "Dataset": data,
            "Modell": type_of_model,
            "Satz": satz,
            "Entities": str(doc.ents),
            "AnzahlEntities": len(doc.ents),
            "EntitiesLabel": [
                f"Entity {str(num + 1)}: {str(ent.text)}, {str(ent.label_)}"
                for num, ent in enumerate(doc.ents)
            ],
        }
    if return_nlp and not return_result:
        return nlp
    elif return_result and not return_nlp:
        return dict_for_json
    elif return_result and return_nlp:
        return nlp, dict_for_json


def write_log(first_part: str, second_part: str = "", log: str = log) -> None:
    import datetime

    with open(log, "a") as f:
        f.write(
            str(datetime.datetime.now()) + ": " + first_part + " " + second_part + "\n"
        )


def write_json(dict_save: dict, json_file: str = json_file) -> None:
    import json

    with open(json_file, "r") as f:
        data = json.load(f)
    data.append(dict_save)
    with open(json_file, "w") as f:
        f.write(json.dumps(data, indent=8))


def load_json(json_file: str = json_file) -> list:
    import json

    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def update_json(
    to_update: list, keyword: str = "test", json_file: str = json_file
) -> None:
    import json

    data = load_json(json_file)
    assert len(to_update) == len(
        data
    ), "list to_update must have the same length as the json file list"
    data = [{**i, keyword: to_update[num]} for num, i in enumerate(data)]
    with open(json_file, "w") as f:
        f.write(json.dumps(data, indent=8))


def start_end_file(
    option: str = "start", start: float = 0, file: str = __file__.split("\\")[-1]
) -> float:
    import time

    if option == "start":
        start = time.time()
        write_log(option, file, log)
    elif option == "end":
        end = time.time() - start
        print("%s seconds" % (end / 60))
        write_log(
            f"{option} {file}:",
            "duration {0:02}:{1:02}:{2:02}".format(
                int(end // 3600), int(end // 60) % 60, round(end % 60, 2)
            ),
            log,
        )
        start = end
    print(
        "\n",
        f"################################# {option} {file} #################################",
        "\n",
    )
    return start
