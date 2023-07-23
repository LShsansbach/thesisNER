################################### NER mit Spacy trainieren ####################################
import input

start = input.start_end_file(file=__file__.split("/")[-1])
################################ Start

# import libraries
import subprocess
import shutil

# input
dataset = input.dataset
trf = input.trf
model = input.model
crf_on = input.crf_on
transformer_on = input.transformer_on
list_for_json_crf = []
list_for_json_trf = []
metalist_for_json_crf = []
metalist_for_json_trf = []


# function
def command_for_spacy(
    dataset, var_list, meta_list, additional="", trf="_crf", command="train"
):
    print(f">>>>>>>>>>>>>>>>>>> {command} {additional}{dataset} <<<<<<<<<<<<<<<<<<<<<<")
    start_command = input.start_end_file(file="Train Spacy")
    if trf == "_crf":
        output = subprocess.check_output(
            # f"python -m spacy {command} ../SpaCy/config{trf_add}.cfg --output ../result/model{trf}/{additional}{dataset} --paths.train ../dataset/SpaCy/{additional}{dataset}_train.spacy --paths.dev ../dataset/SpaCy/{dataset}_test.spacy{trf_two}"
            [
                "python",
                "-m",
                "spacy",
                command,
                f"../SpaCy/config.cfg",
                "--output",
                f"../result/model{trf}/{additional}{dataset}",
                "--paths.train",
                f"../dataset/SpaCy/{additional}{dataset}_train.spacy",
                "--paths.dev",
                f"../dataset/SpaCy/{dataset}_test.spacy",
            ]
        )
    else:
        output = subprocess.check_output(
            # f"python -m spacy {command} ../SpaCy/config{trf_add}.cfg --output ../result/model{trf}/{additional}{dataset} --paths.train ../dataset/SpaCy/{additional}{dataset}_train.spacy --paths.dev ../dataset/SpaCy/{dataset}_test.spacy{trf_two}"
            [
                "python",
                "-m",
                "spacy",
                command,
                f"../SpaCy/config{trf}.cfg",
                "--output",
                f"../result/model{trf}/{additional}{dataset}",
                "--paths.train",
                f"../dataset/SpaCy/{additional}{dataset}_train.spacy",
                "--paths.dev",
                f"../dataset/SpaCy/{dataset}_test.spacy",
                "--gpu-id",
                "0",
            ]
        )
    var_list.append(
        {
            "Output": output.decode(),
            "Dauer": input.start_end_file("end", start_command, file="Train Spacy"),
        }
    )
    meta_list.append(
        input.load_json(f"../result/model{trf}/{additional}{dataset}/{model}/meta.json")
    )
    return var_list, meta_list


# ausführen
################################################### CRF
if crf_on:
    shutil.copy(r"../SpaCy/config.cfg", r"../reporting/config.cfg")
    for word in dataset:
        list_for_json_crf, metalist_for_json_crf = command_for_spacy(
            word, list_for_json_crf, metalist_for_json_crf
        )
        # Snorkel
        list_for_json_crf, metalist_for_json_crf = command_for_spacy(
            word, list_for_json_crf, metalist_for_json_crf, additional="snorkel_"
        )
        # Active Learning
        list_for_json_crf.append([None])
        metalist_for_json_crf.append([None])
        # Data Augmentation
        list_for_json_crf, metalist_for_json_crf = command_for_spacy(
            word, list_for_json_crf, metalist_for_json_crf, additional="da_"
        )
        # Transfer Learning
        list_for_json_crf, metalist_for_json_crf = command_for_spacy(
            word, list_for_json_crf, metalist_for_json_crf, additional="transfer_"
        )
        # Combination
        list_for_json_crf, metalist_for_json_crf = command_for_spacy(
            word, list_for_json_crf, metalist_for_json_crf, additional="comb_"
        )
    input.update_json(list_for_json_crf, "Train_CRF")
    input.update_json(metalist_for_json_crf, "Meta_Train_CRF")

################################################### Transformer
if transformer_on:
    shutil.copy(r"../SpaCy/config_trf.cfg", r"../reporting/config_trf.cfg")
    for word in dataset:
        list_for_json_trf, metalist_for_json_trf = command_for_spacy(
            word, list_for_json_trf, metalist_for_json_trf, trf=trf
        )
        # Snorkel
        list_for_json_trf, metalist_for_json_trf = command_for_spacy(
            word,
            list_for_json_trf,
            metalist_for_json_trf,
            additional="snorkel_",
            trf=trf,
        )
        # Active Learning
        list_for_json_trf.append([None])
        metalist_for_json_trf.append([None])
        # Data Augmentation
        list_for_json_trf, metalist_for_json_trf = command_for_spacy(
            word,
            list_for_json_trf,
            metalist_for_json_trf,
            additional="da_",
            trf=trf,
        )
        # Transfer Learning
        list_for_json_trf, metalist_for_json_trf = command_for_spacy(
            word,
            list_for_json_trf,
            metalist_for_json_trf,
            additional="transfer_",
            trf=trf,
        )
        # Combination
        list_for_json_trf, metalist_for_json_trf = command_for_spacy(
            word,
            list_for_json_trf,
            metalist_for_json_trf,
            additional="comb_",
            trf=trf,
        )
    input.update_json(list_for_json_trf, "Train_TRF")
    input.update_json(metalist_for_json_trf, "Meta_Train_TRF")

################################ End
input.start_end_file("end", start, file=__file__.split("/")[-1])
