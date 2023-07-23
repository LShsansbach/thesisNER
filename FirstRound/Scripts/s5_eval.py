################################### NER mit Spacy trainieren ####################################
import input

start = input.start_end_file(file=__file__.split("/")[-1])
################################ Start

# import libraries
import subprocess

# input
dataset = input.dataset
trf = input.trf
model = input.model
crf_on = input.crf_on
transformer_on = input.transformer_on
list_for_json_trf = []
list_for_json_crf = []


# function
def command_for_spacy(
    dataset, var_list, additional="", trf="_crf", path=dataset, command="evaluate"
):
    print(f">>>>>>>>>>>>>>>>>>> {command} {additional}{dataset} <<<<<<<<<<<<<<<<<<<<<<")
    start_command = input.start_end_file(file="Evaluate Spacy")
    if trf == "_crf":
        output = subprocess.check_output(
            # f"python -m spacy {command} ../result/model{trf}/{additional}{dataset}/{model} ../dataset/SpaCy/{path}_val.spacy"
            [
                "python",
                "-m",
                "spacy",
                command,
                f"../result/model{trf}/{additional}{path}/{model}",
                f"../dataset/SpaCy/{path}_val.spacy",
            ]
        )
    else:
        output = subprocess.check_output(
            [
                "python",
                "-m",
                "spacy",
                command,
                f"../result/model{trf}/{additional}{path}/{model}",
                f"../dataset/SpaCy/{path}_val.spacy",
                "--gpu-id",
                "0",
            ]
        )
    var_list.append(
        {
            "Output": output.decode(),
            "Dauer": input.start_end_file("end", start_command, file="Evaluate Spacy"),
        }
    )
    return var_list


# ausführen
################################################### CRF
if crf_on:
    for word in dataset:
        list_for_json_crf = command_for_spacy(word, list_for_json_crf, path=word)
        # Snorkel
        list_for_json_crf = command_for_spacy(
            word, list_for_json_crf, additional="snorkel_", path=word
        )
        # Active Learning
        list_for_json_crf.append([None])
        # Data Augmentation
        list_for_json_crf = command_for_spacy(
            word, list_for_json_crf, additional="da_", path=word
        )
        # Transfer Learning
        list_for_json_crf = command_for_spacy(
            word, list_for_json_crf, additional="transfer_", path=word
        )
        # Combination
        list_for_json_crf = command_for_spacy(
            word, list_for_json_crf, additional="comb_", path=word
        )
    input.update_json(list_for_json_crf, "Evaluation_CRF")

################################################### Transformer
if transformer_on:
    for word in dataset:
        list_for_json_trf = command_for_spacy(
            word, list_for_json_trf, trf=trf, path=word
        )
        # Snorkel
        list_for_json_trf = command_for_spacy(
            word,
            list_for_json_trf,
            additional="snorkel_",
            trf=trf,
            path=word,
        )
        # Active Learning
        list_for_json_trf.append([None])
        # Data Augmentation
        list_for_json_trf = command_for_spacy(
            word,
            list_for_json_trf,
            additional="da_",
            trf=trf,
            path=word,
        )
        # Transfer Learning
        list_for_json_trf = command_for_spacy(
            word,
            list_for_json_trf,
            additional="transfer_",
            trf=trf,
            path=word,
        )
        # Combination
        list_for_json_trf = command_for_spacy(
            word,
            list_for_json_trf,
            additional="comb_",
            trf=trf,
            path=word,
        )
    input.update_json(list_for_json_trf, "Evaluation_TRF")

################################ End
input.start_end_file("end", start, file=__file__.split("/")[-1])
