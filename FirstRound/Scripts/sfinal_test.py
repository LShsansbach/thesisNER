################################ Laden und Testen des Models ################################
import input

start = input.start_end_file(file=__file__.split("/")[-1])
################################ Start

# import
import json
import pandas as pd

# input
crf_on = input.crf_on
transofrmer_on = input.transformer_on
dict_intins = input.dict_intins
dataset = input.dataset
test_sentence = input.test_sentence
list_json_crf = []
list_json_trf = []


# functions
def test_to_json(
    i: str,
    test_sentence: str = test_sentence,
    trf: str = "crf",
    list_json: list = list_json_crf,
) -> list:
    list_json.append(
        input.load_model(i, test_sentence, return_result=True, type_of_model=trf)
    )
    list_json.append(
        input.load_model(
            f"snorkel_{i}", test_sentence, return_result=True, type_of_model=trf
        )
    )
    list_json.append([None])

    list_json.append(
        input.load_model(
            f"da_{i}", test_sentence, return_result=True, type_of_model=trf
        )
    )
    list_json.append(
        input.load_model(
            f"transfer_{i}", test_sentence, return_result=True, type_of_model=trf
        )
    )
    list_json.append(
        input.load_model(
            f"comb_{i}", test_sentence, return_result=True, type_of_model=trf
        )
    )
    return list_json


def iterate_over_json(
    json_data: list, df: pd.DataFrame, information: str, additional: str = None
) -> pd.DataFrame:
    if additional:
        list_of_keys = list(json_data[0][information][additional].keys())
        for num, i in enumerate(json_data):
            if num != 2:
                columns = {
                    column: i[information][additional][list_of_keys[num2]]
                    for num2, column in enumerate(df.columns)
                }
            df = pd.concat([df, pd.DataFrame([columns])], axis=0, ignore_index=True)
        else:
            columns = {column: "None" for column in df.columns}
            df = pd.concat([df, pd.DataFrame([columns])], axis=0, ignore_index=True)
    else:
        list_of_keys = list(json_data[0][information].keys())
        for num, i in enumerate(json_data):
            if num != 2:
                columns = {
                    column: i[information][list_of_keys[num2]]
                    for num2, column in enumerate(df.columns)
                }
            df = pd.concat([df, pd.DataFrame([columns])], axis=0, ignore_index=True)
        else:
            columns = {column: "None" for column in df.columns}
            df = pd.concat([df, pd.DataFrame([columns])], axis=0, ignore_index=True)
        df = pd.concat(
            [
                df.drop("Number of individual Entities", axis=1),
                pd.json_normalize(df["Number of individual Entities"]).rename(
                    columns={
                        str(number): intins[number] for number in range(len(intins))
                    }
                ),
            ],
            axis=1,
        )
    return df


# ablauf
with open("../reporting/data.json") as f:
    json_data = json.load(f)

for num, i in enumerate(dataset):
    intins = dict_intins[num]
    columns_metrics = [
        "F1_Score",
        "Precision",
        "Recall",
        "Ents per Type",
        "toke2vecLoss",
        "ner_loss",
    ]  # df.columns
    columns_stats = [
        "Dataset",
        "Number of Entities",
        "Number of Sentences",
        "Number of Tokens",
        "Number of Tokens with Entities",
        "Number of individual Entities",
    ]
    ###CRF
    if crf_on:
        list_json_crf = input.update_json(
            test_to_json(i, list_json=list_json_crf), "TestModel_CRF"
        )
        df_metrics = pd.DataFrame(columns=columns_metrics)
        df_metrics = iterate_over_json(
            json_data, df_metrics, "Meta_Train_CRF", additional="performance"
        )
    ###TRF
    if transofrmer_on:
        list_json_trf = input.update_json(
            test_to_json(i, trf="trf", list_json=list_json_trf), "TestModel_TRF"
        )
        df_metrics = pd.DataFrame(columns=columns_metrics)
        df_metrics = iterate_over_json(
            json_data, df_metrics, "Meta_Train_TRF", additional="performance"
        )

    df_stats = pd.DataFrame(columns=columns_stats)
    df_stats = iterate_over_json(json_data, df_stats, "Corpus_Data")
    print(df_stats.head(10))

    with pd.ExcelWriter(f"../Result_{i}.xlsx") as writer:
        df_stats.to_excel(writer, sheet_name="Stats")
        df_metrics.to_excel(writer, sheet_name="Metrics")

################################ End
input.start_end_file("end", start, file=__file__.split("/")[-1])
