#################################### Aufbereitung für Spacy #####################################
import input

start = input.start_end_file(file=__file__.split("/")[-1])
################################ Start

# import libraries
import pandas as pd
import spacy
from spacy.matcher import Matcher
from spacy.tokens import DocBin
from spacy.tokens import Span
import warnings

# input
dataset = input.dataset
label_spec = input.label
dict_intins = input.dict_intins
tokens_text = input.text
snorkel_label = input.snorkel_label
transfer_label = input.transfer_label
da_label = input.da_label
da_tokens = input.da_tokens
dict_sinint = [{v: k for k, v in d.items()} for d in dict_intins]
nlp = spacy.blank(input.language)
list_for_json = []
warnings.filterwarnings("ignore")


# functions
def create_docs(sentences, ner_tags, dict_intins=dict_intins, dict_sinint=dict_sinint):
    docs = []
    list_to_delete = []
    for num, text in enumerate(sentences):
        matcher = Matcher(nlp.vocab)
        list_labels = []
        for num2, tag in enumerate(ner_tags[num]):
            if tag != dict_sinint["O"] and text[num2] != ".":
                list_labels.append(dict_intins[tag])
                pattern = [{"TEXT": text[num2]}]
                matcher.add(dict_intins[tag], [pattern])
        text_sen = (
            " ".join(text)
            .replace(" .", ".")
            .replace(" ,", ",")
            .replace(" ?", "?")
            .replace(" !", "!")
        )
        doc = nlp(text_sen)
        matches = matcher(doc)
        try:
            doc.ents = [
                Span(doc, start, end, label=list_labels[i])
                for i, (match_id, start, end) in enumerate(matches)
            ]
            docs.append(doc)
        except:
            list_to_delete.append(num)
    return list_to_delete, docs


def try_init_label(df, label, sentences, dict_intins, dict_sinint):
    try:
        ner_tags = df[label].tolist()
        list_to_delete, docs = create_docs(
            sentences, ner_tags, dict_intins, dict_sinint
        )
    except:
        list_to_delete = []
        docs = []
    return list_to_delete, docs


def do_spacy(df, dict_intins=dict_intins, dict_sinint=dict_sinint):
    sentences = df[tokens_text].tolist()
    ner_tags = df[label_spec].tolist()
    list_to_delete_snorkel, docs_snorkel = try_init_label(
        df, snorkel_label, sentences, dict_intins, dict_sinint
    )
    list_to_delete_transfer, docs_transfer = try_init_label(
        df, transfer_label, sentences, dict_intins, dict_sinint
    )
    list_to_delete_da, docs_da = try_init_label(
        df, da_label, sentences, dict_intins, dict_sinint
    )
    list_to_delete, docs = create_docs(sentences, ner_tags, dict_intins, dict_sinint)
    set_to_delete = set(
        list_to_delete
        + list_to_delete_snorkel
        + list_to_delete_transfer
        + list_to_delete_da
    )
    return docs, set_to_delete, docs_snorkel, docs_transfer, docs_da


def save_in_json(data_spacy, save_path, tags: str = label_spec):
    tpl_list = []
    for i in data_spacy[tags]:
        for j in i:
            tpl_list.append(j)
    dict_save = {
        "dataset": save_path,
        "number_of_entities": len(set(tpl_list)),
        "number_of_sentences": len(data_spacy[tags]),
        "number_of_tokens": None,
        "number_of_tokens_with_entities": len(tpl_list),
        "form_of_entities": pd.DataFrame(tpl_list)[0].value_counts().to_dict(),
    }
    return dict_save


def loop_delete(
    df,
    df_da,
    set_del,
    intins,
    sinint,
    set_del_da,
    docs,
    docs_snorkel,
    docs_transfer,
    docs_da_train,
    docs_da,
    docs_comb,
    set_del_comb,
    df_comb,
):
    while set_del:
        df = df.drop(list(set_del))
        docs, set_del, docs_snorkel, docs_transfer = do_spacy(df, intins, sinint)[:4]
        assert len(docs) == len(docs_snorkel) and len(docs) == len(
            docs_transfer
        ), f"Unterschiedliche Länge Snorkel oder Transfer und Train docs:{len(docs)} snorkel:{len(docs_snorkel)} transfer:{len(docs_transfer)} df:{len(df)}"
        if set_del_da:
            df_da = df_da.drop(list(set_del_da.intersection(set(df_da.index))))
            docs_da, set_del_da, docs_nothin, docs_nothing2, docs_da_train = do_spacy(
                df_da, intins, sinint
            )
        if set_del_comb:
            df_comb = df_comb.drop(list(set_del_comb.intersection(set(df_comb.index))))
            docs_comb, set_del_comb = do_spacy(df_comb, intins, sinint)[:2]
    return df, docs, docs_snorkel, docs_transfer, docs_da, docs_da_train, docs_comb


def ablauf(
    df_train,
    df_test,
    df_val,
    df_da_train,
    df_comb_train,
    save_path,
    intins=dict_intins,
    sinint=dict_sinint,
):
    docs_train, set_del_train, docs_snorkel_train, docs_transfer_train = do_spacy(
        df_train, intins, sinint
    )[:4]
    docs_da, set_del_da_train, docs_nothin, docs_nothing2, docs_da_train = do_spacy(
        df_da_train, intins, sinint
    )[:5]
    docs_comb, set_del_comb_train = do_spacy(df_comb_train, intins, sinint)[:2]
    docs_test, set_del_test = do_spacy(df_test, intins, sinint)[:2]
    docs_val, set_del_val = do_spacy(df_val, intins, sinint)[:2]
    print(
        f"train_delete: {len(set_del_train)} test_delete: {len(set_del_test)} val_delete: {len(set_del_val)}"
    )
    (
        df_train,
        docs_train,
        docs_snorkel_train,
        docs_transfer_train,
        docs_da,
        docs_da_train,
        docs_comb,
    ) = loop_delete(
        df_train,
        df_da_train,
        set_del_train,
        intins,
        sinint,
        set_del_da_train,
        docs_train,
        docs_snorkel_train,
        docs_transfer_train,
        docs_da_train,
        docs_da,
        docs_comb,
        set_del_comb_train,
        df_comb_train,
    )

    # Json
    list_for_json.append(save_in_json(df_train, save_path))
    list_for_json.append(save_in_json(df_train, f"snorkel_{save_path}", snorkel_label))
    list_for_json.append([None])
    list_for_json.append(save_in_json(df_da_train, f"da_{save_path}"))
    list_for_json.append(
        save_in_json(df_train, f"transfer_{save_path}", transfer_label)
    )
    list_for_json.append(save_in_json(df_comb_train, f"comb_{save_path}"))

    # Save
    df_train.to_csv(f"../dataset/{save_path}/new_train.csv", sep="|")
    df_train.to_pickle(f"../output/new_df_{save_path}.pkl")
    DocBin(docs=docs_train).to_disk(f"../dataset/SpaCy/{save_path}_train.spacy")
    DocBin(docs=docs_test).to_disk(f"../dataset/SpaCy/{save_path}_test.spacy")
    DocBin(docs=docs_val).to_disk(f"../dataset/SpaCy/{save_path}_val.spacy")
    DocBin(docs=docs_snorkel_train).to_disk(
        f"../dataset/SpaCy/snorkel_{save_path}_train.spacy"
    )
    DocBin(docs=docs_transfer_train).to_disk(
        f"../dataset/SpaCy/transfer_{save_path}_train.spacy"
    )
    DocBin(docs=docs_da_train).to_disk(f"../dataset/SpaCy/da_{save_path}_train.spacy")
    DocBin(docs=docs_comb).to_disk(f"../dataset/SpaCy/comb_{save_path}_train.spacy")


# ausführen
for choice in range(len(dataset)):
    print(f">>>>>>>>>>>>>>>>>>> {dataset[choice]} <<<<<<<<<<<<<<<<<<<<<<")
    df_train = pd.read_pickle(f"../output/df_{dataset[choice]}.pkl")
    df_test = pd.read_pickle(f"../dataset/{dataset[choice]}/test.pkl")
    df_val = pd.read_pickle(f"../dataset/{dataset[choice]}/val.pkl")
    df_da = pd.read_pickle(f"../output/df_augmented_{dataset[choice]}.pkl")
    df_da[da_label] = df_da[label_spec]
    df_comb = pd.read_pickle(f"../output/df_comb_{dataset[choice]}.pkl").rename(
        columns={snorkel_label: label_spec, da_tokens: tokens_text, tokens_text: "da"}
    )
    ablauf(
        df_train,
        df_test,
        df_val,
        df_da,
        df_comb,
        dataset[choice],
        dict_intins[choice],
        dict_sinint[choice],
    )
del df_train, df_test, df_val, df_da, df_comb
input.update_json(list_for_json, "Corpus_Data")

################################ End
input.start_end_file("end", start, file=__file__.split("/")[-1])
