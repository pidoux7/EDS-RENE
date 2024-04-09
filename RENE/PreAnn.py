from copy import deepcopy
from itertools import chain, repeat
from typing import Callable, Iterable

import torch
from confit import Cli
from pydantic import DirectoryPath
from spacy.tokens import Doc
from tqdm import tqdm

import edsnlp
from edsnlp import registry, Pipeline
from edsnlp.scorers.ner import create_ner_exact_scorer
from typing import List
import os
import edsnlp.pipes as eds
from edsnlp.connectors.brat import BratConnector


def txt_liste(dossier: str):
    """Extract the text from a folder of txt files and conserve the name of the files

    Args:
        dossier (str): repertory of the txt files

    Returns:
        tuple : list of the text of the files, list of the name of the files
    """
    corpus = []
    files_name = []
    for fichier in os.listdir(dossier):
        chemin = os.path.join(dossier, fichier)
        if os.path.isfile(chemin) and fichier.endswith(".txt"):
            with open(chemin, "r", encoding="utf-8") as f:
                corpus.append(f.read())
                files_name.append(fichier[:-4])
    return corpus, files_name


def pipeline_date(corpus: List[str]):
    """prepare and execute date extraction pipeline

    Args:
        corpus (List[str]): list of text

    Returns:
        List[eds.doc]: list of text with dates
    """
    docs = edsnlp.data.from_iterable(corpus)
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(eds.sentences())
    nlp.add_pipe(eds.normalizer())
    nlp.add_pipe(eds.dates())
    pred_iterator = docs.map_pipeline(nlp)
    return list(pred_iterator)


def prepare_brat(pred_docs: List, files_name: List[str], predicted_rep: str):
    """create the brat files from the predicted documents

    Args:
        pred_docs (List): list of the predicted documents
        files_name (List[str]): list of the name of the files
        predicted_rep (str): repertory where the brat files will be created
    """
    brat = BratConnector(predicted_rep)
    for i, doc in enumerate(pred_docs):
        doc._.note_id = files_name[i]
        doc.spans["pollutions"] = []
    brat.docs2brat(pred_docs)


def fusion_ann(
    original_rep: str, predicted_rep: str, merged_rep: str, files_name: List[str]
):
    """Fusionne les annotations de deux fichiers .ann en un seul fichier .ann

    Args:
        original_rep (str): source directory of the original annotations
        predicted_rep (str): source directory of the predicted annotations
        merged_rep (str): destination directory of the merged annotations
        files_name (List[str]): list of file names to merge
    """
    # Iterate over the files
    for file_name in tqdm(files_name):
        with open(original_rep + file_name + ".ann", "r", encoding="utf-8") as file:
            lines_file_original = file.readlines()

        # Extract max existing id
        existing_ids = max(
            [int(line.split("\t")[0][1:]) for line in lines_file_original]
        )

        # Lire le second fichier et préparer les nouvelles lignes
        new_lines = []
        with open(predicted_rep + file_name + ".ann", "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue  # Skip lines with wrong format
                # Replace 'date' par 'Temporal'
                if "date" in parts[1]:
                    parts[1] = parts[1].replace("date", "Temporal")
                # Generate new identifier
                parts[0] = "T" + str(existing_ids + 1)
                existing_ids += 1
                new_line = "\t".join(parts) + "\n"
                new_lines.append(new_line)

        # Merge lines from the original file with the new lines
        merged_lines = lines_file_original + new_lines

        # Write the merged file
        with open(merged_rep + file_name + ".ann", "w", encoding="utf-8") as file:
            file.writelines(merged_lines)


@registry.adapters.register("ner_adapter")
def ner_adapter(
    path: DirectoryPath,
    skip_empty: bool = False,
):
    def generator(nlp):
        # Read the data from the brat directory and convert it into Docs,
        docs = edsnlp.data.read_standoff(
            path,
            # Store spans in default "ents", and "ml-ner" for the training
            span_setter=["ents", "ml-ner"],
            # Tokenize the training docs with the same tokenizer as the trained model
            tokenizer=nlp.tokenizer,
        )
        for doc in docs:
            if skip_empty and len(doc.ents) == 0:
                continue
            doc.spans["ml-ner"] = doc.ents
            yield doc

    return generator


app = Cli(pretty_exceptions_show_locals=False)


@app.command(name="train", registry=registry)  #
def train(
    nlp: Pipeline,
    train_adapter: Callable[[Pipeline], Iterable[Doc]],
    val_adapter: Callable[[Pipeline], Iterable[Doc]],
    max_steps: int = 1000,
    seed: int = 42,
    lr: float = 3e-4,
    batch_size: int = 4,
):
    # Adapting a dataset
    train_docs = list(train_adapter(nlp))
    val_docs = list(val_adapter(nlp))

    # Complete the initialization with the training data
    nlp.post_init(train_docs)

    # Preprocessing the data
    preprocessed = list(
        nlp.preprocess_many(
            train_docs,
            supervision=True,
        )
    )
    dataloader = torch.utils.data.DataLoader(
        preprocessed,
        batch_size=batch_size,
        collate_fn=nlp.collate,
        shuffle=True,
    )

    scorer = create_ner_exact_scorer(nlp.get_pipe("ner").target_span_getter)

    optimizer = torch.optim.AdamW(
        params=nlp.parameters(),
        lr=lr,
    )

    iterator = chain.from_iterable(repeat(dataloader))

    # Looping through the training data
    best_score = 0
    for step in tqdm(range(max_steps), "Training model", leave=True):
        batch = next(iterator)
        optimizer.zero_grad()

        loss = torch.zeros((), device="cpu")
        with nlp.cache():
            for name, component in nlp.torch_components():
                output = component.module_forward(batch[name])
                if "loss" in output:
                    loss += output["loss"]

        loss.backward()

        optimizer.step()
        nlp.to_disk("model")

        # Evaluating the model

        with nlp.select_pipes(enable=["ner"]):  #
            score = scorer(val_docs, nlp.pipe(deepcopy(val_docs)))

        if (step % 10) == 0:
            print("Score Temporal : ", score["Temporal"])

        if float(score["Temporal"]["f"]) <= float(best_score):
            break

        best_score = score["Temporal"]["f"]


@app.command(name="test", registry=registry)  #
def test(
    nlp: Pipeline,
    test_adapter: Callable[[Pipeline], Iterable[Doc]],
    batch_size: int = 8,
):
    # Adapting a dataset
    test_docs = list(test_adapter(nlp))

    scorer = create_ner_exact_scorer(nlp.get_pipe("ner").target_span_getter)

    # Looping through the training
    with nlp.select_pipes(enable=["ner"]):  #
        pred = nlp.pipe(deepcopy(test_docs))
        score = scorer(test_docs, pred)
        print(score["Temporal"])
        res = list(pred)
        cpt = 0
        for ent in res[0].ents:
            if ent.label_ == "Temporal":
                print(ent.text, ent.label_)
                cpt += 1
        print(cpt)


if __name__ == "__main__":
    original_rep = "/home/pidoux/LIMICS/brat/data/RENE/"
    predicted_rep = "/home/pidoux/LIMICS/brat/data/test/"
    merged_rep = "/home/pidoux/LIMICS/brat/data/merge/"

    doc_iterator = edsnlp.data.read_standoff(
        original_rep,
        span_setter={"ents": "Temporal"},
    )

    true_docs = list(doc_iterator)

    corpus, files_name = txt_liste(original_rep)

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        "eds.ner_crf",
        name="ner",
        config={
            "mode": "joint",
            "target_span_getter": "ml-ner",
            "window": 1,
            "embedding": {
                "@factory": "eds.text_cnn",
                "kernel_sizes": [3],
                "embedding": {
                    "@factory": "eds.transformer",  # Embedding
                    "model": "prajjwal1/bert-tiny",  # BERT model
                },
            },
        },
    )

    train(
        nlp=nlp,
        train_adapter=ner_adapter("/home/pidoux/LIMICS/brat/data/RENE/"),
        val_adapter=ner_adapter("/home/pidoux/LIMICS/brat/data/RENE/"),
        max_steps=100,
        seed=42,
        lr=3e-4,
        batch_size=3,
    )

    pred_docs = test(
        nlp=nlp,
        test_adapter=ner_adapter("/home/pidoux/LIMICS/brat/data/RENE/"),
        batch_size=3,
    )

    # prepare_brat(pred_docs, files_name, predicted_rep)

    # fusion_ann(original_rep, predicted_rep, merged_rep, files_name)
