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
import sys
from RENE.pre_annotation import txt_liste, prepare_brat_train, fusion_ann_train

sys.path.append("/home/pidoux/LIMICS/EDS-RENE/")




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

    return pred, score


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

    pred_docs, score = test(
        nlp=nlp,
        test_adapter=ner_adapter("/home/pidoux/LIMICS/brat/data/RENE/"),
        batch_size=3,
    )

    prepare_brat_train(pred_docs, files_name, predicted_rep)
    fusion_ann_train(original_rep, predicted_rep, merged_rep, files_name)
