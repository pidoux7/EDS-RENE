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

    scorer = create_ner_exact_scorer(nlp.pipes.ner.target_span_getter)

    optimizer = torch.optim.AdamW(
        params=nlp.parameters(),
        lr=lr,
    )

    iterator = chain.from_iterable(repeat(dataloader))

    # Looping through the training data
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

        # Evaluating the model
        if (step % 100) == 0:
            with nlp.select_pipes(enable=["ner"]):  #
                print(scorer(val_docs, nlp.pipe(deepcopy(val_docs))))  #

        nlp.to_disk("/home/pidoux/LIMICS/EDS-RENE/model/")


if __name__ == "__main__":
    app()
