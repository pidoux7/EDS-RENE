---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: lenv
    language: python
    name: python3
---

```python
%reload_ext jupyter_black

import edsnlp
from spacy.tokens import Doc, Span
from spacy import Language
from typing import List, Tuple, Dict, Any, Union, Generator
import os
import pandas as pd
```

```python
dossier_0 = "/home/pidoux/LIMICS/brat/data/merge/"
dossier_1 = "/home/pidoux/LIMICS/brat/data/RENE/"
```

```python
def ouvrir_corpus(liste_dossier: list) -> list[List[Doc]]:
    """Open a corpus of documents from a list of 2 folders

    Args:
        liste_dossier (list): List of folders containing the documents .ann and .txt

    Returns:
        list[List[Doc]]: List of documents
    """
    if len(liste_dossier) != 2:
        raise ValueError("The list should contain two folders")
    corpus = []
    for dossier in liste_dossier:
        doc_iterator = edsnlp.data.read_standoff(
            dossier,
        )
        corpus.append(list(doc_iterator))
    return corpus
```

```python
def is_match(ent_0: Span, ent_1: Span, matching_mode: str = "exact") -> bool:
    """Returns True if the predicted entity matches the ground truth entity.
    If matching_mode = "exact": The function returns True when the entity boundaries are exactly the same.
    If matching_mode = "partial" The function returns True when the entity boundaries are overlaping.

    Args:
        true_ent (Span): Ground truth entity
        pred_ent (Span): Predicted entity
        matching_mode (str): Matching mode. Either "exact" or "partial"

    Returns:
        bool: True if the predicted entity matches entity, False otherwise
    """
    start_char_0, end_char_0 = (ent_0.start_char, ent_0.end_char)
    start_char_1, end_char_1 = (ent_1.start_char, ent_1.end_char)
    if ent_0.label_ == ent_1.label_:
        if matching_mode == "exact":
            if (start_char_0 == start_char_1) or (end_char_0 == end_char_1):
                return True
            else:
                return False
        elif matching_mode == "partial":
            if (end_char_0 < start_char_1) or (start_char_0 > end_char_1):
                return False
            else:
                return True
        else:
            raise ValueError(
                f"Expecting matching_mode to be 'exact' or 'partial' and not {matching_mode}"
            )
    else:
        return False
```

```python
def evaluate(corpus_0: List[Doc], corpus_1: List[Doc]) -> Tuple[dict,dict]:


    scores_0 = {
        "exact": 0,
        "partial": 0,
        "Real_errors": 0,
        "total_ents": 0,
        "real_error": [],
    }

    scores_1 = {
        "exact": 0,
        "partial": 0,
        "Real_errors": 0,
        "total_ents": 0,
        "real_error": [],
    }

    for i, (doc_0, doc_1) in enumerate(zip(corpus_0, corpus_1)):
        for ent_0 in doc_0.ents:
            matched = False
            scores_0["total_ents"] += 1
            for ent_1 in doc_1.ents:
                if is_match(ent_0, ent_1, matching_mode="exact"):
                    scores_0["exact"] += 1
                    matched = True
                if is_match(ent_0, ent_1, matching_mode="partial"):
                    scores_0["partial"] += 1
                    matched = True
                    continue
            if not matched:
                scores_0["Real_errors"] += 1
                scores_0["real_error"].append(
                    (i, ent_0.label_, ent_0.text, ent_0.start_char, ent_0.end_char)
                )

        for ent_1 in doc_1.ents:
            scores_1["total_ents"] += 1
            matched = False
            for ent_0 in doc_0.ents:
                if is_match(ent_0, ent_1, matching_mode="exact"):
                    scores_1["exact"] += 1
                    matched = True
                if is_match(ent_0, ent_1, matching_mode="partial"):
                    scores_1["partial"] += 1
                    matched = True
                    continue
            if not matched:
                scores_1["Real_errors"] += 1
                scores_1["real_error"].append(
                    (i, ent_1.label_, ent_1.text, ent_1.start_char, ent_1.end_char)
                )

    return scores_0, scores_1
```

```python
def eval(
    true_docs: List[Doc], pred_docs: List[Doc], matching_mode: str = "exact"
) -> Dict:
    """Compare documents by documents two list of Spacy Doc and returns a dictionary with:
    - The number of True Positive (TP): predicted entities matching true entities.
    - The number of False Positive (FP): predicted entities not matching any true entities.
    - The number of False Negative (FN): true entities not matching any predicted entities.
    - The Precision (precision): the fraction of relevant instances among the retrieved instances.
    - The Recall (recall): the fraction of relevant instances that were retrieved.
    - The F1-score (f1): the harmonic mean of precision and recall.
    """
    scores = {"TP": 0, "FP": 0, "FN": 0, "FN_liste": [], "FP_liste": []}
    for i, (true_doc, pred_doc) in enumerate(zip(true_docs, pred_docs)):
        for true_ent in true_doc.ents:
            matched = False
            for pred_ent in pred_doc.ents:
                if is_match(true_ent, pred_ent, matching_mode):
                    scores["TP"] += 1
                    matched = True
                    continue
            if not matched:
                scores["FN"] += 1
                scores["FN_liste"].append(
                    (
                        i,
                        true_ent.label_,
                        true_ent.text,
                        true_ent.start_char,
                        true_ent.end_char,
                    )
                )

        for pred_ent in pred_doc.ents:
            matched = False
            for true_ent in true_doc.ents:
                if is_match(true_ent, pred_ent, matching_mode):
                    matched = True
                    continue
            if not matched:
                scores["FP"] += 1
                scores["FP_liste"].append(
                    (
                        i,
                        pred_ent.label_,
                        pred_ent.text,
                        pred_ent.start_char,
                        pred_ent.end_char,
                    )
                )

    if scores["FP"] == 0:
        scores["precision"] = 1.0
    else:
        scores["precision"] = scores["TP"] / (scores["TP"] + scores["FP"])

    if scores["FN"] == 0:
        scores["recall"] = 1.0
    else:
        scores["recall"] = scores["TP"] / (scores["TP"] + scores["FN"])

    if (scores["precision"] + scores["recall"]) == 0:
        scores["f1"] = 0
    else:
        scores["f1"] = (
            2
            * (scores["precision"] * scores["recall"])
            / (scores["precision"] + scores["recall"])
        )

    return scores
```

```python
def extract(span: Span, num_doc: int) -> Tuple:

    return (
        num_doc,
        span.start_char,
        span.end_char,
        span.text,
        span.label_,
        span._.Tech,
        span._.Negation,
        span._.Certainty,
        span._.Temporality,
        span._.Family,
        span._.AttDate,
        span._.AttTemp,
        span._.Action,
        span._.RefTemp,
    )
```

```python
def perfect_match(corpus_0: List[Doc], corpus_1: List[Doc]) -> Tuple:
    scores_total = {"true": 0, "false": 0, "total": 0}
    scores_liste = []
    error = []
    for num_doc, (doc_0, doc_1) in enumerate(zip(corpus_0, corpus_1)):
        scores = {"true": 0, "false": 0, "total": 0}
        for key_0, value_0 in doc_0.spans.items():
            for i in value_0:
                scores["total"] += 1
                scores_total["total"] += 1
                ent_0 = extract(i, num_doc)
                matched = False
                for key2, value2 in doc_1.spans.items():
                    for j in value2:
                        ent_1 = extract(j, num_doc)
                        if ent_0 == ent_1:
                            matched = True
                            scores["true"] += 1
                            scores_total["true"] += 1
                            continue
                if matched == False:
                    print(key_0, ent_0)
                    scores["false"] += 1
                    scores_total["false"] += 1
                    error.append(ent_0)
        scores_liste.append(scores)
    return scores_total, scores_liste
```

```python
def scoring(spans_0: List[Span], spans_1: List[Span], num_doc: int) -> List[Tuple]:
    scores = {"true": 0, "false": 0, "total": 0}
    error = []
    for key_0, value_0 in spans_0.items():
        for i in value_0:
            scores["total"] += 1
            ent_0 = extract(i, num_doc)
            matched = False
            for key_1, value_1 in spans_1.items():
                for j in value_1:
                    ent_1 = extract(j, num_doc)
                    if ent_0 == ent_1:
                        matched = True
                        scores["true"] += 1
                        continue
            if matched == False:
                scores["false"] += 1
                error.append(ent_0)

    return scores, error
```

```python
def perfect_match(corpus_0: List[Doc], corpus_1: List[Doc]) -> Tuple:
    scores_total = {"true_0": 0, "false_0": 0, "true_1": 0, "false_1": 0, "total": 0}
    scores_liste, error_0, error_1 = [], [], []

    for num_doc, (doc_0, doc_1) in enumerate(zip(corpus_0, corpus_1)):
        scores = {"true_0": 0, "false_0": 0, "true_1": 0, "false_1": 0, "total": 0}

        scores_0, error = scoring(doc_0.spans, doc_1.spans, num_doc)
        error_0.extend(error)

        scores_1, error = scoring(doc_1.spans, doc_0.spans, num_doc)
        error_1.extend(error)

        (
            scores["true_0"],
            scores["false_0"],
            # scores["true_1"],
            scores["false_1"],
            scores["total"],
        ) = (
            scores_0["true"],
            scores_0["false"],
            # scores_1["true"],
            scores_1["false"],
            scores_0["total"],
        )
        scores_liste.append(scores)
    return scores_liste, error_0, error_1
```

```python
corpus = ouvrir_corpus([dossier_0, dossier_1])
corpus_0 = corpus[0]
corpus_1 = corpus[1]

scores_0, scores_1 = evaluate(corpus_0, corpus_1)

scores_0 = eval(corpus_0, corpus_1, matching_mode="exact")

scores_liste, error_0, error_1 = perfect_match(corpus_0, corpus_1)
```

```python
df_scores = pd.DataFrame(scores_liste)
sum_row = df_scores.sum()
df_scores = df_scores.append(sum_row, ignore_index=True)
df_scores
```

```python
df_error_0 = pd.DataFrame(
    error_0,
    columns=[
        "Num_doc",
        "start",
        "end",
        "text",
        "label",
        "Tech",
        "Negation",
        "Certainty",
        "Temporality",
        "Family",
        "AttDate",
        "AttTemp",
        "Action",
        "RefTemp",
    ],
)
df_error_0
```

```python
df_error_1 = pd.DataFrame(
    error_1,
    columns=[
        "Num_doc",
        "start",
        "end",
        "text",
        "label",
        "Tech",
        "Negation",
        "Certainty",
        "Temporality",
        "Family",
        "AttDate",
        "AttTemp",
        "Action",
        "RefTemp",
    ],
)
df_error_1
```
