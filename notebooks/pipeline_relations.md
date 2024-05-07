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
import sys

sys.path.insert(0, "/home/pidoux/edsnlp")
import edsnlp

# import edsnlp.pipes as pipes
# import edsnlp.pipes.misc.relations as rel
import edsnlp.pipes.core as eds

sys.path.append("/home/pidoux/LIMICS/EDS-RENE/")
from RENE import *
from RENE.utils import *

%reload_ext jupyter_black
%load_ext autoreload
%autoreload 2
import spacy
from spacy.tokens import Doc, Span
from spacy import Language
from typing import List, Tuple, Dict, Any, Union, Generator
import os
import copy
import math as m
import matplotlib.pyplot as plt
from sklearn.metrics import auc
```

```python
dossier_1 = "/home/pidoux/LIMICS/brat/data/RENE/"
```

```python
doc_iterator = edsnlp.data.read_standoff(
    dossier_1,
)
corpus_true = list(doc_iterator)
corpus_true.sort(key=lambda x: x.text)
```

```python
doc_iterator = edsnlp.data.read_standoff(
    dossier_1,
)
corpus = list(doc_iterator)
corpus.sort(key=lambda x: x.text)
```

```python
relations = [
    {
        "subject": [{"label": "Chemical_and_drugs", "attr": {"Tech": [None]}}],
        "object": [
            {
                "label": "Temporal",
                "attr": {"AttTemp": ["Duration", "Date"]},
            },
            {
                "label": "Chemical_and_drugs",
                "attr": {"Tech": ["dosage", "route", "strength", "form"]},
            },
        ],
        "type": "Depend",
        "inv_type": "inv_Depend",
    },
    {
        "subject": [{"label": "DISO", "attr": {"Tech": [None]}}],
        "object": [
            {
                "label": "Temporal",
                "attr": {"AttTemp": ["Duration", "Date"]},
            },
        ],
        "type": "Depend",
        "inv_type": "inv_Depend",
    },
]
```

```python
nlp = edsnlp.blank("eds")
nlp.add_pipe("eds.sentences")
nlp.add_pipe(
    "eds.relations",
    config={
        "scheme": "./relations.json",
        "use_sentences": True,
        "clean_rel": True,
        "proximity_method": "sym",
        "max_dist": 1,
    },
)
corpus = nlp.pipe(corpus)
```

```python
corpus_pred = corpus
item = [
    "total",
    "dosage",
    "form",
    "route",
    "strength",
    "Temporal",
    "Duration",
    "Frequency",
    "Time",
    "Date",
]
for i in item:
    precision_recall_curve(
        corpus_true=corpus_true,
        corpus_pred=corpus,
        max_dist=100,
        pas=1,
        label=i,
        sents=True,
        method="right",
    )
```

```python

```
