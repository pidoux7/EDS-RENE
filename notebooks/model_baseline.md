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


sys.path.insert(0, "/home/pidoux/LIMICS/edsnlp")
import edsnlp
import edsnlp.pipes.core as eds

sys.path.append("/home/pidoux/LIMICS/EDS-RENE/")
from RENE import *
from RENE.rel_model import model_proximity

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
for doc in corpus:
    for k, ents in doc.spans.items():
        dic = {}
        for ent in ents:
            dic[ent] = 3
for k, v in dic.items():
    print(k, v)
    print(type(k))
```

dictionnaire avec clé span et valeur soit
- string
- liste string
- dictionnaire
- liste de dictionnaire

```python
doc.user_data
```

```python
model = model_proximity()
```

```python
corpus_pred = model.predict(
        corpus, max_dist=i, clean=True, method="right", sents=True
    )
print(i, model.score(corpus_true, corpus_pred))
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
    model.precision_recall_curve(
        corpus_true,
        corpus_pred,
        max_dist=100,
        pas=1,
        label=i,
        method="right",
        sents=True,
    )
```

```python
pred_rels = []
for i, doc in enumerate(corpus_pred):
    pred_rel = []
    for i, span_sub in enumerate(doc.spans["Chemical_and_drugs"]):
        if span_sub._.Tech is None:
            med = [x for x in span_sub._.rel if x["type"] == "Depend"]
            info = [span_sub.text]
            for x in med:
                if x["target"]._.Tech is not None:
                    info.append(x["target"]._.Tech)
                    info.append(x["target"].text)
                if x["target"].label_ == "Temporal":
                    info.append(x["target"].label_)
                    info.append(x["target"].text)
            print(info)
            pred_rel.append(info)
    pred_rels.append(pred_rel)
```

```python

```
