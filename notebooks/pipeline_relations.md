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
import edsnlp.pipes as eds

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
text = """Prise pendant 3 semaines d'Amlodipine 5mg per os une fois par jour mais HTA mal contrôlée.
        Metformine 500 mg deux fois par jour à partir du 27/05/2022.
        Consultation chez un cardiologue le 11/07 pour évaluation de l'HTA, dans l'attente majoration de l'AMLODIPINE à 10 mg.
        """

scheme = {
    "source": [{"label": "drug", "attr": None}],
    "target": [{"label": "dates", "attr": None}, {"label": "durations", "attr": None}],
    "type": "Temporal",
    "inv_type": "inv_Temporal",
}

nlp = edsnlp.blank("eds")

# Extraction of entities
nlp.add_pipe("eds.drugs")
nlp.add_pipe("eds.dates")
nlp.add_pipe("eds.sentences")
nlp.add_pipe(
    "eds.relations",
    config={
        "scheme": scheme,
        "use_sentences": True,
        "clean_rel": True,
        "proximity_method": "sym",
        "max_dist": 45,
    },
)
doc = nlp(text)
```

```python
for label in doc.spans:
    print("Label: ", label, "\t Entities :", doc.spans[label])
    for span in doc.spans[label]:
        print("\t Entity :", span, "\t Relations :", span._.rel)
    print("\n")
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
nlp = edsnlp.blank("eds")
nlp.add_pipe("eds.sentences")
nlp.add_pipe(
    "eds.relations",
    config={
        "scheme": "./relations.json",
        "use_sentences": True,
        "clean_rel": True,
        "proximity_method": "sym",
        "max_dist": 45,
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
