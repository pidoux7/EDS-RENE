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
nlp = edsnlp.blank("eds")
nlp.add_pipe(
    "eds.relations",
    config={
        "use_sentences": False,
        "clean_rel": False,
        "max_dist": 45,
    },
)
```

```python
corpus = nlp.pipe(corpus)

type(corpus)
```

```python
for doc in corpus:
    for k, v in doc.spans.items():
        for span in v:
            print(span._.rel)
```
