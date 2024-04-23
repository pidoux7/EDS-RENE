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
# if not Span.has_extension("rel"):
# Span.set_extension("rel", default=[])
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
# nlp = edsnlp.blank("eds")
# nlp.add_pipe("sentencizer")
# doc_iterator = doc_iterator.map_pipeline(nlp)
corpus = list(doc_iterator)
corpus.sort(key=lambda x: x.text)
```

```python
for doc in corpus:
    for sent in doc.sents:
        print(sent.text)

        print(sent.start, sent.end)
        print(sent.start_char, sent.end_char)
        print(sent.ents)
        print("\n")
```

```python
for doc in corpus:
    for k, ents in doc.spans.items():
        for ent in ents:
            print(ent._.rel)
```

```python
model = model_proximity()
for i in range(0, 101, 1):
    corpus_pred = model.predict(
        corpus, max_dist=i, clean=True, method="right", sents=True
    )
    print(i, model.score(corpus_true, corpus_pred))
```

```python
corpus_pred = model.predict(corpus, max_dist=45, clean=True)
model.precision_recall_curve(
    corpus_true, corpus_pred, max_dist=100, pas=1, method="right", sents=True
)
```

```python
model = model_proximity()
corpus_pred = model.predict(corpus, max_dist=45, clean=True, method="right")
s = model.score(corpus_true, corpus_pred)
print(s)
```

```python
pred_rels = []
for i, doc in enumerate(corpus_pred):
    pred_rel = []
    for i, span_sub in enumerate(doc.spans["Chemical_and_drugs"]):
        if span_sub._.Tech is None:
            med = [x for x in span_sub._.rel if x["nature"] == "Depend"]
            info = [span_sub.text]
            for x in med:
                if x["span"]._.Tech is not None:
                    info.append(x["span"]._.Tech)
                    info.append(x["span"].text)
                if x["span"].label_ == "Temporal":
                    info.append(x["span"].label_)
                    info.append(x["span"].text)
            print(info)
            pred_rel.append(info)
    pred_rels.append(pred_rel)
```

```python

```
