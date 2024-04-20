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

# Pré Annotation dates et ajout sur fichier BRAT


### Import

```python
%load_ext autoreload
%load_ext nb_black
%autoreload 2
```

```python
from typing import List
import os
from tqdm import tqdm
import edsnlp
import edsnlp.pipes as eds
from edsnlp.connectors.brat import BratConnector
from spacy import displacy
import sys
sys.path.append("/home/pidoux/LIMICS/EDS-RENE/")
from RENE.pre_annotation import *
```

### Dossier BRAT

```python
original_rep = "/home/pidoux/LIMICS/brat/data/RENE/"
predicted_rep = "/home/pidoux/LIMICS/brat/data/test/"
merged_rep = "/home/pidoux/LIMICS/brat/data/merge/"
```

### Import des docs et prédictions des entités dates avec EDS-NLP pipeline PRED

```python
doc_iterator = edsnlp.data.read_standoff(
    original_rep,
    span_setter={"ents": "Temporal"},
)

true_docs = list(doc_iterator)

corpus, files_name = txt_liste(original_rep)

pred_docs = pipeline_date(corpus)

prepare_brat(pred_docs, files_name, predicted_rep)

fusion_ann(original_rep, predicted_rep, merged_rep, files_name)
```

```python
# displacy.render(pred_docs[2], style="span", options={"spans_key": "dates"} )
# print(pred_docs[0].spans)
# pred_docs[0].ents = pred_docs[0].spans["dates"]
# displacy.render(pred_docs, style="ent")
```

```python
entities= ["Chemical_and_drugs", "DISO", "Temporal", "Constantes", "BIO", "BIO_comp"]
total = 0
for ent in entities:
    doc_iterator = edsnlp.data.read_standoff(
        merged_rep,
        span_setter={"ents": ent},
    )

    true_docs = list(doc_iterator)

    nb_med_doc = []
    for doc in true_docs:
        nb_med_doc.append(len(doc.ents))
    print(ent)
    print(f'Number of {ent} entities per document: {nb_med_doc}')
    print(f'Total number of {ent} entities per document: {sum(nb_med_doc)}')
    print(f'Mean number of {ent} entities per document: {sum(nb_med_doc)/len(nb_med_doc)}\n')
    total += sum(nb_med_doc)
print(f'Total number of entities: {total}')
```

```python
total = 0
for ent in entities:
    doc_iterator = edsnlp.data.read_standoff(
        original_rep,
        span_setter={"ents": ent},
    )

    true_docs = list(doc_iterator)

    nb_med_doc = []
    for doc in true_docs:
        nb_med_doc.append(len(doc.ents))
    print(ent)
    print(f'Number of {ent} entities per document: {nb_med_doc}')
    print(f'Total number of {ent} entities per document: {sum(nb_med_doc)}')
    print(f'Mean number of {ent} entities per document: {sum(nb_med_doc)/len(nb_med_doc)}\n')
    total += sum(nb_med_doc)
print(f'Total number of entities: {total}')
```
