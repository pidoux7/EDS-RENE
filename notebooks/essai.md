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
```

```python
original_rep = "/home/pidoux/LIMICS/brat/data/RENE/"
predicted_rep = "/home/pidoux/LIMICS/brat/data/test/"
merged_rep = "/home/pidoux/LIMICS/brat/data/merge/"
```

```python
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
```

```python
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
    nlp.add_pipe(eds.negation())
    nlp.add_pipe(eds.hypothesis())
    nlp.add_pipe(eds.family())
    nlp.add_pipe(eds.dates())
    pred_iterator = docs.map_pipeline(nlp)
    return pred_iterator
```

```python
doc_iterator = edsnlp.data.read_standoff(
    original_rep,
    span_setter={"ents": "Temporal"},
)

true_docs = list(doc_iterator)

corpus, files_name = txt_liste(original_rep)

pred_docs = pipeline_date(corpus)
```

```python
import pandas as pd
from datetime import datetime
entities = []
for i,doc in enumerate(pred_docs):
    for ent in doc.spans["dates"]:
        note_datetime = datetime(year=2019, month=8, day=27)
        d = dict(
            file_number=i,
            lexical_variant=ent.text,
            sentence= ent.sent,
            debut=ent.start,
            fin=ent.end,
            normalized_variant=ent._.date.to_datetime(
            note_datetime=note_datetime,
            infer_from_context=True,
            tz=None,
            default_day=15,
        ),
            label=ent.label_,
            negation=ent._.negation,
            hypothesis=ent._.hypothesis,
            family=ent._.family,
        )
        entities.append(d)

    df = pd.DataFrame.from_records(entities)
```

```python
df
```

```python
def convert_doc_to_rows(doc):
    entities = []

    for ent in doc.ents:
        d = dict(
            start=ent.start_char,
            end=ent.end_char,
            label=ent.label_,
            lexical_variant=ent.text,
            negation=ent._.negation,
            hypothesis=ent._.hypothesis,
            family=ent._.family,
        )
        entities.append(d)

    for date in doc.spans.get("dates", []):
        d = dict(
            begin=date.start_char,
            end=date.end_char,
            label="date",
            entity_text=date.text,
        )
        entities.append(d)

    return entities


df = pred_docs.to_pandas(converter=convert_doc_to_rows)
# or equivalently:
df = pred_docs.to_pandas(
    converter="ents",
    span_getter=["ents", "dates"],
    span_attributes=["negation", "hypothesis", "family"],)
```

```python
from edsnlp.viz import QuickExample
nlp = edsnlp.blank("eds")
nlp.add_pipe(eds.sentences())
nlp.add_pipe(eds.normalizer())
nlp.add_pipe(eds.negation())
nlp.add_pipe(eds.hypothesis())
nlp.add_pipe(eds.family())
nlp.add_pipe(eds.dates())
E = QuickExample(nlp)

txt = E(corpus[0],as_dataframe=True)
```

```python
txt
```

```python
import os

# Obtenir le chemin absolu du fichier script.py
cwd = os.getcwd
data_path = os.path.join(cwd, 'data', 'data.csv')
print(data_path)
```
