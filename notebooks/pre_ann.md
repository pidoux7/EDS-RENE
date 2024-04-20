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
```

### Dossier BRAT

```python
original_rep = "/home/pidoux/LIMICS/brat/data/RENE/"
predicted_rep = "/home/pidoux/LIMICS/brat/data/test/"
merged_rep = "/home/pidoux/LIMICS/brat/data/merge/"
```

### Import des docs et prédictions des entités dates avec EDS-NLP pipeline PRED

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
    nlp.add_pipe(eds.dates())
    pred_iterator = docs.map_pipeline(nlp)
    return list(pred_iterator)
```

```python
def prepare_brat(pred_docs: List, files_name: List[str], predicted_rep: str):
    """create the brat files from the predicted documents

    Args:
        pred_docs (List): list of the predicted documents
        files_name (List[str]): list of the name of the files
        predicted_rep (str): repertory where the brat files will be created
    """
    brat = BratConnector(predicted_rep)
    for i, doc in enumerate(pred_docs):
        doc._.note_id = files_name[i]
        doc.spans["pollutions"] = []
    brat.docs2brat(pred_docs)
```

```python
def fusion_ann(
    original_rep: str, predicted_rep: str, merged_rep: str, files_name: List[str]
):
    """Fusionne les annotations de deux fichiers .ann en un seul fichier .ann

    Args:
        original_rep (str): source directory of the original annotations
        predicted_rep (str): source directory of the predicted annotations
        merged_rep (str): destination directory of the merged annotations
        files_name (List[str]): list of file names to merge
    """
    # Iterate over the files
    for file_name in tqdm(files_name):
        with open(original_rep + file_name + ".ann", "r", encoding="utf-8") as file:
            lines_file_original = file.readlines()

        # Extract max existing id
        existing_ids = max(
            [int(line.split("\t")[0][1:]) for line in lines_file_original]
        )

        # Lire le second fichier et préparer les nouvelles lignes
        new_lines = []
        with open(predicted_rep + file_name + ".ann", "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue  # Skip lines with wrong format
                # Replace 'date' par 'Temporal'
                if "date" in parts[1]:
                    parts[1] = parts[1].replace("date", "Temporal")
                # Generate new identifier
                parts[0] = "T" + str(existing_ids + 1)
                existing_ids += 1
                new_line = "\t".join(parts) + "\n"
                new_lines.append(new_line)

        # Merge lines from the original file with the new lines
        merged_lines = lines_file_original + new_lines

        # Write the merged file
        with open(merged_rep + file_name + ".ann", "w", encoding="utf-8") as file:
            file.writelines(merged_lines)
```

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

```python

```
