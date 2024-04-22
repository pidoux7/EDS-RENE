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

# Evaluation inter-annotateurs

```python
import sys

sys.path.insert(0, "/home/pidoux/LIMICS/edsnlp")
import edsnlp

sys.path.append("/home/pidoux/LIMICS/EDS-RENE/")
from RENE.eval_annotator import *

from spacy.tokens import Doc, Span
from spacy import Language
from typing import List, Tuple, Dict, Any, Union, Generator
import os
import pandas as pd

%load_ext autoreload
%load_ext nb_black
%autoreload 2
```

# Ouverture corpus

```python
dossier_0 = "/home/pidoux/LIMICS/brat/data/merge/"
dossier_1 = "/home/pidoux/LIMICS/brat/data/RENE/"
corpus = ouvrir_corpus([dossier_0, dossier_1])
corpus_0 = corpus[0]
corpus_1 = corpus[1]
```

# Evaluation Entitiés

```python
NER_scores = evaluate_ner(corpus_0, corpus_1)
data_exact, data_partial = process_scores(NER_scores)
df_exact = create_dataframe(data_exact)
df_partial = create_dataframe(data_partial)
```

```python
df_exact
```

```python
df_partial
```

# Evaluation Attributs

```python
scores_liste, error_0, error_1,accord = perfect_match_ner(corpus_0, corpus_1)
attribut = ["label", "Tech", "Negation", "Certainty", "Temporality", "Family", "AttDate", "AttTemp", "Action", "RefTemp"]
df_concat = pd.DataFrame()
for i in attribut:
    df_summary = creer_tableau_synthese(accord, error_0, error_1,regroupement=i )
    df_concat = pd.concat([df_concat, df_summary], axis=0)
df_concat
```

```python
df_scores = pd.DataFrame(scores_liste)
sum_row = df_scores.sum()
df_scores = df_scores.append(sum_row, ignore_index=True)
df_scores
```

```python
view_df_ner(error_0)
```

```python
view_df_ner(error_1)
```

# Evaluation Relations

```python
scores_liste_rel, error_0_rel, error_1_rel = perfect_match_rel(corpus_0, corpus_1)
```

```python
df_scores_rel = pd.DataFrame(scores_liste_rel)
sum_row = df_scores_rel.sum()
df_scores_rel = df_scores_rel.append(sum_row, ignore_index=True)
df_scores_rel
```

```python
view_df_rel(error_0_rel)
```

```python
view_df_rel(error_1_rel)
```

```python

```

```python

```

```python

```
