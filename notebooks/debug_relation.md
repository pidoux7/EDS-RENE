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

%reload_ext jupyter_black
%load_ext autoreload
%autoreload 2
from spacy.tokens import Doc, Span
from spacy import Language
from typing import List, Tuple, Dict, Any, Union, Generator
import os
```

```python
dossier_0 = "/home/pidoux/LIMICS/brat/data/merge/"
dossier_1 = "/home/pidoux/LIMICS/brat/data/RENE/"
```

```python
doc_iterator = edsnlp.data.read_standoff(
    dossier_1,
)
```

```python
corpus = list(doc_iterator)
```

```python
rel = 0
for k, v in corpus[1].spans.items():
    for span in v:
        if len(span._.rel) >= 1:
            print(span.text, span._.rel)
            rel += len(span._.rel)
print(rel)
```

entity_id

[{'nature': 'depend', 'span': grippe}, ..]
