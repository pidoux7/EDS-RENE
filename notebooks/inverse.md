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
import os
import glob

def read_ann_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def write_ann_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)

def get_entity_type(line):
    parts = line.split()
    return parts[1] if len(parts) > 1 else None

def get_relation_args(line):
    parts = line.split()
    return parts[2][6:], parts[3][6:]

def find_entity_by_id(lines, entity_id):
    for line in lines:
        if line.startswith('T') and line.split()[0] == 'T'+entity_id:
            if line.split()[1] == 'Chemical_and_drugs':
                return True
    return False

def process_ann_file(file_path):
    lines = read_ann_file(file_path)
    entities = {line.split()[0]: get_entity_type(line) for line in lines if line.startswith('T')}
    new_lines = []
    for line in lines:
        if line.startswith('R'):
            parts = line.split()
            rel_type = parts[1]
            arg1, arg2 = get_relation_args(line)
            if find_entity_by_id(lines, arg1) and find_entity_by_id(lines, arg2):
                p = parts[2]
                parts[2] = parts[3]
                parts[3] = p
                new_line = '\t'.join(parts) + '\n'
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    write_ann_file(file_path, new_lines)

def process_all_ann_files(directory):
    # Utilise glob pour trouver tous les fichiers .ann dans le répertoire
    ann_files = glob.glob(os.path.join(directory, "*.ann"))
    for ann_file in ann_files:
        process_ann_file(ann_file)


directory = "./RENE/"  # Remplacez par le chemin de votre dossier contenant les fichiers .ann
process_all_ann_files(directory)
```

```python

```
