from typing import List
import os
from tqdm import tqdm
import edsnlp
import edsnlp.pipes as eds
from edsnlp.connectors.brat import BratConnector


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


def prepare_brat_train(
    pred_docs: List, files_name: List[str], predicted_rep: str, original_rep: str
):
    """create the brat files from the predicted documents
        WARNING : this function is used for training,
        it will only keep the Temporal entities but it doesn't work for the moment
    Args:
        pred_docs (List): list of the predicted documents
        files_name (List[str]): list of the name of the files
        predicted_rep (str): repertory where the brat files will be created
    """
    brat = BratConnector(predicted_rep)

    doc_iterator = edsnlp.data.read_standoff(original_rep)
    nlp = edsnlp.blank("eds")
    pred_iterator = doc_iterator.map_pipeline(nlp)
    pred_date = list(pred_iterator)
    pred_docs = list(pred_docs)
    for i, (pred_doc, cop_doc) in enumerate(zip(pred_docs, pred_date)):
        cop_doc._.note_id = files_name[i]
        cop_doc.ents = [ent for ent in pred_doc.ents if ent.label_ == "Temporal"]
        cop_doc.spans = {"Temporal": cop_doc.ents}
        print(cop_doc.ents)
        pred_date[i] = cop_doc
    brat.docs2brat(pred_date)


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


def fusion_ann_train(
    original_rep: str, predicted_rep: str, merged_rep: str, files_name: List[str]
):
    """Fusionne les annotations de deux fichiers .ann en un seul fichier .ann
    WARNING : this function is used for training, but it doesn't work for the moment

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
