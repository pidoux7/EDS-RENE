import sys
from spacy.tokens import Doc, Span
from typing import List, Tuple, Dict, Any, Generator
import pandas as pd
import edsnlp

sys.path.insert(0, "/home/pidoux/LIMICS/edsnlp")


def ouvrir_corpus(liste_dossier: List) -> List[List[Doc]]:
    """Open a corpus of documents from a list of folders

    Args:
        liste_dossier (list): List of folders containing the documents .ann and .txt

    Returns:
        list[List[Doc]]: List of documents
    """
    if len(liste_dossier) != 2:
        raise ValueError("The list should contain two folders")
    corpus = []
    for dossier in liste_dossier:
        doc_iterator = edsnlp.data.read_standoff(
            dossier,
        )
        docs = list(doc_iterator)
        docs.sort(key=lambda x: x.text)
        corpus.append(docs)

    return corpus


def is_match(ent_0: Span, ent_1: Span, matching_mode: str = "exact") -> bool:
    """Returns True if the predicted entity matches the ground truth entity.
    If matching_mode = "exact": The function returns True when
    the entity boundaries are exactly the same.
    If matching_mode = "partial" The function returns True when
    the entity boundaries are overlaping.

    Args:
        true_ent (Span): Ground truth entity
        pred_ent (Span): Predicted entity
        matching_mode (str): Matching mode. Either "exact" or "partial"

    Returns:
        bool: True if the predicted entity matches entity, False otherwise
    """
    start_char_0, end_char_0 = (ent_0.start_char, ent_0.end_char)
    start_char_1, end_char_1 = (ent_1.start_char, ent_1.end_char)
    if ent_0.label_ == ent_1.label_:
        if matching_mode == "exact":
            if (start_char_0 == start_char_1) or (end_char_0 == end_char_1):
                return True
            else:
                return False
        elif matching_mode == "partial":
            if (end_char_0 < start_char_1) or (start_char_0 > end_char_1):
                return False
            else:
                return True
        else:
            raise ValueError(f"exact or partial and not {matching_mode}")
    else:
        return False


def evaluate_ner(corpus_0: List[Doc], corpus_1: List[Doc]) -> Dict:
    """Evaluation of NER annotations between two annotated corpora

    Args:
        corpus_0 (List[Doc]): corpus annotated by the first annotator
        corpus_1 (List[Doc]): corpus annotated by the second annotator

    Returns:
        Tuple[dict,dict]: Evaluation scores
    """

    scores = {
        "exact_accord": 0,
        "exact_errors_0": 0,
        "exact_errors_1": 0,
        "partial_accord": 0,
        "partial_errors_0": 0,
        "partial_errors_1": 0,
        "total_ents": 0,
        "exact_accord_liste": [],
        "partial_accord_liste": [],
        "exact_error_0_liste": [],
        "exact_error_1_liste": [],
        "partial_error_0_liste": [],
        "partial_error_1_liste": [],
    }

    for i, (doc_0, doc_1) in enumerate(zip(corpus_0, corpus_1)):
        for k, v in doc_0.spans.items():
            for ent_0 in v:
                e = (i, ent_0.label_, ent_0.text, ent_0.start_char, ent_0.end_char)
                matched_partial = False
                matched_exact = False
                scores["total_ents"] += 1
                for k1, v1 in doc_1.spans.items():
                    for ent_1 in v1:
                        if is_match(ent_0, ent_1, matching_mode="exact"):
                            scores["exact_accord"] += 1
                            scores["exact_accord_liste"].append(e)
                            matched_exact = True
                        if is_match(ent_0, ent_1, matching_mode="partial"):
                            scores["partial_accord"] += 1
                            scores["partial_accord_liste"].append(e)
                            matched_partial = True
                            continue
                if not matched_exact:
                    scores["exact_errors_0"] += 1
                    scores["exact_error_0_liste"].append(e)
                if not matched_partial:
                    scores["partial_errors_0"] += 1
                    scores["partial_error_0_liste"].append(e)

        for k, v in doc_1.spans.items():
            for ent_1 in v:
                e = (i, ent_1.label_, ent_1.text, ent_1.start_char, ent_1.end_char)
                scores["total_ents"] += 1
                matched_exact = False
                matched_partial = False
                for k1, v1 in doc_0.spans.items():
                    for ent_0 in v1:
                        if is_match(ent_0, ent_1, matching_mode="exact"):
                            matched_exact = True
                        if is_match(ent_0, ent_1, matching_mode="partial"):
                            matched_partial = True
                            continue
                if not matched_exact:
                    scores["exact_errors_1"] += 1
                    scores["exact_error_1_liste"].append(e)
                if not matched_partial:
                    scores["partial_error_1"] += 1
                    scores["partial_error_1_liste"].append(e)
    scores["total_ents"] /= 2

    return scores


def process_scores(scores: Dict) -> Tuple[Dict, Dict]:
    """Process the scores to extract the number of entities
    that are in exact match, partial match or not matching

    Args:
        scores (Dict): Scores from the evaluation

    Returns:
        Tuple[Dict, Dict]: Dictionaries containing the number of entities
        that are in exact match, partial match or not matching
    """
    data_exact = {}
    data_partial = {}

    for key in [
        "exact_accord_liste",
        "exact_error_0_liste",
        "exact_error_1_liste",
        "partial_accord_liste",
        "partial_error_0_liste",
        "partial_error_1_liste",
    ]:
        for entry in scores[key]:
            idx, label, text, start, end = entry
            target_dict = data_exact if "exact" in key else data_partial

            if label not in target_dict:
                target_dict[label] = {"accord": 0, "error_0": 0, "error_1": 0}

            if "accord" in key:
                target_dict[label]["accord"] += 1
            elif "error" in key:
                if "0" in key:
                    target_dict[label]["error_0"] += 1
                elif "1" in key:
                    target_dict[label]["error_1"] += 1

    return data_exact, data_partial


def create_dataframe(data_dict: Dict) -> pd.DataFrame:
    """Create a DataFrame from the data dictionary

    Args:
        data_dict (Dict): Data dictionary

    Returns:
        pd.DataFrame: DataFrame containing the data
    """
    labels = ["Total"]
    accords = [0]
    error_0s = [0]
    error_1s = [0]
    percentages = [0]
    totals = [0]

    for label, counts in data_dict.items():
        accord = counts["accord"]
        error_0 = counts["error_0"]
        error_1 = counts["error_1"]
        total = accord + error_0 + error_1
        error_percentage = ((error_0 + error_1) / total * 100) if total > 0 else 0

        labels.append(label)
        accords.append(accord)
        error_0s.append(error_0)
        error_1s.append(error_1)
        percentages.append(f"{error_percentage:.2f}%")
        totals.append(total)

        accords[0] += accord
        error_0s[0] += error_0
        error_1s[0] += error_1
        totals[0] += total

    if totals[0] > 0:
        total_errors = error_0s[0] + error_1s[0]
        overall_error_percentage = total_errors / totals[0] * 100
        percentages[0] = f"{overall_error_percentage:.2f}%"

    df = pd.DataFrame(
        {
            "Label": labels,
            "Accord": accords,
            "Error 0": error_0s,
            "Error 1": error_1s,
            "% Error Total": percentages,
            "Total": totals,
        }
    )

    total_row = df.iloc[0]
    df = df.iloc[1:].append(total_row)

    return df


def PRF_ner(
    true_docs: List[Doc], pred_docs: List[Doc], matching_mode: str = "exact"
) -> Dict:
    """Compare documents by documents two list of Spacy Doc
    and returns a dictionary with:
    - The number of True Positive (TP):
        predicted entities matching true entities.
    - The number of False Positive (FP):
        predicted entities not matching any true entities.
    - The number of False Negative (FN):
        true entities not matching any predicted entities.
    - The Precision (precision):
        the fraction of relevant instances among the retrieved instances.
    - The Recall (recall):
        the fraction of relevant instances that were retrieved.
    - The F1-score (f1):
        the harmonic mean of precision and recall.
    """
    scores = {"TP": 0, "FP": 0, "FN": 0, "FN_liste": [], "FP_liste": []}
    for i, (true_doc, pred_doc) in enumerate(zip(true_docs, pred_docs)):
        for true_ent in true_doc.ents:
            matched = False
            for pred_ent in pred_doc.ents:
                if is_match(true_ent, pred_ent, matching_mode):
                    scores["TP"] += 1
                    matched = True
                    continue
            if not matched:
                scores["FN"] += 1
                scores["FN_liste"].append(
                    (
                        i,
                        true_ent.label_,
                        true_ent.text,
                        true_ent.start_char,
                        true_ent.end_char,
                    )
                )

        for pred_ent in pred_doc.ents:
            matched = False
            for true_ent in true_doc.ents:
                if is_match(true_ent, pred_ent, matching_mode):
                    matched = True
                    continue
            if not matched:
                scores["FP"] += 1
                scores["FP_liste"].append(
                    (
                        i,
                        pred_ent.label_,
                        pred_ent.text,
                        pred_ent.start_char,
                        pred_ent.end_char,
                    )
                )

    if scores["FP"] == 0:
        scores["precision"] = 1.0
    else:
        scores["precision"] = scores["TP"] / (scores["TP"] + scores["FP"])

    if scores["FN"] == 0:
        scores["recall"] = 1.0
    else:
        scores["recall"] = scores["TP"] / (scores["TP"] + scores["FN"])

    if (scores["precision"] + scores["recall"]) == 0:
        scores["f1"] = 0
    else:
        scores["f1"] = (
            2
            * (scores["precision"] * scores["recall"])
            / (scores["precision"] + scores["recall"])
        )

    return scores


def extract(span: Span, num_doc: int) -> Tuple:
    """extract information from a span

    Args:
        span (Span): span to extract information from
        num_doc (int): document number

    Returns:
        Tuple: Tuple containing the extracted information
    """
    return (
        num_doc,
        span.start_char,
        span.end_char,
        span.text,
        span.label_,
        span._.Tech,
        span._.Negation,
        span._.Certainty,
        span._.Temporality,
        span._.Family,
        span._.AttDate,
        span._.AttTemp,
        span._.Action,
        span._.RefTemp,
    )


def scoring(
    spans_0: List[Span], spans_1: List[Span], num_doc: int
) -> Tuple[List[Tuple], List[str]]:
    """scoring function to evaluate the match of NER and their attributes

    Args:
        spans_0 (List[Span]): spans from the first annotator
        spans_1 (List[Span]): spans from the second annotator
        num_doc (int): number of the document

    Returns:
        Tuple[List[Tuple], List[str]]: list of tuples with the attributes of the spans
    """
    scores = {"true": 0, "false": 0, "total": 0}
    error = []
    concord = []
    for key_0, value_0 in spans_0.items():
        for i in value_0:
            scores["total"] += 1
            ent_0 = extract(i, num_doc)
            matched = False
            for key_1, value_1 in spans_1.items():
                for j in value_1:
                    ent_1 = extract(j, num_doc)
                    if ent_0 == ent_1:
                        matched = True
                        scores["true"] += 1
                        concord.append(ent_0)
                        continue
            if not matched:
                scores["false"] += 1
                error.append(ent_0)

    return scores, error, concord


def perfect_match_ner(corpus_0: List[Doc], corpus_1: List[Doc]) -> Tuple:
    """Evaluation of NER and their attributes annotations between two annotated corpora


    Args:
        corpus_0 (List[Doc]): annotated corpus by the first annotator
        corpus_1 (List[Doc]): annotated corpus by the second annotator

    Returns:
        Tuple: Evaluation scores and list of errors
    """
    scores_liste, error_0, error_1, accord = [], [], [], []

    for num_doc, (doc_0, doc_1) in enumerate(zip(corpus_0, corpus_1)):
        scores = {"accord": 0, "error_0": 0, "error_1": 0, "total": 0}

        scores_0, error, concord = scoring(doc_0.spans, doc_1.spans, num_doc)
        error_0.extend(error)
        accord.extend(concord)

        scores_1, error, concord = scoring(doc_1.spans, doc_0.spans, num_doc)
        error_1.extend(error)

        (
            scores["accord"],
            scores["error_0"],
            scores["error_1"],
            scores["total"],
        ) = (
            scores_0["true"],
            scores_0["false"],
            scores_1["false"],
            scores_0["true"] + scores_0["false"] + scores_1["false"],
        )
        scores_liste.append(scores)

    return scores_liste, error_0, error_1, accord


def creer_tableau_synthese(
    accord: List[Dict[str, Any]],
    erreur_0: List[Dict[str, Any]],
    erreur_1: List[Dict[str, Any]],
    regroupement: str,
) -> pd.DataFrame:
    """Crée un tableau synthétique des erreurs de NER

    Args:
        accord (List[Dict[str, Any]]): liste des annotations en accord
        erreur_0 (List[Dict[str, Any]]): liste des annotations en erreur_0
        erreur_1 (List[Dict[str, Any]]): liste des annotations en erreur_1
        regroupement (str): colonne sur laquelle regrouper les annotations

    Returns:
        pd.DataFrame: tableau synthétique des erreurs de NER
    """
    liste = [
        "num_doc",
        "start_char",
        "end_char",
        "text",
        "label",
        "Tech",
        "Negation",
        "Certainty",
        "Temporality",
        "Family",
        "AttDate",
        "AttTemp",
        "Action",
        "RefTemp",
    ]

    df_accord = pd.DataFrame(accord, columns=liste)
    df_erreur_0 = pd.DataFrame(erreur_0, columns=liste)
    df_erreur_1 = pd.DataFrame(erreur_1, columns=liste)

    counts_accord = df_accord[regroupement].value_counts()
    counts_erreur_0 = df_erreur_0[regroupement].value_counts()
    counts_erreur_1 = df_erreur_1[regroupement].value_counts()

    df_summary = pd.DataFrame(
        {
            "Accord": counts_accord,
            "Erreur_0": counts_erreur_0,
            "Erreur_1": counts_erreur_1,
        }
    ).fillna(0)

    df_summary = df_summary.astype(int)

    df_summary["Total"] = df_summary.sum(axis=1)
    df_summary["Pourcentage d'Erreur"] = (
        (df_summary["Erreur_0"] + df_summary["Erreur_1"]) / df_summary["Total"] * 100
    ).round(2)

    total_row = df_summary.sum()
    total_row["Pourcentage d'Erreur"] = (
        (total_row["Erreur_0"] + total_row["Erreur_1"]) / total_row["Total"] * 100
    ).round(2)
    df_summary.loc["Total " + regroupement] = total_row

    return df_summary


def view_df_ner(error_liste: List[str]) -> pd.DataFrame:
    """visualize the errors in a DataFrame

    Args:
        error_liste (List[str]): List of errors

    Returns:
        pd.DataFrame: DataFrame of errors
    """
    return pd.DataFrame(
        error_liste,
        columns=[
            "Num_doc",
            "start",
            "end",
            "text",
            "label",
            "Tech",
            "Negation",
            "Certainty",
            "Temporality",
            "Family",
            "AttDate",
            "AttTemp",
            "Action",
            "RefTemp",
        ],
    )


def extract_rel_sujet(span_sujet: Span, num_doc: int) -> Tuple:
    """extract information from a span

    Args:
        span (Span): span to extract information from
        num_doc (int): document number

    Returns:
        Tuple: Tuple containing the extracted information
    """
    return (
        num_doc,
        span_sujet.start_char,
        span_sujet.end_char,
        span_sujet.text,
        span_sujet.label_,
    )


def extract_rel_objet(span_objet: Dict) -> Tuple:
    """extract information from a span

    Args:
        span (Span): span to extract information from
        num_doc (int): document number

    Returns:
        Tuple: Tuple containing the extracted information
    """
    return (
        span_objet["nature"],
        span_objet["span"].start_char,
        span_objet["span"].end_char,
        span_objet["span"].text,
        span_objet["span"].label_,
    )


def extract_rel(span: Span, num_doc: int) -> Generator[Tuple, None, None]:
    """extract information from a span

    Args:
        span (Span): span to extract information from
        num_doc (int): document number

    Yields:
        Generator[Tuple, None, None]:
            Generator of tuples containing the extracted information
    """
    sujet = extract_rel_sujet(span, num_doc)
    for objet in span._.rel:
        objet = extract_rel_objet(objet)
        yield sujet + objet


def scoring_rel(
    spans_0: Dict[str, List[Span]], spans_1: Dict[str, List[Span]], num_doc: int
) -> Tuple[Dict[str, int], List[Tuple]]:
    scores = {"true": 0, "false": 0, "total": 0}
    error = []
    for key_0, value_0 in spans_0.items():
        for span_0 in value_0:
            if span_0._.rel == []:
                continue
            else:
                for ent_0 in extract_rel(
                    span_0, num_doc
                ):  # Traite chaque valeur générée
                    matched = False
                    scores["total"] += 1
                    for key_1, value_1 in spans_1.items():
                        for span_1 in value_1:
                            if span_1._.rel == []:
                                continue
                            else:
                                for ent_1 in extract_rel(
                                    span_1, num_doc
                                ):  # De même ici
                                    if ent_0 == ent_1:
                                        matched = True
                                        scores["true"] += 1
                                        continue

                    if not matched:
                        scores["false"] += 1
                        error.append(ent_0)
    return scores, error


def perfect_match_rel(corpus_0: List[Doc], corpus_1: List[Doc]) -> Tuple:
    """Evaluation of NER and their attributes annotations between two annotated corpora


    Args:
        corpus_0 (List[Doc]): annotated corpus by the first annotator
        corpus_1 (List[Doc]): annotated corpus by the second annotator

    Returns:
        Tuple: Evaluation scores and list of errors
    """
    scores_liste, error_0, error_1 = [], [], []

    for num_doc, (doc_0, doc_1) in enumerate(zip(corpus_0, corpus_1)):
        scores = {"accord": 0, "error_0": 0, "error_1": 0, "total": 0}

        scores_0, error = scoring_rel(doc_0.spans, doc_1.spans, num_doc)
        error_0.extend(error)

        scores_1, error = scoring_rel(doc_1.spans, doc_0.spans, num_doc)
        error_1.extend(error)

        (
            scores["accord"],
            scores["error_0"],
            scores["error_1"],
            scores["total"],
        ) = (
            scores_0["true"],
            scores_0["false"],
            scores_1["false"],
            (scores_0["true"] + scores_0["false"] + scores_1["false"]),
        )
        scores_liste.append(scores)
    return scores_liste, error_0, error_1


def view_df_rel(error_liste: List[str]) -> pd.DataFrame:
    """visualize the errors in a DataFrame

    Args:
        error_liste (List[str]): List of errors

    Returns:
        pd.DataFrame: DataFrame of errors
    """
    return pd.DataFrame(
        error_liste,
        columns=[
            "Num_doc",
            "start_suj",
            "end_suj",
            "text_suj",
            "label_suj",
            "nature_rel",
            "start_obj",
            "end_obj",
            "text_obj",
            "label_obj",
        ],
    )
