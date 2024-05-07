import sys
from spacy.tokens import Doc
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from sklearn.metrics import auc

import edsnlp

sys.path.insert(0, "/home/pidoux/edsnlp")
sys.path.append("/home/pidoux/LIMICS/EDS-RENE/")


def score(corpus_true: List[Doc], corpus_pred: List[Doc]) -> Dict[str, Any]:
    """Score les relations prédites

    Args:
        corpus_true (List[Doc]): corpus avec les vraies relations
        corpus_pred (List[Doc]): corpus avec les relations prédites

    Returns:
        Dict[str, Any]: Scores: TP, FP, FN, TOT (nombres d'entité itérée),
                                precision, rappel, f1
    """
    corpus_pred = list(corpus_pred)
    dic1 = iteration_eval(corpus_true, corpus_pred)
    dic2 = iteration_eval(corpus_pred, corpus_true)
    dic = {}
    for k, v in dic1.items():
        dic[k] = {"TP": dic1[k]["pos"], "FN": dic1[k]["neg"]}
    for k, v in dic2.items():
        dic[k]["FP"] = dic2[k]["neg"]
    dic = calculate_prf(dic)
    return dic


def calculate_prf(dictionnaire: dict) -> dict:
    for k, v in dictionnaire.items():
        dictionnaire[k]["precision"] = (
            dictionnaire[k]["TP"] / (dictionnaire[k]["TP"] + dictionnaire[k]["FP"])
            if dictionnaire[k]["TP"] + dictionnaire[k]["FP"] != 0
            else 0
        )
        dictionnaire[k]["rappel"] = (
            dictionnaire[k]["TP"] / (dictionnaire[k]["TP"] + dictionnaire[k]["FN"])
            if dictionnaire[k]["TP"] + dictionnaire[k]["FN"] != 0
            else 0
        )
        dictionnaire[k]["f1"] = (
            2
            * (dictionnaire[k]["precision"] * dictionnaire[k]["rappel"])
            / (dictionnaire[k]["precision"] + dictionnaire[k]["rappel"])
            if dictionnaire[k]["precision"] + dictionnaire[k]["rappel"] != 0
            else 0
        )
    return dictionnaire


def iteration_eval(corpus_t: List[Doc], corpus_p: List[Doc]) -> Tuple[int, int, int]:
    """Itère sur les documents et les entités pour évaluer les relations

    Args:
        corpus_t (List[Doc]): corpus avec les vraies relations
        corpus_p (List[Doc]): corpus avec les relations prédites

    Returns:
        Tuple[int, int, int]: TP, FP, TOT
    """
    dic = {
        "total": {"pos": 0, "neg": 0, "tot": 0},
        "dosage": {"pos": 0, "neg": 0},
        "form": {"pos": 0, "neg": 0},
        "route": {"pos": 0, "neg": 0},
        "strength": {"pos": 0, "neg": 0},
        "Temporal": {"pos": 0, "neg": 0},
        "Duration": {"pos": 0, "neg": 0},
        "Frequency": {"pos": 0, "neg": 0},
        "Time": {"pos": 0, "neg": 0},
        "Date": {"pos": 0, "neg": 0},
    }
    # itere doc
    for i, doc in enumerate(corpus_t):
        for k, v in doc.spans.items():
            # itere span_true
            for j, span_true in enumerate(v):
                dic["total"]["tot"] += 1
                if len(span_true._.rel) >= 1:
                    true_tuple_s = (span_true.start_char, span_true.end_char)
                    true_label_s = span_true.label_
                    # itere rel_true
                    for rel_true in span_true._.rel:
                        matched = False
                        if rel_true["type"][:4] != "inv_":
                            # itere span_pred
                            for span_pred in corpus_p[i].spans[k]:
                                if len(span_pred._.rel) >= 1:
                                    pred_tuple_s = (
                                        span_pred.start_char,
                                        span_pred.end_char,
                                    )
                                    pred_label_s = span_pred.label_
                                    if (
                                        pred_tuple_s == true_tuple_s
                                        and true_label_s == pred_label_s
                                    ):
                                        # itere rel_pred
                                        for rel_pred in span_pred._.rel:
                                            if (
                                                rel_pred["type"] == rel_true["type"]
                                                and rel_pred["target"].start_char
                                                == rel_true["target"].start_char
                                                and rel_pred["target"].end_char
                                                == rel_true["target"].end_char
                                                and rel_pred["target"].label_
                                                == rel_true["target"].label_
                                            ):
                                                if (
                                                    rel_true["target"].label_
                                                    == "Chemical_and_drugs"
                                                    and rel_true["target"]._.Tech
                                                    is not None
                                                ):
                                                    dic[rel_true["target"]._.Tech][
                                                        "pos"
                                                    ] += 1
                                                if (
                                                    rel_true["target"].label_
                                                    == "Temporal"
                                                ):
                                                    dic[rel_true["target"].label_][
                                                        "pos"
                                                    ] += 1
                                                    if (
                                                        rel_true["target"]._.AttTemp
                                                        is not None
                                                    ):
                                                        dic[
                                                            rel_true["target"]._.AttTemp
                                                        ]["pos"] += 1
                                                dic["total"]["pos"] += 1
                                                matched = True
                                                continue
                            if not matched:
                                if (
                                    rel_true["target"].label_ == "Chemical_and_drugs"
                                    and rel_true["target"]._.Tech is not None
                                ):
                                    dic[rel_true["target"]._.Tech]["neg"] += 1
                                if rel_true["target"].label_ == "Temporal":
                                    dic[rel_true["target"].label_]["neg"] += 1
                                    if rel_true["target"]._.AttTemp is not None:
                                        dic[rel_true["target"]._.AttTemp]["neg"] += 1
                                dic["total"]["neg"] += 1
    return dic


def precision_recall_curve(
    corpus_true: List[Doc],
    corpus_pred: List[Doc],
    max_dist: tuple[int, int] = 50,
    pas: int = 1,
    label: str = "total",
    method: str = "start",
    sents=False,
) -> None:
    """Trace la courbe de précision-rappel

    Args:
        corpus_true (List[Doc]): corpus avec les vraies relations
        corpus_pred (List[Doc]): corpus avec les relations prédites
        max_dist (int): distance maximale pour le chart
        pas (int): espacement entre les points
        label (str, optional): label de l'entité à évaluer.
                                Defaults to "total".
        method (str, optional): Méthode de calcul de la distance.
                                Defaults to "start".
        sents (bool, optional): Si True, itère sur les phrases.
                                Defaults to False.
    """

    distances = list(range(0, max_dist, pas))
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for dist in distances:
        nlp = edsnlp.blank("eds")
        nlp.add_pipe("eds.sentences")
        nlp.add_pipe(
            "eds.relations",
            config={
                "scheme": "./relations.json",
                "use_sentences": sents,
                "clean_rel": True,
                "proximity_method": method,
                "max_dist": dist,
            },
        )
        predicted_corpus = nlp.pipe(corpus_pred)
        scores = score(corpus_true, predicted_corpus)
        precision_scores.append(scores[label]["precision"])
        recall_scores.append(scores[label]["rappel"])
        f1_scores.append(scores[label]["f1"])

    plt.figure(figsize=(10, 5))
    plt.plot(distances, precision_scores, label="Precision")
    plt.plot(distances, recall_scores, label="Recall")
    plt.plot(distances, f1_scores, label="F1 Score")
    plt.xlabel("Distance Threshold")
    plt.ylabel("Scores")
    plt.title(f"Precision-Recall-F1 score {label}")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(recall_scores, precision_scores, marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision Recall Curve: {auc(recall_scores, precision_scores)}")
    plt.grid(True)
    plt.show()
