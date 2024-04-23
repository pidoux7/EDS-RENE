import sys
import math as m
import matplotlib.pyplot as plt
from spacy.tokens import Doc, Span
from typing import List, Tuple, Dict, Any
from edsnlp import edsnlp

sys.path.insert(0, "/home/pidoux/LIMICS/edsnlp")


# typer la classe et les fonctions
class model_proximity:
    def __init__(self):
        self.corpus = None

    def clean_rel(self, corpus: List[Doc]) -> List[Doc]:
        """supprime les relations dans le corpus

        Args:
            corpus (List[Doc]): Documents à nettoyer

        Returns:
            List[Doc]: Documents nettoyés
        """
        self.corpus = corpus
        for i, doc in enumerate(corpus):
            for k, v in doc.spans.items():
                for j, span in enumerate(v):
                    self.corpus[i].spans[k][j]._.rel = []
        return corpus

    def predict(
        self,
        corpus: List[Doc],
        max_dist: int,
        clean: bool = True,
        method: str = "start",
        sents=False,
    ) -> List[Doc]:
        """prédit les relations de dépendance entre les entités

        Args:
            corpus (List[Doc]): corpus à prédire
            max_dist (int): distance maximale pour laquelle
                            une relation de dépendance est possible en char
            clean (bool, optional): Nettoie les relations avant de prédire.
                                    Defaults to True.

        Returns:
            List[Doc]: corpus avec les relations prédites
        """
        self.corpus = corpus
        if clean:
            self.corpus = self.clean_rel(self.corpus)
        if sents:
            nlp = edsnlp.blank("eds")
            nlp.add_pipe("sentencizer")
            doc_iterator = nlp.pipe(corpus)
            corpus = list(doc_iterator)
            corpus.sort(key=lambda x: x.text)
            self.corpus = corpus

        for num_doc, doc in enumerate(self.corpus):
            for num_span_obj, span_obj in enumerate(doc.spans["Chemical_and_drugs"]):
                if span_obj._.Tech is not None:
                    if sents:
                        self.iterate_over_sub_sents(
                            num_doc,
                            num_span_obj,
                            span_obj,
                            "Chemical_and_drugs",
                            max_dist,
                            method,
                        )
                    else:
                        self.iterate_over_sub(
                            num_doc,
                            num_span_obj,
                            span_obj,
                            "Chemical_and_drugs",
                            max_dist,
                            method,
                        )
            for num_span_obj, span_obj in enumerate(doc.spans["Temporal"]):
                self.iterate_over_sub(
                    num_doc, num_span_obj, span_obj, "Temporal", max_dist, method
                )
        return self.corpus

    def iterate_over_sub_sents(
        self,
        num_doc: int,
        num_span_obj: int,
        span_obj: Span,
        label_obj: str,
        max_dist: int,
        method: str = "start",
    ) -> None:
        conserver = {"id_obj": None, "id_sub": None, "dist": m.inf}
        info_obj = {"start": span_obj.start_char, "end": span_obj.end_char}
        for sent in self.corpus[num_doc].sents:
            if (
                info_obj["start"] >= sent.start_char
                and info_obj["end"] <= sent.end_char
            ):
                info_obj["start_sent"] = sent.start_char
                info_obj["end_sent"] = sent.end_char
                break  # peut etre implementer l'exception

        for num_span_sub, span_sub in enumerate(
            self.corpus[num_doc].spans["Chemical_and_drugs"]
        ):
            if (
                span_sub._.Tech is None
                and span_sub.start_char >= info_obj["start_sent"]
                and span_sub.end_char <= info_obj["end_sent"]
            ):
                dist = self.distance(span_sub, span_obj, method)
                if dist < conserver["dist"] and dist != 0:
                    conserver["dist"] = dist
                    conserver["id_sub"] = num_span_sub
                    conserver["id_obj"] = num_span_obj
                    conserver["span_sub"] = span_sub
                    conserver["span_obj"] = span_obj
        if (
            conserver["dist"] <= max_dist
            and conserver["id_sub"] is not None
            and conserver["id_obj"] is not None
        ):
            self.corpus[num_doc].spans["Chemical_and_drugs"][
                conserver["id_sub"]
            ]._.rel.append({"nature": "Depend", "span": conserver["span_obj"]})

            self.corpus[num_doc].spans[label_obj][conserver["id_obj"]]._.rel.append(
                {"nature": "inv_Depend", "span": conserver["span_sub"]}
            )

    def iterate_over_sub(
        self,
        num_doc: int,
        num_span_obj: int,
        span_obj: Span,
        label_obj: str,
        max_dist: int,
        method: str = "start",
    ) -> None:
        """Itère sur les entités pour trouver les relations de dépendance

        Args:
            num_doc (int): Numero de document
            num_span_obj (int): Numero de l'entité objet dans le document
            span_obj (Span): Entité objet
            label_obj (str): label de l'entité objet
            max_dist (int): distance maximale pour laquelle une relation
                            de dépendance est possible en char
        """
        conserver = {"id_obj": None, "id_sub": None, "dist": m.inf}
        for num_span_sub, span_sub in enumerate(
            self.corpus[num_doc].spans["Chemical_and_drugs"]
        ):
            if span_sub._.Tech is None:
                dist = self.distance(span_sub, span_obj, method)
                if dist < conserver["dist"] and dist != 0:
                    conserver["dist"] = dist
                    conserver["id_sub"] = num_span_sub
                    conserver["id_obj"] = num_span_obj
                    conserver["span_sub"] = span_sub
                    conserver["span_obj"] = span_obj
        if (
            conserver["dist"] <= max_dist
            and conserver["id_sub"] is not None
            and conserver["id_obj"] is not None
        ):
            self.corpus[num_doc].spans["Chemical_and_drugs"][
                conserver["id_sub"]
            ]._.rel.append({"nature": "Depend", "span": conserver["span_obj"]})

            self.corpus[num_doc].spans[label_obj][conserver["id_obj"]]._.rel.append(
                {"nature": "inv_Depend", "span": conserver["span_sub"]}
            )

    def distance(self, span_sub: Span, span_obj: Span, method: str = "start") -> int:
        """Calcul la distance entre deux spans

        Args:
            span1 (Span): Span objet
            span2 (Span): Span sujet
            method (str, optional): Méthode de calcul de la distance.
                                    Defaults to "start".

        Returns:
            int: Distance entre les spans
        """
        if method == "start":
            return m.fabs(span_obj.start_char - span_sub.start_char)
        elif method == "end":
            return m.fabs(span_obj.end_char - span_sub.end_char)
        elif method == "middle":
            return m.fabs(
                (span_obj.start_char + span_obj.end_char) / 2
                - (span_sub.start_char + span_sub.end_char) / 2
            )
        elif method == "right":
            return m.fabs(span_obj.start_char - span_sub.end_char)
        elif method == "left":
            return m.fabs(span_obj.end_char - span_sub.start_char)

    def score(self, corpus_true: List[Doc], corpus_pred: List[Doc]) -> Dict[str, Any]:
        """Score les relations prédites

        Args:
            corpus_true (List[Doc]): corpus avec les vraies relations
            corpus_pred (List[Doc]): corpus avec les relations prédites

        Returns:
            Dict[str, Any]: Scores: TP, FP, FN, TOT (nombres d'entité itérée),
                                    precision, rappel, f1
        """
        self.corpus_true = corpus_true
        self.corpus_pred = corpus_pred
        scores = {"TP": 0, "FP": 0, "FN": 0, "TOT": 0}
        scores["TP"], scores["FN"], scores["TOT"] = self.iteration_eval(
            corpus_true, corpus_pred
        )
        _, scores["FP"], tot = self.iteration_eval(corpus_pred, corpus_true)
        scores["precision"] = (
            scores["TP"] / (scores["TP"] + scores["FP"])
            if scores["TP"] + scores["FP"] != 0
            else 0
        )
        scores["rappel"] = (
            scores["TP"] / (scores["TP"] + scores["FN"])
            if scores["TP"] + scores["FN"] != 0
            else 0
        )
        scores["f1"] = (
            2
            * (scores["precision"] * scores["rappel"])
            / (scores["precision"] + scores["rappel"])
            if scores["precision"] + scores["rappel"] != 0
            else 0
        )
        return scores

    def iteration_eval(
        self, corpus_t: List[Doc], corpus_p: List[Doc]
    ) -> Tuple[int, int, int]:
        """Itère sur les documents et les entités pour évaluer les relations

        Args:
            corpus_t (List[Doc]): corpus avec les vraies relations
            corpus_p (List[Doc]): corpus avec les relations prédites

        Returns:
            Tuple[int, int, int]: TP, FP, TOT
        """
        POS = 0
        NEG = 0
        TOT = 0
        # itere doc
        for i, doc in enumerate(corpus_t):
            for k, v in doc.spans.items():
                # itere span_true
                for j, span_true in enumerate(v):
                    TOT += 1
                    if len(span_true._.rel) >= 1:
                        true_tuple_s = (span_true.start_char, span_true.end_char)
                        true_label_s = span_true.label_
                        # itere rel_true
                        for rel_true in span_true._.rel:
                            matched = False
                            if rel_true["nature"][:4] != "inv_":
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
                                                    rel_pred["nature"]
                                                    == rel_true["nature"]
                                                    and rel_pred["span"].start_char
                                                    == rel_true["span"].start_char
                                                    and rel_pred["span"].end_char
                                                    == rel_true["span"].end_char
                                                    and rel_pred["span"].label_
                                                    == rel_true["span"].label_
                                                ):
                                                    POS += 1
                                                    matched = True
                                                    continue
                                if not matched:
                                    NEG += 1
        return POS, NEG, TOT

    def precision_recall_curve(
        self,
        corpus_true: List[Doc],
        corpus_pred: List[Doc],
        max_dist: int,
        pas: int,
        method: str = "start",
        sents=False,
    ) -> None:
        """Trace la courbe de précision-rappel

        Args:
            corpus_true (List[Doc]): corpus avec les vraies relations
            corpus_pred (List[Doc]): corpus avec les relations prédites
            max_dist (int): distance maximale pour le chart
            pas (int): espacement entre les points
        """
        distances = list(range(0, max_dist, pas))
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for dist in distances:
            predicted_corpus = self.predict(
                corpus_pred, dist, method=method, sents=sents
            )
            scores = self.score(corpus_true, predicted_corpus)
            precision_scores.append(scores["precision"])
            recall_scores.append(scores["rappel"])
            f1_scores.append(scores["f1"])

        plt.figure(figsize=(10, 5))
        plt.plot(distances, precision_scores, label="Precision")
        plt.plot(distances, recall_scores, label="Recall")
        plt.plot(distances, f1_scores, label="F1 Score")
        plt.xlabel("Distance Threshold")
        plt.ylabel("Scores")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(recall_scores, precision_scores, marker="o")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision vs. Recall Curve")
        plt.grid(True)
        plt.show()
