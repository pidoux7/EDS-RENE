import sys
import math as m
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/pidoux/LIMICS/edsnlp")


class model_proximity:
    def __init__(self):
        self.corpus = None

    def clean_rel(self, corpus):
        self.corpus = corpus
        for i, doc in enumerate(corpus):
            for k, v in doc.spans.items():
                for j, span in enumerate(v):
                    corpus[i].spans[k][j]._.rel = []
        return corpus

    def predict(self, corpus, max_dist=50, clean=True):
        self.corpus = corpus
        if clean:
            self.corpus = self.clean_rel(self.corpus)
        for num_doc, doc in enumerate(self.corpus):
            for num_span_obj, span_obj in enumerate(doc.spans["Chemical_and_drugs"]):
                if span_obj._.Tech is not None:
                    self.iterate_over_sub(
                        num_doc,
                        num_span_obj,
                        span_obj,
                        "Chemical_and_drugs",
                        max_dist,
                    )
            for num_span_obj, span_obj in enumerate(doc.spans["Temporal"]):
                self.iterate_over_sub(
                    num_doc, num_span_obj, span_obj, "Temporal", max_dist
                )
        return self.corpus

    def iterate_over_sub(self, num_doc, num_span_obj, span_obj, label_obj, max_dist=50):
        conserver = {"id_obj": None, "id_sub": None, "dist": m.inf}
        for num_span_sub, span_sub in enumerate(
            self.corpus[num_doc].spans["Chemical_and_drugs"]
        ):
            if span_sub._.Tech is None:
                dist = m.fabs(span_obj.start_char - span_sub.start_char)
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

    def score(self, corpus_true, corpus_pred):
        self.corpus_true = corpus_true
        self.corpus_pred = corpus_pred
        scores = {"TP": 0, "FP": 0, "FN": 0, "TOT": 0}
        scores["TP"], scores["FN"], scores["TOT"] = self.iteration_eval(
            corpus_true, corpus_pred
        )
        _, scores["FP"], tot = self.iteration_eval(corpus_pred, corpus_true)
        scores["TOT"] += tot
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

    def precision_recall_curve(self, corpus_true, corpus_pred, max_dist=100, pas=5):
        distances = list(range(0, max_dist, pas))
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for dist in distances:
            predicted_corpus = self.predict(corpus_pred, dist)
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

    def iteration_eval(self, corpus_t, corpus_p):
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
