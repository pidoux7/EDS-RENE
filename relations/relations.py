from typing import Dict, Iterable, List, Union

from loguru import logger

from spacy.tokens import Doc, Span

from edsnlp.core import PipelineProtocol
from edsnlp.pipes.misc.relations import patterns
import math as m


class RelationsMatcher:
    """ """

    def __init__(
        self,
        nlp: PipelineProtocol,
        name: str = "relations",
        *,
        attr: Union[List[Dict[str, str]], Dict[str, str]] = None,
        use_sentences: bool = False,
        proximity_method: str = "right",
        clean_rel: bool = True,
        max_dist: int = 45,
    ):
        self.nlp = nlp
        if not isinstance(name, str):
            raise ValueError("name must be a string")
        self.name = name

        if attr is None:
            attr = patterns.relations
        if isinstance(attr, dict):
            attr = [attr]
        if not self.check_attr(attr):
            raise ValueError(
                """"
                attr must be a list of dictionaries with
                keys 'sub', 'obj', 'type', 'inv_type'
                """
            )
        self.attr = attr

        if not isinstance(use_sentences, bool):
            raise ValueError("use_sentences must be a boolean")
        self.use_sentences = use_sentences and (
            "eds.sentences" in nlp.pipe_names or "sentences" in nlp.pipe_names
        )
        if use_sentences and not self.use_sentences:
            logger.warning(
                "You have requested that the pipeline use annotations "
                "provided by the `eds.sentences` pipeline, but it was not set. "
                "Skipping that step."
            )

        if self.proximity_method not in ["start", "end", "middle", "right", "left"]:
            raise ValueError(
                """proximity_method must be one of 'start',
                'end', 'middle', 'right', 'left'"""
            )
        self.proximity_method = proximity_method

        if not isinstance(clean_rel, bool):
            raise ValueError("clean_rel must be a boolean")
        self.clean_rel = clean_rel

        if not isinstance(max_dist, int):
            raise ValueError("max_dist must be an integer")
        self.max_dist = max_dist

        self.set_extensions()

    def check_attr(self, attr: Union[List[Dict[str, str]], Dict[str, str]]) -> bool:
        """Exemple
        attr = [
            {'subject': [
                {'label': 'Chemical_and_drugs', 'attributs': None}
                ],
            'object': [
                {'label': 'Chemical_and_drugs', 'attributs': 'Tech'}
                ],
            'type': 'Depend', 'inv_type': 'inv_Depend'},
            ]"""
        # verify that the format and type correspond to the expected format
        for a in attr:
            if not isinstance(a, dict):
                return False
            if not all(key in a for key in ["subject", "object", "type", "inv_type"]):
                return False
            if not isinstance(a["subject"], list) or not isinstance(a["object"], list):
                return False
        return True

    @classmethod
    def set_extensions(cls) -> None:
        if not Span.has_extension("rel"):
            Span.set_extension("rel", default=[])

    def _enhance_with_sentences(
        self, subjects: Iterable, objects: Iterable, sentences: Iterable
    ) -> List:
        """_summary_

        Args:
            subjects (Iterable): _description_
            objects (Iterable): _description_
            sentences (Iterable): _description_

        Returns:
            List: _description_
        """
        return []

    def __call__(self, doc: Doc) -> Doc:
        """find the relations in the doc based on the proximity of the attributes

        Args:
            doc (Doc): the doc to be processed

        Returns:
            Doc: the doc with the relations added
        """
        if self.clean_rel:
            doc = self.clean_relations(doc)

        doc = self.find_relations(doc)

        if self.use_sentences:
            sentences = doc.sents

        return sentences, doc

    def find_relations(self, doc: Doc) -> Doc:
        """_summary_

        Args:
            doc (Doc): _description_

        Returns:
            Doc: _description_
        """
        for dict_attr in self.attr:
            for dict_attr_obj in dict_attr["object"]:
                for i, label_obj in enumerate(dict_attr_obj["label"]):
                    if label_obj in doc.spans:
                        for num_span_obj, span_obj in doc.spans[label_obj]:
                            if span_obj._.Tech is not None:
                                self.iterate_over(
                                    doc, num_span_obj, span_obj, label_obj
                                )

        return doc

    def iterate_over(
        self,
        doc: Doc,
        num_span_obj: int,
        span_obj: Span,
        label_obj: str,
    ) -> None:
        """Itère sur les entités pour trouver les relations de dépendance

        Args:
            num_doc (int): Numero de document
            num_span_obj (int): Numero de l'entité objet dans le document
            span_obj (Span): Entité objet
            label_obj (str): label de l'entité objet
            max_dist (int): distance maximale pour laquelle une relation
                            de dépendance est possible en char
            method (str): Méthode de calcul de la distance
        """
        conserver = {"id_obj": None, "id_sub": None, "dist": m.inf}
        for num_span_sub, span_sub in enumerate(doc.spans["Chemical_and_drugs"]):
            if span_sub._.Tech is None:
                dist = self.distance(span_sub, span_obj, self.proximity_method)
                if dist < conserver["dist"] and dist != 0:
                    conserver["dist"] = dist
                    conserver["id_sub"] = num_span_sub
                    conserver["id_obj"] = num_span_obj
                    conserver["span_sub"] = span_sub
                    conserver["span_obj"] = span_obj
        if (
            conserver["dist"] <= self.max_dist
            and conserver["id_sub"] is not None
            and conserver["id_obj"] is not None
        ):
            doc.spans["Chemical_and_drugs"][conserver["id_sub"]]._.rel.append(
                {"type": "Depend", "target": conserver["span_obj"]}
            )

            doc.spans[label_obj][conserver["id_obj"]]._.rel.append(
                {"type": "inv_Depend", "target": conserver["span_sub"]}
            )

        def distance(
            self, span_sub: Span, span_obj: Span, method: str = "start"
        ) -> int:
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

    def clean_relations(self, doc: Doc) -> Doc:
        """Remove the relations from the doc

        Args:
            doc (Doc): the doc to be processed

        Returns:
            Doc: the doc with the relations removed
        """
        for label, spans in doc.spans.items():
            for span in spans:
                if span._.rel:
                    span._.rel = []
        return doc
