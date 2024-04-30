from edsnlp.core import registry

from .relations import RelationsMatcher

DEFAULT_CONFIG = dict(
    attr="TEXT",
    use_sentences=False,
    clean_rel=True,
    max_dist=45,
)

create_component = registry.factory.register(
    "eds.relations",
    assigns=["doc.spans", "doc.ents"],
    deprecated=["relations"],
)(RelationsMatcher)
