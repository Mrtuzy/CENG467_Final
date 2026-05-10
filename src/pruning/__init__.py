from .no_pruning import NoPruningPruner
from .naive_truncation import NaiveTruncationPruner
from .recomp import RecompPruner

PRUNER_REGISTRY: dict = {
    "no_pruning": NoPruningPruner,
    "naive_truncation": NaiveTruncationPruner,
    "recomp": RecompPruner,
}
