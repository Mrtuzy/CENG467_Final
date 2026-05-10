from .no_pruning import NoPruningPruner
from .naive_truncation import NaiveTruncationPruner
from .recomp import RecompPruner
from .llmlingua2 import LLMLingua2Pruner
from .provence import ProvencePruner
from .history_pruning import HistoryPruner
from .combined import CombinedPruner

PRUNER_REGISTRY: dict = {
    "no_pruning": NoPruningPruner,
    "naive_truncation": NaiveTruncationPruner,
    "recomp": RecompPruner,
    "llmlingua2": LLMLingua2Pruner,
    "provence": ProvencePruner,
    "history_pruning": HistoryPruner,
    "combined": CombinedPruner,
}
