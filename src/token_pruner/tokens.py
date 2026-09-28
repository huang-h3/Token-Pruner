"""Token containers and pruning configuration, independent of any model."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


class PruningScope(str, Enum):
    """How a token budget or selection map is shared across temporal groups."""

    LOCAL = "local"
    GLOBAL = "global"
    SHARED = "shared"


class TokenReducer(str, Enum):
    """Operation used to reduce content tokens."""

    PRESERVE = "preserve"  # for anchor groups
    GATHER = "gather"  # top-k
    FOLDER = "folder"
    GLOBAL_FOLDER = "global_folder"
    SHARED_FOLDING = "shared_folding"


class SignalSource(str, Enum):
    """Source used to construct a token-selection signal."""

    NONE = "none"
    HEVC = "hevc"
    RANDOM = "random"
    UNIFORM = "uniform"


def _as_enum(value, enum_type):
    return value if isinstance(value, enum_type) else enum_type(str(value))


@dataclass(frozen=True)
class SharedFoldingOptions:
    """Shared-folding options."""

    mode: str = "temporal-diff"
    patch_width: int | None = None
    block_size: int = 2
    temperature: float = 1.0  # for softmax
    slots_per_block: int = 1
    pooling: str = "weighted"
    residual_scale: float = 0.2
    assignment: str = "similarity"
    min_slots_per_block: int = 1
    score_alpha: float = 0.5
    score_normalization: str = "auto"
    temporal_feature_norm: str = "layernorm"
    temporal_spatial_normalization: str = "none"
    block_wise: bool = True
    global_k: int | None = None


@dataclass(frozen=True)
class PruningConfig:
    """Model-independent token reduction plan."""

    scope: PruningScope | str
    reducer: TokenReducer | str
    anchor_reducer: TokenReducer | str = TokenReducer.FOLDER
    keep_per_partition: int | None = None
    keep_total: int | None = None
    signal_source: SignalSource | str = SignalSource.NONE
    shared_folding: SharedFoldingOptions | None = None
    uniform_partitions: bool = False
    min_per_partition: int = 0

    def __post_init__(self):
        object.__setattr__(self, "min_per_partition", int(self.min_per_partition))
        for name, enum_type in (
            ("scope", PruningScope),
            ("reducer", TokenReducer),
            ("anchor_reducer", TokenReducer),
            ("signal_source", SignalSource),
        ):
            object.__setattr__(
                self,
                name,
                _as_enum(getattr(self, name), enum_type),
            )
        if (
            self.reducer == TokenReducer.GATHER
            and self.signal_source == SignalSource.NONE
        ):
            raise RuntimeError("gather requires a selection signal.")
        if (
            self.reducer not in {TokenReducer.GATHER, TokenReducer.SHARED_FOLDING}
            and self.signal_source != SignalSource.NONE
        ):
            raise RuntimeError(
                f"{self.reducer.value} does not consume a selection signal."
            )
        if self.reducer == TokenReducer.GLOBAL_FOLDER and self.scope != PruningScope.GLOBAL:
            raise RuntimeError("global_folder requires global scope.")
        if self.reducer == TokenReducer.SHARED_FOLDING and self.shared_folding is None:
            raise RuntimeError("shared_folding reducer requires shared_folding options.")


@dataclass
class PruningSignal:
    """Selection information prepared before token reduction."""

    visible_indices: Any = None
    scores: torch.Tensor | None = None
    anchor_partitions: torch.Tensor | None = None

    def to(self, device):
        """Move tensor fields while preserving ragged Python containers."""

        for name in ("visible_indices", "scores", "anchor_partitions"):
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
            elif isinstance(value, list):
                setattr(
                    self,
                    name,
                    [
                        item.to(device) if isinstance(item, torch.Tensor) else item
                        for item in value
                    ],
                )
        return self


def _token_samples(tokens):
    """One ``[N, D]`` tensor per sample, from any of the accepted layouts."""

    if not isinstance(tokens, torch.Tensor):
        return list(tokens)
    return [tokens] if tokens.ndim == 2 else list(tokens.unbind(0))


def _stack_if_uniform(samples):
    lengths = {int(sample.shape[0]) for sample in samples}
    return torch.stack(samples, dim=0) if len(lengths) == 1 else samples


@dataclass
class PruneTokens:
    """Flat or ragged content tokens plus optional topology metadata."""

    tokens: torch.Tensor | list[torch.Tensor]
    partition_ids: torch.Tensor | list[torch.Tensor] | None = None
    cls_tokens: torch.Tensor | None = None

    def __post_init__(self):
        self._samples = _token_samples(self.tokens)
        ids = self.partition_ids
        if ids is None:
            rows = [
                torch.zeros(sample.shape[0], dtype=torch.long, device=sample.device)
                for sample in self._samples
            ]
        elif isinstance(ids, torch.Tensor):
            rows = [ids] * len(self._samples) if ids.ndim == 1 else list(ids.unbind(0))
        else:
            rows = list(ids)
        self._partition_rows = [
            torch.as_tensor(row, dtype=torch.long, device=sample.device).flatten()
            for row, sample in zip(rows, self._samples)
        ]

    @classmethod
    def from_partitioned(cls, tokens, cls_tokens=None):
        """Adapt dense ``[B, P, N, D]`` tokens to the flat core interface."""
        batch_size, partitions, items, hidden_dim = tokens.shape
        partition_ids = torch.arange(
            partitions, device=tokens.device
        ).repeat_interleave(items).unsqueeze(0).expand(batch_size, -1)
        return cls(
            tokens=tokens.reshape(batch_size, partitions * items, hidden_dim),
            partition_ids=partition_ids,
            cls_tokens=cls_tokens,
        )

    @property
    def samples(self):
        return self._samples

    @property
    def partition_rows(self):
        return self._partition_rows

    @property
    def batch_size(self):
        return len(self._samples)

    @property
    def hidden_dim(self):
        return int(self._samples[0].shape[-1])

    @property
    def partition_counts(self):
        return [
            int(row.max().item()) + 1 if row.numel() else 0
            for row in self._partition_rows
        ]

    def cls_for_partition(self, batch_idx, partition_idx):
        if self.cls_tokens.ndim == 3:
            return self.cls_tokens[batch_idx : batch_idx + 1]
        return self.cls_tokens[batch_idx : batch_idx + 1, partition_idx]

    def cls_for_partitions(self, batch_idx, partition_indices):
        if self.cls_tokens.ndim == 3:
            return self.cls_tokens[batch_idx : batch_idx + 1]
        return self.cls_tokens[batch_idx, partition_indices].mean(dim=0, keepdim=True)


class PruneResult:
    """Flat/ragged pruning output with derived partition views."""

    def __init__(self, samples, partition_ids=None, partition_counts=None,
                 cls_tokens=None, source_indices=None):
        self.samples = list(samples)
        if partition_ids is None:
            partition_ids = [
                torch.zeros(sample.shape[0], dtype=torch.long, device=sample.device)
                for sample in self.samples
            ]
        elif isinstance(partition_ids, torch.Tensor):
            partition_ids = list(partition_ids.unbind(0))
        self.partition_ids = list(partition_ids)
        if partition_counts is None:
            partition_counts = [
                int(row.max().item()) + 1 if row.numel() else 0
                for row in self.partition_ids
            ]
        self.partition_counts = list(partition_counts)
        self.cls_tokens = cls_tokens
        self.source_indices = source_indices

    @property
    def partitions(self):
        return [
            [sample[row == partition] for partition in range(count)]
            for sample, row, count in zip(
                self.samples, self.partition_ids, self.partition_counts
            )
        ]

    @property
    def partition_token_counts(self):
        return [
            [int(partition.shape[0]) for partition in sample]
            for sample in self.partitions
        ]

    def require_uniform_partitions(self):
        return torch.stack(
            [torch.stack(sample, dim=0) for sample in self.partitions],
            dim=0,
        )

    def pack_with_global_cls(self):
        samples = [
            torch.cat((self.cls_tokens[idx], sample), dim=0)
            for idx, sample in enumerate(self.samples)
        ]
        return _stack_if_uniform(samples)
