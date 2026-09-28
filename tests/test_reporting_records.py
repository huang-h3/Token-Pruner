from pathlib import Path
from unittest import mock

import pytest
import torch

from token_pruner.records import build_benchmark_result
from token_pruner.hevc_cache import prepare_hevc_batch_selection


def test_result_writer_rejects_overriding_reserved_sections():
    with pytest.raises(RuntimeError, match="reserved"):
        build_benchmark_result(
            timing={},
            memory={},
            params={},
            extra={"params": {"model_type": "unexpected"}},
        )


def test_all_failed_hevc_batch_does_not_call_selector(tmp_path):
    selector = mock.Mock()
    class FakeStore:
        def __init__(self, permanent):
            self.permanent_dir = Path(permanent)

        def get_or_encode(self, *_args, **_kwargs):
            return None

    with (
        mock.patch("token_pruner.hevc_cache.HevcArtifactStore", FakeStore),
    ):
        result = prepare_hevc_batch_selection(
            ["/videos/a.mp4", "/videos/b.mp4"],
            tmp_path, selector, torch.device("cpu"),
        )

    selector.assert_not_called()
    assert result.kept_indices == []
    assert result.failed_indices == [0, 1]
