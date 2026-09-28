import unittest

import pytest
import torch

from token_pruner.selectors import select_group_topk
from token_pruner.shared_folding import (
    _coverage_hard_blocks,
    _global_hard_select,
)
from token_pruner.selection import distribute_budget


class SelectionEquivalenceTests(unittest.TestCase):
    def test_global_and_local_match_reference_topk_with_ties(self):
        scores = torch.tensor(
            [
                [4.0, 4.0, 1.0, 0.0],
                [4.0, 2.0, 4.0, 0.0],
                [3.0, 4.0, 4.0, 0.0],
            ]
        )
        groups = [0, 2]
        budget = 3
        candidates = torch.cat(
            [torch.arange(group * 4, (group + 1) * 4) for group in groups]
        )
        expected_global = candidates[
            torch.topk(scores[groups].reshape(-1), k=budget).indices
        ].sort().values
        self.assertTrue(
            torch.equal(
                select_group_topk(scores, groups, budget, 4, "global"),
                expected_global,
            )
        )

        budget = 3
        shares = distribute_budget(budget, groups)
        expected_local = torch.cat(
            [
                torch.topk(scores[group], k=shares[group]).indices.sort().values
                + group * 4
                for group in groups
            ]
        )
        self.assertTrue(
            torch.equal(
                select_group_topk(scores, groups, budget, 4, "local"),
                expected_local,
            )
        )

    def test_global_shared_folding_matches_stable_reference_selection(self):
        scores = torch.tensor([[2.0, 2.0, 1.0, 2.0, 0.0]])
        patches = torch.arange(1 * 1 * 5 * 2, dtype=torch.float32).reshape(1, 1, 5, 2)
        ranked = torch.argsort(scores, descending=True, stable=True)[:, :3]
        expected = patches.gather(2, torch.sort(ranked, dim=-1).values[:, None, :, None].expand(1, 1, 3, 2))
        actual = _global_hard_select(patches, scores, 3)
        self.assertTrue(torch.equal(actual, expected))

    def test_coverage_hard_matches_floor_then_global_reference_selection(self):
        score_blocks = torch.tensor(
            [[[5.0, 5.0, 1.0, 0.0], [5.0, 2.0, 2.0, 0.0]]]
        )
        spatial = torch.tensor([[[2, 0, 3, 1], [6, 4, 7, 5]]])
        patches = torch.arange(1 * 1 * 2 * 4 * 1, dtype=torch.float32).reshape(1, 1, 2, 4, 1)
        mandatory = torch.topk(score_blocks, k=1, dim=-1).indices
        mask = torch.zeros_like(score_blocks, dtype=torch.bool).scatter_(-1, mandatory, True)
        extra = torch.topk(
            score_blocks.masked_fill(mask, float("-inf")).reshape(1, -1),
            k=2,
            dim=-1,
        ).indices
        selected = torch.cat(
            [(mandatory + torch.tensor([0, 4]).view(1, 2, 1)).reshape(1, -1), extra], dim=1
        )
        selected_spatial = torch.gather(spatial.reshape(1, -1), 1, selected)
        expected = patches.reshape(1, 1, -1, 1).gather(
            2,
            selected.gather(1, selected_spatial.argsort(dim=-1))[:, None, :, None],
        )
        actual = _coverage_hard_blocks(patches, score_blocks, spatial, 2, 1)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()


def test_global_clamp_to_group_truncates_overflow_but_unclamped_raises():
    scores = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    with pytest.raises(RuntimeError, match="available selection capacity"):
        select_group_topk(scores, [0, 1], 10, 4, "global")
    selected = select_group_topk(
        scores,
        [0, 1],
        10,
        4,
        "global",
        clamp_to_group=True,
    )
    assert torch.equal(selected, torch.arange(8))


def test_global_floor_covers_every_block():
    scores = torch.tensor([[100.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    selected = select_group_topk(
        scores,
        [0, 1, 2],
        3,
        2,
        "global",
        floor=1,
    )
    assert selected.tolist() == [0, 2, 4]
