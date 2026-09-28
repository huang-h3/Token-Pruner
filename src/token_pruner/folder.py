"""FOLDER, a token merging approach, loaded from the reference implementation.
Download from https://github.com/anakin-skywalker-Joseph/Folder
"""

import importlib.util
import os
from pathlib import Path

import torch

FOLDER_ROOT_ENV = "FOLDER_ROOT"


class folder_wrapper:
    def __init__(self, folder_root=None):
        folder_root = folder_root or os.environ.get(FOLDER_ROOT_ENV)
        if not folder_root:
            raise RuntimeError(
                f"FOLDER merging loads the reference folder.py. Set {FOLDER_ROOT_ENV} to the directory that contains it. Can be downloaded from https://github.com/anakin-skywalker-Joseph/Folder "
            )
        path = Path(folder_root).expanduser() / "folder.py"
        spec = importlib.util.spec_from_file_location("folder_reference", path)
        self.folder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.folder)

    def reduce(self, cls_token, frame_tokens, keep_patches_local):
        total = frame_tokens.shape[1]
        if keep_patches_local >= total:
            return frame_tokens
        merged, _, _ = self.folder.merge_features(
            torch.cat([cls_token, frame_tokens], dim=1),
            metric=None,
            size=None,
            r=total - keep_patches_local,
            class_token=True,
        )
        return merged[:, 1:, :]
