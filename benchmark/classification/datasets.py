"""Kinetics-400 and Something-Something-v2 classification datasets."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from benchmark.paths import DATA_ROOT


def _clean_label(label):
    return str(label).strip().strip('"').replace("[", "").replace("]", "")


def _unique_in_order(values):
    seen = set()
    unique = []
    for value in values:
        value = _clean_label(value)
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _is_generic_label_map(label_to_id):
    return all(label == f"LABEL_{idx}" for label, idx in label_to_id.items())


def build_label_to_id(labels, model_config=None, explicit_label_to_id=None):
    """Build a dataset label-to-id map aligned with a model head when possible."""
    labels = _unique_in_order(labels)

    explicit_label_to_id = {
        _clean_label(label): int(idx)
        for label, idx in (explicit_label_to_id or {}).items()
    }

    if model_config is None:
        if explicit_label_to_id:
            return explicit_label_to_id
        return {label: idx for idx, label in enumerate(labels)}

    model_label_to_id = {
        _clean_label(label): int(idx)
        for label, idx in model_config.label2id.items()
    }
    if labels and all(label in model_label_to_id for label in labels):
        return model_label_to_id

    model_label_to_id_lower = {label.lower(): idx for label, idx in model_label_to_id.items()}
    if labels and all(label.lower() in model_label_to_id_lower for label in labels):
        return {label: model_label_to_id_lower[label.lower()] for label in labels}

    if _is_generic_label_map(model_label_to_id):
        if explicit_label_to_id and len(explicit_label_to_id) == int(model_config.num_labels):
            return explicit_label_to_id
        if len(labels) == int(model_config.num_labels):
            print("Model uses generic LABEL_i names; using dataset label order for accuracy labels.")
            return {label: idx for idx, label in enumerate(labels)}

    raise RuntimeError("Cannot align dataset labels with model labels.")


class _VideoClassificationDataset(Dataset):
    name = None
    default_split = None

    def _add_samples(self, rows, video_name_for, sample_limit=None):
        for row in rows:
            video_name = video_name_for(row)
            video_path = self.video_root / video_name
            if not video_path.exists():
                self.skipped_videos.append(video_name)
                continue

            label = row["label"]
            self.samples.append(
                {
                    "video_path": video_path,
                    "label": self.label_to_id[label],
                }
            )
            if sample_limit is not None and len(self.samples) >= sample_limit:
                break

    def _finalize(self, random_samples, seed, label_mapping_path):
        if random_samples is not None:
            n = min(int(random_samples), len(self.samples))
            rng = np.random.default_rng(seed)
            selected = rng.choice(len(self.samples), size=n, replace=False)
            self.samples = [self.samples[int(i)] for i in selected]

        self.write_label_mapping(label_mapping_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def write_label_mapping(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            for label, label_id in sorted(self.label_to_id.items(), key=lambda item: item[1]):
                f.write(f"{label_id}\t{label}\n")


class KineticsDataset(_VideoClassificationDataset):
    name = "k400"
    default_root = DATA_ROOT / "kinetics400"
    default_split = "test"

    def __init__(
        self,
        model_config=None,
        split=None,
        data_root=None,
        random_samples=None,
        seed=42,
        csv_path=None,
        video_root=None,
        label_to_id=None,
    ):
        self.root = Path(data_root) if data_root is not None else self.default_root
        self.split = split or self.default_split
        self.csv_path = Path(csv_path) if csv_path is not None else self.root / "csv" / f"{self.split}.csv"
        self.video_root = Path(video_root) if video_root is not None else self.root / "videos"
        self.samples = []
        self.skipped_videos = []

        rows = []
        labels = []
        with self.csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["label"] = _clean_label(row["label"])
                labels.append(row["label"])
                rows.append(row)

        self.label_to_id = (
            {_clean_label(label): int(idx) for label, idx in label_to_id.items()}
            if label_to_id is not None
            else build_label_to_id(labels, model_config)
        )

        row_iter = rows
        sample_limit = int(random_samples) if random_samples is not None else None
        if sample_limit is not None:
            rng = np.random.default_rng(seed)
            row_iter = [rows[int(i)] for i in rng.permutation(len(rows))]

        self._add_samples(
            row_iter,
            lambda row: (
                f"{row['youtube_id']}_"
                f"{int(row['time_start']):06d}_"
                f"{int(row['time_end']):06d}.mp4"
            ),
            sample_limit,
        )

        self._finalize(None, seed, self.csv_path.parent / "converted_labels.txt")


class SSv2Dataset(_VideoClassificationDataset):
    name = "ssv2"
    default_root = DATA_ROOT / "ssv2"
    default_split = "test"

    def __init__(
        self,
        model_config=None,
        split=None,
        data_root=None,
        random_samples=None,
        seed=42,
    ):
        self.root = Path(data_root) if data_root is not None else self.default_root
        self.split = split or self.default_split
        self.video_root = self.root / "videos"
        self.labels_root = self.root / "labels"
        self.csv_root = self.root / "csv"
        self.samples = []
        self.skipped_videos = []

        official_label_to_id = self._read_official_label_to_id()
        rows = self._read_split_rows()
        labels = [row["label"] for row in rows]
        self.label_to_id = build_label_to_id(
            labels,
            model_config,
            explicit_label_to_id=official_label_to_id,
        )

        self._add_samples(rows, lambda row: f"{row['id']}.webm")

        self.write_csv_manifest(self.csv_root / f"{self.split}.csv", rows)
        self._finalize(random_samples, seed, self.csv_root / "converted_labels.txt")

    def _read_official_label_to_id(self):
        labels_path = self.labels_root / "labels.json"
        with labels_path.open() as f:
            labels = json.load(f)
        return {_clean_label(label): int(idx) for label, idx in labels.items()}

    def _read_split_rows(self):
        if self.split == "test" and (self.labels_root / "test-answers.csv").is_file():
            return self._read_test_answers()

        split_path = self.labels_root / f"{self.split}.json"
        with split_path.open() as f:
            rows = json.load(f)

        normalized_rows = []
        for row in rows:
            if "template" in row:
                label = _clean_label(row["template"])
            elif "label" in row:
                label = _clean_label(row["label"])
            else:
                raise RuntimeError(
                    f"SSv2 split {self.split} does not contain labels. "
                    "Use validation/train or provide test-answers.csv."
                )
            normalized_rows.append({"id": str(row["id"]), "label": label})
        return normalized_rows

    def _read_test_answers(self):
        rows = []
        answers_path = self.labels_root / "test-answers.csv"
        with answers_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                video_id, label = line.split(";", 1)
                rows.append({"id": video_id, "label": _clean_label(label)})
        return rows

    def write_csv_manifest(self, output_path, rows):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=("video", "label"))
            writer.writeheader()
            for row in rows:
                writer.writerow({"video": f"{row['id']}.webm", "label": row["label"]})


DATASET_REGISTRY = {
    "k400": KineticsDataset,
    "kinetics400": KineticsDataset,
    "ssv2": SSv2Dataset,
    "something-something-v2": SSv2Dataset,
}
DATASET_NAMES = tuple(DATASET_REGISTRY.keys())


def build_dataset(dataset_name, model_config=None, random_samples=None, seed=42):
    dataset_key = str(dataset_name).lower()
    return DATASET_REGISTRY[dataset_key](
        model_config=model_config,
        random_samples=random_samples,
        seed=seed,
    )


def collate_video_batch(samples):
    return {
        "video_paths": [str(sample["video_path"]) for sample in samples],
        "labels": torch.tensor([sample["label"] for sample in samples]),
    }
