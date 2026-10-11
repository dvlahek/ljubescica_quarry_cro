#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATE_RE = re.compile(r"^(\d{4}-\d{2})_(visual_rgb|mask)\.png$", re.IGNORECASE)


@dataclass
class Pair:
    date: str
    rgb: Path
    mask: Path
    kind: str


@dataclass
class ResultRow:
    experiment: str
    classifier: str
    train_source: str
    test_source: str
    train_dates: str
    test_date: str
    feature_source: str
    n_train_pixels: int
    n_test_pixels: int
    accuracy: float
    precision_quarry: float
    recall_quarry: float
    f1_quarry: float
    iou_quarry: float
    tn: int
    fp: int
    fn: int
    tp: int
    note: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RF/SVM checks for Ljubeščica quarry masks.")
    p.add_argument("--data_dir", default="quarry_timelapse_science_qc_v8")
    p.add_argument("--manual_dates", default="2017-07,2025-08")
    p.add_argument("--exclude_dates", default="2018-07,2022-07")
    p.add_argument("--months", default="3,4,5,6,7,8,9,10")
    p.add_argument("--n_train_per_class_per_image", type=int, default=5000)
    p.add_argument("--n_test_per_class", type=int, default=0)
    p.add_argument("--rf_trees", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_prefix", default="benchmark_manual_and_pseudo")
    p.add_argument("--skip_manual_cross_year", action="store_true")
    p.add_argument("--skip_pseudo_to_manual", action="store_true")
    return p.parse_args()


def split_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def allowed_month(date: str, months: Optional[set[int]]) -> bool:
    return True if months is None else int(date.split("-")[1]) in months


def discover_pairs(data_dir: Path, manual_dates: Sequence[str], exclude_dates: Sequence[str], months: Optional[set[int]]) -> List[Pair]:
    files: dict[str, dict[str, Path]] = {}
    for path in data_dir.iterdir():
        match = DATE_RE.match(path.name)
        if not match or not allowed_month(match.group(1), months):
            continue
        date, kind = match.group(1), match.group(2).lower()
        files.setdefault(date, {})[kind] = path

    manual_set = set(manual_dates)
    exclude_set = set(exclude_dates)
    pairs: List[Pair] = []
    for date in sorted(files):
        record = files[date]
        if "visual_rgb" not in record or "mask" not in record:
            continue
        if date in exclude_set and date not in manual_set:
            continue
        pairs.append(Pair(date, record["visual_rgb"], record["mask"], "manual" if date in manual_set else "pseudo"))
    return pairs


def load_rgb(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0).reshape(-1, 3)


def load_mask(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L")) > 127).astype(np.uint8).reshape(-1)


def sample_balanced(X: np.ndarray, y: np.ndarray, n_per_class: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if n_per_class <= 0:
        return X, y
    idx = []
    for cls in [0, 1]:
        cls_idx = np.flatnonzero(y == cls)
        if len(cls_idx) == 0:
            raise ValueError(f"Class {cls} is missing from an image.")
        idx.append(rng.choice(cls_idx, size=min(n_per_class, len(cls_idx)), replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def load_training_set(pairs: Sequence[Pair], n_per_class: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for pair in pairs:
        X = load_rgb(pair.rgb)
        y = load_mask(pair.mask)
        if len(X) != len(y):
            raise ValueError(f"Image/mask size mismatch for {pair.date}")
        Xi, yi = sample_balanced(X, y, n_per_class, rng)
        Xs.append(Xi)
        ys.append(yi)
    return np.vstack(Xs), np.concatenate(ys)


def load_test_set(pair: Pair, n_per_class: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    X = load_rgb(pair.rgb)
    y = load_mask(pair.mask)
    if len(X) != len(y):
        raise ValueError(f"Image/mask size mismatch for {pair.date}")
    return sample_balanced(X, y, n_per_class, rng)


def classifiers(seed: int, rf_trees: int):
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=rf_trees,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced_subsample",
        ),
        "Linear SVM": make_pipeline(
            StandardScaler(),
            SGDClassifier(loss="hinge", alpha=1e-4, max_iter=2000, tol=1e-4, random_state=seed, class_weight="balanced"),
        ),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray):
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) else 0.0
    return acc, float(precision), float(recall), float(f1), iou, int(tn), int(fp), int(fn), int(tp)


def run_one(experiment: str, train_pairs: Sequence[Pair], test_pair: Pair, train_source: str, test_source: str, args: argparse.Namespace, rng: np.random.Generator, note: str) -> List[ResultRow]:
    X_train, y_train = load_training_set(train_pairs, args.n_train_per_class_per_image, rng)
    X_test, y_test = load_test_set(test_pair, args.n_test_per_class, rng)
    rows: List[ResultRow] = []
    for clf_name, clf in classifiers(args.seed, args.rf_trees).items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc, pr, re, f1, iou, tn, fp, fn, tp = evaluate(y_test, pred)
        rows.append(ResultRow(
            experiment=experiment,
            classifier=clf_name,
            train_source=train_source,
            test_source=test_source,
            train_dates=",".join(p.date for p in train_pairs),
            test_date=test_pair.date,
            feature_source="archived RGB PNG proxy",
            n_train_pixels=int(len(y_train)),
            n_test_pixels=int(len(y_test)),
            accuracy=acc,
            precision_quarry=pr,
            recall_quarry=re,
            f1_quarry=f1,
            iou_quarry=iou,
            tn=tn,
            fp=fp,
            fn=fn,
            tp=tp,
            note=note,
        ))
    return rows


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data folder not found: {data_dir}")

    manual_dates = split_csv(args.manual_dates)
    exclude_dates = split_csv(args.exclude_dates)
    months = None if args.months.strip() == "" else {int(x) for x in split_csv(args.months)}
    pairs = discover_pairs(data_dir, manual_dates, exclude_dates, months)
    manual_pairs = [p for p in pairs if p.kind == "manual"]
    pseudo_pairs = [p for p in pairs if p.kind == "pseudo"]

    found_manual = {p.date for p in manual_pairs}
    missing = [d for d in manual_dates if d not in found_manual]
    if missing:
        raise SystemExit(f"Missing manual endpoint image/mask pairs: {missing}")

    rng = np.random.default_rng(args.seed)
    rows: List[ResultRow] = []

    if not args.skip_manual_cross_year:
        for test_pair in manual_pairs:
            train_pairs = [p for p in manual_pairs if p.date != test_pair.date]
            rows.extend(run_one(
                "manual_cross_year",
                train_pairs,
                test_pair,
                "manual endpoint mask",
                "manual endpoint mask",
                args,
                rng,
                "Non-circular two-date benchmark using manual endpoint masks.",
            ))

    if not args.skip_pseudo_to_manual and pseudo_pairs:
        for test_pair in manual_pairs:
            rows.extend(run_one(
                "pseudo_to_manual",
                pseudo_pairs,
                test_pair,
                "workflow-derived pseudo-label masks",
                "manual endpoint mask",
                args,
                rng,
                "Pseudo-label transfer check; training labels are not independent ground truth.",
            ))

    if not rows:
        raise SystemExit("No benchmark rows produced.")

    df = pd.DataFrame([asdict(r) for r in rows])
    results_path = Path(f"{args.out_prefix}_results.csv")
    summary_path = Path(f"{args.out_prefix}_summary.csv")
    json_path = Path(f"{args.out_prefix}_results.json")

    df.to_csv(results_path, index=False)
    summary = df.groupby(["experiment", "classifier"], as_index=False)[["accuracy", "precision_quarry", "recall_quarry", "f1_quarry", "iou_quarry"]].agg(["mean", "std"])
    summary.columns = ["_".join([c for c in col if c]) for col in summary.columns.to_flat_index()]
    summary.to_csv(summary_path, index=False)
    json_path.write_text(json.dumps({
        "feature_source": "archived RGB PNG proxy",
        "manual_dates": manual_dates,
        "excluded_dates": exclude_dates,
        "n_pairs_all": len(pairs),
        "n_manual_pairs": len(manual_pairs),
        "n_pseudo_pairs": len(pseudo_pairs),
        "results": [asdict(r) for r in rows],
        "interpretation_warning": "manual_cross_year is non-circular; pseudo_to_manual is a pseudo-label transfer check, not independent validation.",
    }, indent=2), encoding="utf-8")

    print(df[["experiment", "classifier", "test_date", "accuracy", "f1_quarry", "iou_quarry"]].to_string(index=False))
    print(f"wrote: {results_path}, {summary_path}, {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
