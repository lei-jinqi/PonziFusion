#!/usr/bin/env python3
from __future__ import annotations

"""Clean SADPonzi entry for PonziFusion dataset_6166.csv.

Default usage:
    python sadponzi.py

This file keeps the original SADPonzi symbolic-execution detector, but changes the
entry point from "one bytecode file only" to "run on PonziFusion dataset_6166.csv".
It also still supports the original single-contract mode:
    python sadponzi.py -i path/to/runtime_bytecode.bin
"""

import argparse
import csv
import json
import logging
import random
import re
import statistics
import time
from pathlib import Path
from typing import Optional

try:
    import resource
except ImportError:
    resource = None

from teether.exploit import combined_exploit
from teether.project import Project
import teether.ponziSchemes as ponziSchemes

logging.basicConfig(level=logging.ERROR)

# Project layout:
# PonziFusion/
#   dataset/dataset_6166.csv
#   train_codes/RQ1/PA/SADPonzi/sadponzi.py
# Therefore parents[4] points to the PonziFusion project root on both Windows and Linux servers.
DEFAULT_DATASET = Path(__file__).resolve().parents[4] / "dataset" / "dataset_6166.csv"
DEFAULT_OUTPUT_DIR = Path("sadponzi_results")
HEX_RE = re.compile(r"^[0-9a-fA-F]+$")

CONFIG = {
    "target-address": "0x1234",
    "shellcode-address": "0x1000",
    "target_amount": "+1000",
    "savefile": None,
    "initial-storage": None,
    "initial-balance": None,
    "flags": None,
    "looplimit": 10,
}


def set_memory_limit() -> None:
    """Keep original SADPonzi memory limit when the platform supports resource."""
    if resource is None:
        return
    mem_limit = 32 * 1024 * 1024 * 1024
    try:
        rsrc = resource.RLIMIT_VMEM
    except Exception:
        rsrc = resource.RLIMIT_AS
    try:
        resource.setrlimit(rsrc, (mem_limit, mem_limit))
    except Exception:
        pass


def normalize_bytecode(bytecode: str) -> str:
    code = str(bytecode).strip()
    if code.lower().startswith("0x"):
        code = code[2:]
    return code.lower()


def is_valid_bytecode(bytecode: str, min_bytes: int = 10) -> bool:
    if not bytecode:
        return False
    if len(bytecode) % 2 != 0:
        return False
    if not HEX_RE.match(bytecode):
        return False
    return len(bytecode) // 2 >= min_bytes


def detect_bytecode(bytecode: str, source_name: str = "<memory>"):
    """Run SADPonzi on one runtime bytecode string.

    Returns:
        pred: 1 means Ponzi, 0 means Non-Ponzi.
        scheme: detected Ponzi scheme name or non_ponzi.
        report: original-style SADPonzi report text.
    """
    code = bytes.fromhex(normalize_bytecode(bytecode))
    project = Project(code)

    amount = CONFIG["target_amount"].strip()
    amount_check = "+"
    if amount[0] in ("=", "+", "-"):
        amount_check = amount[0]
        amount = amount[1:]
    amount = int(amount)

    initial_storage = {}
    initial_storage_file = CONFIG["initial-storage"]
    if initial_storage_file:
        with open(initial_storage_file, "rb") as handle:
            initial_storage = {int(k, 16): int(v, 16) for k, v in json.load(handle).items()}

    flags = CONFIG["flags"] or {"CALL", "CALLCODE", "DELEGATECALL", "SELFDESTRUCT"}
    result = combined_exploit(
        project,
        int(CONFIG["target-address"], 16),
        int(CONFIG["shellcode-address"], 16),
        amount,
        amount_check,
        initial_storage,
        CONFIG["initial-balance"],
        flags=flags,
        looplimit=CONFIG["looplimit"],
    )

    if result:
        scheme = ponziSchemes.SchemeDict[result[1]]
        report = f"Found: {scheme}\n {source_name} \n Reward Expr:\n{result[0]}"
        return 1, scheme, report

    report = f"Passed: Non-Ponzi\n {source_name}"
    return 0, "non_ponzi", report


def detect_file(input_path: Path, output_path: Optional[Path] = None) -> None:
    bytecode = normalize_bytecode(input_path.read_text(encoding="utf-8", errors="ignore"))
    pred, scheme, report = detect_bytecode(bytecode, str(input_path))
    print(report)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")


def load_dataset(csv_path: Path, min_bytes: int):
    """Load dataset_6166.csv and return valid rows plus skipped rows."""
    rows = []
    skipped = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"address", "bytecode", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")

        for index, row in enumerate(reader):
            address = str(row.get("address", "")).strip().lower()
            bytecode = normalize_bytecode(row.get("bytecode", ""))
            try:
                label = int(str(row.get("label", "")).strip())
            except Exception:
                label = -1

            if label not in (0, 1):
                skipped.append({"index": index, "address": address, "reason": "invalid_label"})
                continue
            if not is_valid_bytecode(bytecode, min_bytes=min_bytes):
                skipped.append({"index": index, "address": address, "reason": "invalid_or_too_short_bytecode"})
                continue

            rows.append({
                "index": index,
                "address": address,
                "bytecode": bytecode,
                "label": label,
            })
    return rows, skipped


def roc_auc_score_binary(y_true, y_score) -> float:
    """Compute ROC-AUC without sklearn, using average ranks for ties."""
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = sum(1 for y in y_true if y == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.0

    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum_pos += avg_rank * sum(1 for _, y in pairs[i:j] if y == 1)
        i = j
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision_score_binary(y_true, y_score) -> float:
    """Compute AUC-PR/AP without sklearn.

    SADPonzi only outputs hard labels, so y_score is usually 0/1. This still
    gives a deterministic AP value for table compatibility, but it is not a
    probability-ranking metric like neural-network AUC-PR.
    """
    n_pos = sum(1 for y in y_true if y == 1)
    if n_pos == 0:
        return 0.0

    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    i = 0
    while i < len(pairs):
        score = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == score:
            if pairs[i][1] == 1:
                tp += 1
            else:
                fp += 1
            i += 1
        recall = tp / n_pos
        precision = tp / (tp + fp) if tp + fp else 0.0
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


def compute_metrics(results):
    tp = sum(1 for row in results if row["label"] == 1 and row["pred"] == 1)
    fp = sum(1 for row in results if row["label"] == 0 and row["pred"] == 1)
    tn = sum(1 for row in results if row["label"] == 0 and row["pred"] == 0)
    fn = sum(1 for row in results if row["label"] == 1 and row["pred"] == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(results) if results else 0.0
    y_true = [int(row["label"]) for row in results]
    y_score = [float(row.get("score", row["pred"])) for row in results]
    auc = roc_auc_score_binary(y_true, y_score) if results else 0.0
    auc_pr = average_precision_score_binary(y_true, y_score) if results else 0.0

    schemes = {}
    statuses = {}
    for row in results:
        schemes[row["scheme"]] = schemes.get(row["scheme"], 0) + 1
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1

    return {
        "samples": len(results),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "pre": precision,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "auc_pr": auc_pr,
        "auc-pr": auc_pr,
        "accuracy": accuracy,
        "scheme_counts": schemes,
        "status_counts": statuses,
    }


def make_stratified_folds(results, n_splits: int, seed: int):
    """Create deterministic stratified folds over already detected samples."""
    rng = random.Random(seed)
    pos_indices = [i for i, row in enumerate(results) if int(row["label"]) == 1]
    neg_indices = [i for i, row in enumerate(results) if int(row["label"]) == 0]
    rng.shuffle(pos_indices)
    rng.shuffle(neg_indices)

    folds = [[] for _ in range(n_splits)]
    for i, idx in enumerate(pos_indices):
        folds[i % n_splits].append(idx)
    for i, idx in enumerate(neg_indices):
        folds[i % n_splits].append(idx)
    for fold in folds:
        fold.sort()
    return folds


def summarize_cross_validation(results, n_splits: int = 10, seed: int = 42):
    folds = make_stratified_folds(results, n_splits=n_splits, seed=seed)
    fold_rows = []
    metric_names = ["pre", "recall", "f1", "auc", "auc_pr"]
    for fold_id, fold_indices in enumerate(folds, start=1):
        fold_results = [results[i] for i in fold_indices]
        metrics = compute_metrics(fold_results)
        fold_rows.append({
            "fold": fold_id,
            "samples": metrics["samples"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "tn": metrics["tn"],
            "fn": metrics["fn"],
            "pre": metrics["pre"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],
            "auc_pr": metrics["auc_pr"],
        })

    summary = {"folds": n_splits, "seed": seed}
    for name in metric_names:
        values = [float(row[name]) for row in fold_rows]
        summary[f"{name}_mean"] = statistics.mean(values) if values else 0.0
        summary[f"{name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return fold_rows, summary


def write_csv(path: Path, rows, fieldnames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_existing_report(report_path: Path):
    """Parse an existing per-contract report so interrupted runs can resume."""
    report = report_path.read_text(encoding="utf-8", errors="ignore")
    first_line = ""
    for line in report.splitlines():
        line = line.strip()
        if line:
            first_line = line
            break

    if first_line.startswith("Found:"):
        scheme = first_line[len("Found:"):].strip() or "ponzi"
        return 1, scheme, report
    if first_line.startswith("Passed: Non-Ponzi"):
        return 0, "non_ponzi", report
    if first_line.startswith("ERROR:"):
        return 0, "error", report
    return 0, "unknown", report


def run_dataset(csv_path: Path, output_dir: Path, limit: int = 0, min_bytes: int = 10, folds: int = 10, seed: int = 42) -> None:
    valid_rows, skipped_rows = load_dataset(csv_path, min_bytes=min_bytes)
    if limit > 0:
        valid_rows = valid_rows[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"[+] dataset: {csv_path}")
    print(f"[+] valid samples: {len(valid_rows)}, skipped samples: {len(skipped_rows)}")

    for pos, row in enumerate(valid_rows, start=1):
        address = row["address"] or f"row_{row['index']}"
        safe_address = address.replace("0x", "")
        report_path = reports_dir / f"{safe_address}.txt"
        start_time = time.time()

        if report_path.exists():
            pred, scheme, report = parse_existing_report(report_path)
            status = "cached"
            error = ""
        else:
            try:
                pred, scheme, report = detect_bytecode(row["bytecode"], address)
                status = "ok"
                error = ""
            except Exception as exc:
                pred = 0
                scheme = "error"
                report = f"ERROR: {type(exc).__name__}: {exc}\n {address}"
                status = "error"
                error = f"{type(exc).__name__}: {exc}"

            report_path.write_text(report, encoding="utf-8", errors="ignore")
        result_row = {
            "index": row["index"],
            "address": address,
            "label": row["label"],
            "pred": pred,
            "score": float(pred),
            "scheme": scheme,
            "status": status,
            "seconds": round(time.time() - start_time, 3),
            "report_file": str(report_path),
            "error": error,
        }
        results.append(result_row)

        if pos % 50 == 0 or pos == len(valid_rows):
            metrics = compute_metrics(results)
            print(
                f"[{pos}/{len(valid_rows)}] "
                f"pre={metrics['pre']:.4f} recall={metrics['recall']:.4f} "
                f"f1={metrics['f1']:.4f} auc={metrics['auc']:.4f} auc-pr={metrics['auc_pr']:.4f}"
            )

    metrics = compute_metrics(results)
    fold_rows, cv_summary = summarize_cross_validation(results, n_splits=folds, seed=seed) if folds > 1 else ([], {})
    write_csv(
        output_dir / "predictions.csv",
        results,
        ["index", "address", "label", "pred", "score", "scheme", "status", "seconds", "report_file", "error"],
    )
    write_csv(output_dir / "skipped.csv", skipped_rows, ["index", "address", "reason"])
    if fold_rows:
        write_csv(
            output_dir / "ten_fold_metrics.csv",
            fold_rows,
            ["fold", "samples", "tp", "fp", "tn", "fn", "pre", "recall", "f1", "auc", "auc_pr"],
        )
    summary = {
        "overall": metrics,
        "ten_fold_cv": cv_summary,
        "note": "SADPonzi is a symbolic/static detector and has no probability output; auc and auc-pr are computed from hard 0/1 predictions as scores.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[+] overall metrics")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if cv_summary:
        print("[+] ten-fold cross-validation summary")
        print(json.dumps(cv_summary, indent=2, ensure_ascii=False))
    print(f"[+] predictions: {output_dir / 'predictions.csv'}")
    print(f"[+] ten-fold metrics: {output_dir / 'ten_fold_metrics.csv'}")
    print(f"[+] summary: {output_dir / 'summary.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SADPonzi for PonziFusion dataset_6166.csv")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="CSV dataset path. Default is PonziFusion dataset_6166.csv.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for reports and metrics.")
    parser.add_argument("--limit", type=int, default=0, help="Only run first N valid samples for smoke testing.")
    parser.add_argument("--min-bytecode-bytes", type=int, default=10, help="Skip bytecode shorter than this number of bytes.")
    parser.add_argument("--folds", type=int, default=10, help="Number of stratified cross-validation folds used for reporting. Default: 10.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified fold assignment.")
    parser.add_argument("-i", "--input", type=Path, default=None, help="Original single bytecode-file mode.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output report path for single bytecode-file mode.")
    return parser.parse_args()


if __name__ == "__main__":
    set_memory_limit()
    args = parse_args()
    if args.input is not None:
        print(f"[+] Processing single bytecode file: {args.input}")
        detect_file(args.input, args.output)
    else:
        run_dataset(args.dataset, args.output_dir, limit=args.limit, min_bytes=args.min_bytecode_bytes, folds=args.folds, seed=args.seed)
