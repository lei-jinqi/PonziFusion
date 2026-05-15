import argparse
import csv
import json
import math
import random
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset


SOLIDITY_KEYWORDS = {
    "pragma", "solidity", "contract", "interface", "library", "is", "using", "for",
    "function", "modifier", "constructor", "fallback", "receive", "event", "emit",
    "struct", "enum", "mapping", "address", "bool", "string", "bytes", "byte",
    "int", "uint", "uint8", "uint16", "uint24", "uint32", "uint40", "uint48", "uint56", "uint64",
    "uint72", "uint80", "uint88", "uint96", "uint104", "uint112", "uint120", "uint128", "uint136",
    "uint144", "uint152", "uint160", "uint168", "uint176", "uint184", "uint192", "uint200",
    "uint208", "uint216", "uint224", "uint232", "uint240", "uint248", "uint256",
    "int8", "int16", "int24", "int32", "int40", "int48", "int56", "int64", "int72", "int80",
    "int88", "int96", "int104", "int112", "int120", "int128", "int136", "int144", "int152",
    "int160", "int168", "int176", "int184", "int192", "int200", "int208", "int216", "int224",
    "int232", "int240", "int248", "int256", "bytes1", "bytes2", "bytes3", "bytes4", "bytes5",
    "bytes6", "bytes7", "bytes8", "bytes9", "bytes10", "bytes11", "bytes12", "bytes13", "bytes14",
    "bytes15", "bytes16", "bytes17", "bytes18", "bytes19", "bytes20", "bytes21", "bytes22",
    "bytes23", "bytes24", "bytes25", "bytes26", "bytes27", "bytes28", "bytes29", "bytes30",
    "bytes31", "bytes32", "public", "private", "internal", "external", "payable", "view", "pure",
    "returns", "return", "if", "else", "while", "do", "break", "continue", "throw", "revert",
    "require", "assert", "try", "catch", "unchecked", "assembly", "let", "switch", "case", "default",
    "calldata", "memory", "storage", "constant", "immutable", "virtual", "override", "abstract",
    "indexed", "anonymous", "new", "delete", "true", "false", "this", "super", "msg", "tx", "block",
    "now", "ether", "wei", "gwei", "seconds", "minutes", "hours", "days", "weeks", "years"
}

TOKEN_PATTERN = re.compile(
    r"0x[a-fA-F0-9]+|\d+\.\d+|\d+|[A-Za-z_][A-Za-z0-9_]*|"
    r"==|!=|<=|>=|&&|\|\||=>|\+\+|--|\+=|-=|\*=|/=|%=|<<|>>|"
    r"[{}()\[\];,.:?+\-*/%<>=!&|^~]"
)
STRING_PATTERN = re.compile(r"\"(?:\\.|[^\\\"])*\"|'(?:\\.|[^\\'])*'")
BLOCK_COMMENT_PATTERN = re.compile(r"/\*.*?\*/", re.DOTALL)
LINE_COMMENT_PATTERN = re.compile(r"//.*")


@dataclass
class SBTExample:
    path: str
    label: int
    tokens: List[str]
    parser_backend: str


class Vocabulary:
    def __init__(self, token_to_id: Dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {idx: token for token, idx in token_to_id.items()}
        self.pad_id = token_to_id["<PAD>"]
        self.unk_id = token_to_id["<UNK>"]

    @classmethod
    def build(cls, examples: Sequence[SBTExample], min_freq: int, max_vocab_size: int) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for example in examples:
            counter.update(example.tokens)
        token_to_id = {"<PAD>": 0, "<UNK>": 1}
        for token, freq in counter.most_common(max_vocab_size - len(token_to_id)):
            if freq < min_freq:
                continue
            token_to_id[token] = len(token_to_id)
        return cls(token_to_id)

    def encode(self, tokens: Sequence[str], max_len: int) -> List[int]:
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens[:max_len]]
        if len(ids) < max_len:
            ids.extend([self.pad_id] * (max_len - len(ids)))
        return ids


class SBTDataset(Dataset):
    def __init__(self, examples: Sequence[SBTExample], vocab: Vocabulary, max_len: int):
        self.examples = list(examples)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        input_ids = torch.tensor(self.vocab.encode(example.tokens, self.max_len), dtype=torch.long)
        label = torch.tensor(example.label, dtype=torch.float32)
        return input_ids, label, example.path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding, errors="ignore")
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore")


def strip_comments_and_strings(source: str) -> str:
    source = BLOCK_COMMENT_PATTERN.sub(" ", source)
    source = LINE_COMMENT_PATTERN.sub(" ", source)
    source = STRING_PATTERN.sub(" <STR> ", source)
    return source


def normalize_leaf_token(token: str, node_type: str = "", normalize_identifiers: bool = True) -> str:
    token = token.strip()
    if not token:
        return "<EMPTY>"
    lower = token.lower()
    if node_type in {"string_literal", "unicode_string_literal", "hex_string_literal"} or token == "<STR>":
        return "<STR>"
    if re.fullmatch(r"0x[a-fA-F0-9]+", token):
        return "<HEX>"
    if re.fullmatch(r"\d+(\.\d+)?", token):
        return "<NUM>"
    if normalize_identifiers and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token) and lower not in SOLIDITY_KEYWORDS:
        return "#"
    return lower


def tokenize_solidity(source: str, normalize_identifiers: bool = True) -> List[str]:
    source = strip_comments_and_strings(source)
    return [normalize_leaf_token(token, normalize_identifiers=normalize_identifiers) for token in TOKEN_PATTERN.findall(source)] or ["<EMPTY>"]


def build_fallback_sbt(source: str, normalize_identifiers: bool = True) -> List[str]:
    """A deterministic fallback when an AST parser is unavailable.

    The paper uses Solidity source -> AST -> SBT. The preferred backend below uses tree-sitter
    to produce an actual AST. This fallback keeps the script runnable in bare Python
    environments by making a shallow block/statement tree from lexical tokens; its backend name
    is recorded as lexical_fallback and should not be reported as the strict AST-SBT setting.
    """
    tokens = tokenize_solidity(source, normalize_identifiers=normalize_identifiers)
    sbt: List[str] = ["(SourceUnit"]
    stack: List[str] = ["SourceUnit"]
    in_statement = False
    for token in tokens:
        if token == "{":
            if in_statement:
                sbt.append(")Statement")
                in_statement = False
            sbt.append("(Block")
            stack.append("Block")
        elif token == "}":
            if in_statement:
                sbt.append(")Statement")
                in_statement = False
            if len(stack) > 1:
                node = stack.pop()
                sbt.append(f"){node}")
        elif token == ";":
            if in_statement:
                sbt.append(")Statement")
                in_statement = False
        else:
            if not in_statement:
                sbt.append("(Statement")
                in_statement = True
            sbt.append(token)
    if in_statement:
        sbt.append(")Statement")
    while len(stack) > 1:
        sbt.append(f"){stack.pop()}")
    sbt.append(")SourceUnit")
    return sbt


class SoliditySBTExtractor:
    def __init__(
        self,
        lib_path: Path,
        normalize_identifiers: bool = True,
        allow_fallback: bool = True,
        parser_backend: str = "antlr",
        antlr_script: Optional[Path] = None,
    ):
        self.normalize_identifiers = normalize_identifiers
        self.allow_fallback = allow_fallback
        self.parser = None
        self.backend = "lexical_fallback"
        self.parser_backend = parser_backend
        self.antlr_script = (antlr_script or Path("dl_reproduction/solidity_ast_to_sbt.js")).resolve()

        if parser_backend == "antlr":
            if self._antlr_available():
                self.backend = "solidity_parser_antlr_sbt"
            elif not allow_fallback:
                raise RuntimeError(
                    "solidity-parser-antlr is required for paper-aligned AST-SBT extraction. "
                    "Run npm install solidity-parser-antlr or use --allow_fallback."
                )
        elif parser_backend == "tree_sitter":
            self._setup_tree_sitter(lib_path)
        else:
            if not allow_fallback:
                raise RuntimeError(f"Unknown parser backend: {parser_backend}")

    def _antlr_available(self) -> bool:
        if not self.antlr_script.exists():
            return False
        probe = subprocess.run(
            ["node", "-e", "require('solidity-parser-antlr'); console.log('ok')"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return probe.returncode == 0

    def _setup_tree_sitter(self, lib_path: Path) -> None:
        try:
            from tree_sitter import Language, Parser  # type: ignore
            resolved_lib_path = lib_path.resolve()
            if resolved_lib_path.exists():
                language = Language(str(resolved_lib_path), "solidity")
                parser = Parser()
                parser.set_language(language)
                self.parser = parser
                self.backend = "tree_sitter_ast_sbt"
        except Exception as exc:
            if not self.allow_fallback:
                raise RuntimeError(
                    "tree_sitter Solidity parser is unavailable. Use --parser_backend antlr "
                    "or run with --allow_fallback."
                ) from exc
        if self.parser is None and not self.allow_fallback:
            raise RuntimeError(f"AST parser is unavailable; checked Solidity language library at {lib_path}")

    def source_to_sbt(self, source: str) -> Tuple[List[str], str]:
        if self.backend == "solidity_parser_antlr_sbt":
            try:
                result = subprocess.run(
                    ["node", str(self.antlr_script)],
                    input=source,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.strip())
                raw_tokens = json.loads(result.stdout)
                tokens = [normalize_leaf_token(str(token), normalize_identifiers=self.normalize_identifiers) for token in raw_tokens]
                return tokens or ["<EMPTY>"], self.backend
            except Exception:
                if not self.allow_fallback:
                    raise
                return build_fallback_sbt(source, normalize_identifiers=self.normalize_identifiers), "lexical_fallback_after_antlr_error"

        if self.parser is None:
            return build_fallback_sbt(source, normalize_identifiers=self.normalize_identifiers), self.backend
        try:
            source_bytes = source.encode("utf-8", errors="ignore")
            tree = self.parser.parse(source_bytes)
            tokens: List[str] = []
            self._visit_tree_sitter_node(tree.root_node, source_bytes, tokens)
            return tokens or ["<EMPTY>"], self.backend
        except Exception:
            if not self.allow_fallback:
                raise
            return build_fallback_sbt(source, normalize_identifiers=self.normalize_identifiers), "lexical_fallback_after_parse_error"

    def _node_text(self, node, source_bytes: bytes) -> str:
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore").strip()

    def _visit_tree_sitter_node(self, node, source_bytes: bytes, out: List[str]) -> None:
        named_children = list(getattr(node, "named_children", []))
        node_type = str(getattr(node, "type", "node"))
        if not named_children:
            text = self._node_text(node, source_bytes)
            if text:
                out.append(normalize_leaf_token(text, node_type=node_type, normalize_identifiers=self.normalize_identifiers))
            return
        out.append(f"({node_type}")
        for child in named_children:
            self._visit_tree_sitter_node(child, source_bytes, out)
        out.append(f"){node_type}")


def load_sbt_examples(data_dir: Path, extractor: SoliditySBTExtractor, sample_per_class: int = 0) -> List[SBTExample]:
    ponzi_dir = data_dir / "ponzi"
    non_ponzi_dir = data_dir / "non_ponzi"
    if not ponzi_dir.exists() or not non_ponzi_dir.exists():
        raise FileNotFoundError(f"Expected {ponzi_dir} and {non_ponzi_dir} to exist.")

    examples: List[SBTExample] = []
    for label, folder in [(1, ponzi_dir), (0, non_ponzi_dir)]:
        paths = sorted(folder.glob("*.sol"))
        if sample_per_class > 0:
            paths = paths[:sample_per_class]
        for path in paths:
            source = read_text(path)
            tokens, backend = extractor.source_to_sbt(source)
            examples.append(SBTExample(path=path.as_posix(), label=label, tokens=tokens, parser_backend=backend))
    if not examples:
        raise RuntimeError(f"No .sol files found under {data_dir}")
    return examples


def count_labels(examples: Sequence[SBTExample]) -> Dict[str, int]:
    ponzi = sum(example.label for example in examples)
    non_ponzi = len(examples) - ponzi
    return {"total": len(examples), "ponzi": int(ponzi), "non_ponzi": int(non_ponzi)}


def make_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class MTCformerStrict(nn.Module):
    """Paper-aligned MTCformer: SBT tokens -> embedding -> multi-channel TextCNN -> Transformer -> FC."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        embed_dim: int = 200,
        kernel_sizes: Sequence[int] = (3, 4, 5, 6, 7),
        conv_channels: int = 200,
        nhead: int = 20,
        num_layers: int = 7,
        dim_feedforward: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.kernel_sizes = tuple(kernel_sizes)
        self.conv_channels = conv_channels
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=conv_channels, kernel_size=(kernel_size, embed_dim))
            for kernel_size in self.kernel_sizes
        ])
        self.position = SinusoidalPositionalEncoding(max_len=len(self.kernel_sizes), d_model=conv_channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(len(self.kernel_sizes) * conv_channels, conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(conv_channels, 1),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids).unsqueeze(1)
        channel_features: List[torch.Tensor] = []
        for conv in self.convs:
            feature = F.relu(conv(embedded)).squeeze(3)
            pooled = F.max_pool1d(feature, kernel_size=feature.size(2)).squeeze(2)
            channel_features.append(pooled)
        mtc_matrix = torch.stack(channel_features, dim=1)
        mtc_matrix = self.position(mtc_matrix)
        encoded = self.transformer(mtc_matrix)
        return self.classifier(encoded).squeeze(-1)


class CostSensitiveBCEWithLogitsLoss(nn.Module):
    """MTCformer weighted cross entropy with paper parameter p.

    Weight_ponzi = p * v / (u + v), Weight_non_ponzi = (1 / p) * u / (u + v),
    where u is the number of Ponzi samples and v is the number of non-Ponzi samples in training.
    """

    def __init__(self, train_examples: Sequence[SBTExample], p: float):
        super().__init__()
        u = float(sum(example.label for example in train_examples))
        v = float(len(train_examples) - int(u))
        total = max(u + v, 1.0)
        self.positive_weight_value = p * v / total if total else 1.0
        self.negative_weight_value = (1.0 / p) * u / total if p > 0 else u / total

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        weights = torch.where(
            labels > 0.5,
            torch.as_tensor(self.positive_weight_value, device=labels.device, dtype=labels.dtype),
            torch.as_tensor(self.negative_weight_value, device=labels.device, dtype=labels.dtype),
        )
        return (losses * weights).mean()


def make_loader(examples: Sequence[SBTExample], vocab: Vocabulary, max_len: int, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = SBTDataset(examples, vocab=vocab, max_len=max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for input_ids, labels, _paths in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += float(loss.item()) * input_ids.size(0)
        total_items += input_ids.size(0)
    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    model.eval()
    y_true: List[int] = []
    y_prob: List[float] = []
    paths: List[str] = []
    for input_ids, labels, batch_paths in loader:
        logits = model(input_ids.to(device))
        probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        y_prob.extend(float(prob) for prob in probs)
        y_true.extend(int(v) for v in labels.numpy().tolist())
        paths.extend(list(batch_paths))
    y_pred = [1 if prob >= threshold else 0 for prob in y_prob]
    metrics = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc"] = float("nan")
    try:
        metrics["auc_pr"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["auc_pr"] = float("nan")
    records = [
        {"path": path, "label": label, "prob_ponzi": prob, "pred": pred, "threshold": threshold}
        for path, label, prob, pred in zip(paths, y_true, y_prob, y_pred)
    ]
    return metrics, records


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_metrics_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "repeat", "fold", "epoch", "train_loss", "threshold", "precision", "recall", "f1", "auc", "auc_pr",
        "best_val_f1", "num_train", "num_val", "num_test", "parser_backend",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_metric_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for key in ("precision", "recall", "f1", "auc", "auc_pr"):
        values = np.array([float(row[key]) for row in rows], dtype=np.float64)
        values = values[~np.isnan(values)]
        if len(values) == 0:
            summary[key] = {"mean": float("nan"), "std": float("nan")}
        else:
            summary[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    return summary


def train_eval_fold(
    repeat_index: int,
    fold_index: int,
    train_val_examples: Sequence[SBTExample],
    test_examples: Sequence[SBTExample],
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
) -> Dict[str, object]:
    if args.val_ratio > 0:
        labels = [example.label for example in train_val_examples]
        train_examples, val_examples = train_test_split(
            list(train_val_examples),
            test_size=args.val_ratio,
            random_state=args.seed + repeat_index * 1000 + fold_index,
            stratify=labels,
        )
    else:
        train_examples = list(train_val_examples)
        val_examples = []

    vocab = Vocabulary.build(train_examples, min_freq=args.min_freq, max_vocab_size=args.max_vocab_size)
    train_loader = make_loader(train_examples, vocab, args.max_len, args.batch_size, shuffle=True)
    val_loader = make_loader(val_examples, vocab, args.max_len, args.batch_size, shuffle=False) if val_examples else None
    test_loader = make_loader(test_examples, vocab, args.max_len, args.batch_size, shuffle=False)

    model = MTCformerStrict(
        vocab_size=len(vocab.token_to_id),
        pad_id=vocab.pad_id,
        embed_dim=args.embed_dim,
        kernel_sizes=tuple(args.kernel_sizes),
        conv_channels=args.conv_channels,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    criterion = CostSensitiveBCEWithLogitsLoss(train_examples, p=args.cost_p)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    fold_dir = run_dir / f"repeat_{repeat_index:02d}_fold_{fold_index:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_path = fold_dir / "best_model.pt"
    best_epoch = 0
    best_val_f1 = float("nan")
    best_train_loss = 0.0
    history: List[Dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, args.max_grad_norm)
        if val_loader is not None:
            val_metrics, _ = evaluate(model, val_loader, device, threshold=args.threshold)
            history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{key}": value for key, value in val_metrics.items()}})
            print(
                f"[MTCformer-strict] repeat={repeat_index:02d} fold={fold_index:02d} "
                f"epoch={epoch:03d} loss={train_loss:.4f} val_f1={val_metrics['f1']:.4f} val_recall={val_metrics['recall']:.4f}"
            )
            if math.isnan(best_val_f1) or val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_epoch = epoch
                best_train_loss = train_loss
                torch.save(model.state_dict(), best_path)
        else:
            history.append({"epoch": epoch, "train_loss": train_loss})
            print(
                f"[MTCformer-strict] repeat={repeat_index:02d} fold={fold_index:02d} "
                f"epoch={epoch:03d} loss={train_loss:.4f}"
            )
            best_epoch = epoch
            best_train_loss = train_loss
            torch.save(model.state_dict(), best_path)

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics, test_records = evaluate(model, test_loader, device, threshold=args.threshold)
    split_rows = [
        {"split": "train", "path": example.path, "label": example.label, "num_tokens": len(example.tokens)} for example in train_examples
    ] + [
        {"split": "val", "path": example.path, "label": example.label, "num_tokens": len(example.tokens)} for example in val_examples
    ] + [
        {"split": "test", "path": example.path, "label": example.label, "num_tokens": len(example.tokens)} for example in test_examples
    ]
    save_json(fold_dir / "history.json", history)
    save_jsonl(fold_dir / "splits.jsonl", split_rows)
    save_jsonl(fold_dir / "predictions_test.jsonl", test_records)
    save_json(fold_dir / "vocab.json", vocab.token_to_id)

    backends = sorted({example.parser_backend for example in train_examples + val_examples + list(test_examples)})
    row: Dict[str, object] = {
        "repeat": repeat_index,
        "fold": fold_index,
        "epoch": best_epoch,
        "train_loss": best_train_loss,
        "best_val_f1": best_val_f1,
        "num_train": len(train_examples),
        "num_val": len(val_examples),
        "num_test": len(test_examples),
        "parser_backend": "+".join(backends),
        **test_metrics,
    }
    save_json(fold_dir / "summary.json", row)
    return row


def run_quick_split(examples: Sequence[SBTExample], args: argparse.Namespace, run_dir: Path, device: torch.device) -> Dict[str, object]:
    labels = [example.label for example in examples]
    train_val_examples, test_examples = train_test_split(
        list(examples),
        test_size=args.test_ratio,
        random_state=args.seed,
        stratify=labels,
    )
    row = train_eval_fold(1, 1, train_val_examples, test_examples, args, run_dir, device)
    save_metrics_csv(run_dir / "quick_metrics.csv", [row])
    summary = {
        "paper_alignment": {
            "input": "Solidity source code -> AST -> SBT sequence",
            "model": "embedding -> multi-channel TextCNN -> Transformer -> fully connected classifier",
            "loss": "cost-sensitive weighted BCE, p controls Ponzi/non-Ponzi weights",
            "evaluation": "single stratified train/val/test split for fast F1 preview",
        },
        "metrics": row,
    }
    save_json(run_dir / "summary.json", summary)
    return summary


def run_repeated_cv(examples: Sequence[SBTExample], args: argparse.Namespace, run_dir: Path, device: torch.device) -> Dict[str, object]:
    labels = np.array([example.label for example in examples])
    metric_rows: List[Dict[str, object]] = []
    for repeat_index in range(1, args.repeats + 1):
        splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed + repeat_index)
        for fold_index, (train_val_indices, test_indices) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
            train_val_examples = [examples[index] for index in train_val_indices]
            test_examples = [examples[index] for index in test_indices]
            row = train_eval_fold(repeat_index, fold_index, train_val_examples, test_examples, args, run_dir, device)
            metric_rows.append(row)
            save_metrics_csv(run_dir / "fold_metrics.csv", metric_rows)
            save_json(run_dir / "summary.json", {"metrics": summarize_metric_rows(metric_rows), "folds_finished": len(metric_rows)})
    summary = {
        "paper_alignment": {
            "input": "Solidity source code -> AST -> SBT sequence",
            "model": "embedding -> multi-channel TextCNN -> Transformer -> fully connected classifier",
            "loss": "cost-sensitive weighted BCE, p controls Ponzi/non-Ponzi weights",
            "evaluation": f"{args.n_splits}-fold cross validation; in each round one fold is used as test set and the remaining folds are used as training set",
        },
        "folds_finished": len(metric_rows),
        "metrics": summarize_metric_rows(metric_rows),
    }
    save_metrics_csv(run_dir / "fold_metrics.csv", metric_rows)
    save_json(run_dir / "summary.json", summary)
    return summary


def parse_kernel_sizes(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict paper-based MTCformer reproduction for smart Ponzi detection.")
    parser.add_argument("--data_dir", type=str, default="dataset/RQ1", help="Directory containing ponzi/ and non_ponzi/ Solidity files.")
    parser.add_argument("--output_dir", type=str, default="dl_results_mtcformer_10fold_lite_balanced")
    parser.add_argument("--tree_sitter_lib", type=str, default="my-languages.so", help="Compiled tree-sitter Solidity language library, only used by --parser_backend tree_sitter.")
    parser.add_argument("--parser_backend", choices=["antlr", "tree_sitter", "fallback"], default="antlr", help="AST parser backend. The paper used solidity-parser-antlr, so antlr is the default.")
    parser.add_argument("--antlr_script", type=str, default="dl_reproduction/solidity_ast_to_sbt.js", help="Node.js helper that converts Solidity AST to SBT tokens.")
    parser.add_argument("--allow_fallback", action="store_true", help="Use lexical_fallback if the selected AST parser fails.")
    parser.add_argument("--keep_identifiers", action="store_true", help="Keep user-defined identifiers instead of normalizing them to #.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_mode", choices=["quick", "cv"], default="cv", help="quick runs one train/val/test split for a fast F1 preview; cv runs k-fold cross validation.")
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--val_ratio", type=float, default=0.0, help="Validation ratio inside the training folds. Use 0.0 to match the paper-style setting: nine folds for training and one fold for testing.")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--max_len", type=int, default=192)
    parser.add_argument("--max_vocab_size", type=int, default=30000)
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--kernel_sizes", type=parse_kernel_sizes, default=parse_kernel_sizes("3,4,5,6,7"))
    parser.add_argument("--conv_channels", type=int, default=48)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--cost_p", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for reporting precision/recall/F1. Lower values usually reduce precision and increase recall; AUC/AUC-PR are threshold-independent.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sample_per_class", type=int, default=0, help="Only load the first N contracts per class; intended for parser/dry-run checks, not final experiments.")
    parser.add_argument("--dry_run", action="store_true", help="Only parse Solidity files to SBT and write config; do not start training.")
    args = parser.parse_args()

    sys.setrecursionlimit(10000)
    set_seed(args.seed)
    device = make_device(args.device)
    run_dir = Path(args.output_dir) / f"mtcformer_strict_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    extractor = SoliditySBTExtractor(
        lib_path=Path(args.tree_sitter_lib),
        normalize_identifiers=not args.keep_identifiers,
        allow_fallback=args.allow_fallback,
        parser_backend=args.parser_backend,
        antlr_script=Path(args.antlr_script),
    )
    examples = load_sbt_examples(Path(args.data_dir), extractor, sample_per_class=args.sample_per_class)
    config = vars(args).copy()
    config.update({
        "device": str(device),
        "data_counts": count_labels(examples),
        "parser_backends": sorted({example.parser_backend for example in examples}),
        "run_dir": run_dir.as_posix(),
        "paper_source": "literature_pdfs/sensors-21-06417.pdf",
    })
    save_json(run_dir / "config.json", config)
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print(f"Run directory: {run_dir.as_posix()}")

    if args.dry_run:
        sample_rows = [
            {
                "path": example.path,
                "label": example.label,
                "num_sbt_tokens": len(example.tokens),
                "parser_backend": example.parser_backend,
                "first_tokens": example.tokens[:30],
            }
            for example in examples[: min(len(examples), 10)]
        ]
        save_json(run_dir / "dry_run_samples.json", sample_rows)
        print("Dry run finished: parsed SBT samples only; training was not started.")
        print(json.dumps(sample_rows, ensure_ascii=False, indent=2))
        return

    if args.eval_mode == "quick":
        summary = run_quick_split(examples, args, run_dir, device)
        print("Final quick-split summary:")
    else:
        summary = run_repeated_cv(examples, args, run_dir, device)
        print("Final repeated-CV summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
