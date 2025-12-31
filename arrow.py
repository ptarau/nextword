#!/usr/bin/env python3
"""
Unified trainer + generator (REPL) for a word-level "token-as-operator" Arrow LM.

Key properties:
- One sentence per line (space-separated words).
- Shared normalization + vocabulary between train/generate.
- Training augments each sentence with ALL contiguous subsequences (sufpref),
  globally deduplicated; Prolog file (.pl) is written ONLY for original sentences.
- Checkpoints ALWAYS save vocabulary (itos/stoi + special token ids) and config.
- Robust resume/load: can resume training or generate from a checkpoint.
  If an older checkpoint lacks vocab, you can pass --data to rebuild it deterministically.

NEW (this version):
- REPL supports GNU-readline-like line editing and persistent history (history.txt)
  when Python's `readline` module is available (typical on Linux/macOS).

Usage:
  Train:
    python arrow.py train --data file.txt --out_dir ckpts --seq_len 64
  Resume:
    python arrow.py train --data file.txt --out_dir ckpts --resume ckpts/ckpt_latest.pt
  Generate (REPL):
    python arrow.py generate --ckpt ckpts/ckpt_latest.pt --repl --history_file history.txt
  Generate (prompt):
    python arrow.py generate --ckpt ckpts/ckpt_latest.pt --prompt "the operator"
"""

from __future__ import annotations
import argparse
import os
import re
import math
import time
import atexit
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Text normalization + Prolog
# -------------------------

_PUNCT_RE = re.compile(r"[,\.;:\?!\"()\[\]\{\}]")


def normalize_line_to_words(line: str) -> List[str]:
    """
    Normalize to match training & generation:
    - strip
    - remove punctuation like , ; . ? ! : " ( ) [ ] { }
    - lowercase
    - split on whitespace
    """
    line = line.strip()
    if not line:
        return []
    line = _PUNCT_RE.sub(" ", line)
    line = re.sub(r"\s+", " ", line).strip()
    if not line:
        return []
    return line.lower().split(" ")


def prolog_quote_atom(w: str) -> str:
    """
    Quote when needed (e.g., contains uppercase, starts with non-lowercase, has non-alnum/_).
    After normalization we lowercase, but we still keep quoting rules robust.
    Also escape single quotes as doubled quotes for Prolog.
    """
    if w == "":
        return "''"
    # Prolog atom allowed unquoted: starts with lowercase letter, then lowercase/digit/_ only
    if re.fullmatch(r"[a-z][a-z0-9_]*", w):
        return w
    w2 = w.replace("'", "''")
    return f"'{w2}'"


def write_prolog_file(txt_path: Path, original_sents: List[List[str]]) -> Path:
    """
    Writes <file.pl> next to <file.txt> with only ORIGINAL sentences:
      sent([the,cat,sits,on,the,mat]).
    """
    pl_path = txt_path.with_suffix(".pl")
    with pl_path.open("w", encoding="utf-8") as f:
        for ws in original_sents:
            if not ws:
                continue
            atoms = ", ".join(prolog_quote_atom(w) for w in ws)
            f.write(f"sent([{atoms}]).\n")
    return pl_path


# -------------------------
# Augmentation: all contiguous subsequences + global dedup
# -------------------------


def sufpref(xs: List[str]) -> Iterable[List[str]]:
    n = len(xs)
    for i in range(n):
        for j in range(i + 1, n + 1):
            yield xs[i:j]


def build_augmented_sentences(original_sents: List[List[str]]) -> List[List[str]]:
    """
    Add all contiguous subsequences for each sentence and globally deduplicate.
    Keeps original sentences included. Returns list of unique sequences (lists of words).
    """
    seen = set()
    out: List[List[str]] = []
    for ws in original_sents:
        if not ws:
            continue
        for seg in sufpref(ws):
            t = tuple(seg)
            if t in seen:
                continue
            seen.add(t)
            out.append(seg)
    return out


# -------------------------
# Vocabulary
# -------------------------


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_id: int
    eos_id: int
    unk_id: int

    def encode_words(self, words: List[str]) -> List[int]:
        return [self.stoi.get(w, self.unk_id) for w in words]

    def decode_ids(self, ids: List[int]) -> List[str]:
        out = []
        for i in ids:
            if 0 <= i < len(self.itos):
                out.append(self.itos[i])
            else:
                out.append("<badid>")
        return out

    def decode_text(self, ids: List[int], stop_at_eos: bool = True) -> str:
        words = []
        for i in ids:
            if stop_at_eos and i == self.eos_id:
                break
            if i == self.pad_id:
                continue
            words.append(self.itos[i] if 0 <= i < len(self.itos) else "<badid>")
        return " ".join(words).strip()


def build_vocab_from_sentences(sents: List[List[str]]) -> Vocab:
    """
    Builds vocab deterministically:
    special tokens: <pad>, <eos>, <unk>
    then words sorted lexicographically for deterministic ids.
    """
    specials = ["<pad>", "<eos>", "<unk>"]
    wordset = set()
    for ws in sents:
        for w in ws:
            if w:
                wordset.add(w)
    words = sorted(wordset)
    itos = specials + words
    stoi = {w: i for i, w in enumerate(itos)}
    return Vocab(
        stoi=stoi,
        itos=itos,
        pad_id=stoi["<pad>"],
        eos_id=stoi["<eos>"],
        unk_id=stoi["<unk>"],
    )


def vocab_to_state(v: Vocab) -> dict:
    return {
        "itos": v.itos,
        "stoi": v.stoi,
        "pad_id": v.pad_id,
        "eos_id": v.eos_id,
        "unk_id": v.unk_id,
    }


def vocab_from_state(d: dict) -> Vocab:
    itos = d["itos"]
    stoi = d.get("stoi") or {w: i for i, w in enumerate(itos)}
    pad_id = int(d.get("pad_id", stoi.get("<pad>", 0)))
    eos_id = int(d.get("eos_id", stoi.get("<eos>", 1)))
    unk_id = int(d.get("unk_id", stoi.get("<unk>", 2)))
    return Vocab(stoi=stoi, itos=itos, pad_id=pad_id, eos_id=eos_id, unk_id=unk_id)


# -------------------------
# Model (match the working trainer/generator keys: h0, s, U, V, out)
# -------------------------


class ArrowTokenLM(nn.Module):
    """
    Simple recurrent "token-as-operator" model:
      h_{t+1} = tanh( s(emb(x_t)) + U h_t )
      logits = out(h_{t+1})
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.h0 = nn.Parameter(torch.zeros(d_model))
        self.s = nn.Embedding(vocab_size, d_model)
        self.U = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(
            d_model, d_model, bias=False
        )  # kept for compatibility / extension
        self.out = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T] token ids
        returns logits: [B, T, V] predicting next token at each position (teacher-forcing)
        """
        B, T = x.shape
        h = self.h0.unsqueeze(0).expand(B, -1)  # [B, D]
        logits = []
        for t in range(T):
            xt = x[:, t]  # [B]
            e = self.s(xt)  # [B, D]
            h = torch.tanh(e + self.U(h))
            logits.append(self.out(h))  # [B, V]
        return torch.stack(logits, dim=1)


# -------------------------
# Dataset helpers
# -------------------------


def make_batch(
    encoded_sents: List[List[int]],
    pad_id: int,
    eos_id: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample a batch (uniform) and build:
      x: [B, T]  input tokens
      y: [B, T]  target next tokens
      m: [B, T]  mask (1 for valid target positions, 0 for padding)
    We append eos and pad/trunc to seq_len.
    """
    import random

    B = batch_size
    T = seq_len
    x = torch.full((B, T), pad_id, dtype=torch.long)
    y = torch.full((B, T), pad_id, dtype=torch.long)
    m = torch.zeros((B, T), dtype=torch.float32)
    for i in range(B):
        sent = random.choice(encoded_sents)
        # append eos
        ids = sent + [eos_id]
        if len(ids) < 2:
            ids = [eos_id, eos_id]
        # truncate to T+1 so we can make x/y of length T
        ids = ids[: T + 1]
        # build x/y
        xi = ids[:-1]
        yi = ids[1:]
        # pad if needed
        if len(xi) < T:
            xi = xi + [pad_id] * (T - len(xi))
            yi = yi + [pad_id] * (T - len(yi))
        x[i] = torch.tensor(xi, dtype=torch.long)
        y[i] = torch.tensor(yi, dtype=torch.long)
        # mask targets where yi is not pad
        mi = [1.0 if t != pad_id else 0.0 for t in yi]
        m[i] = torch.tensor(mi, dtype=torch.float32)
    return x.to(device), y.to(device), m.to(device)


def masked_ce_loss(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    logits: [B, T, V], targets: [B, T], mask: [B, T]
    """
    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.reshape(B * T, V), targets.reshape(B * T), reduction="none"
    )
    loss = loss.reshape(B, T)
    loss = (loss * mask).sum() / (mask.sum().clamp_min(1.0))
    return loss


# -------------------------
# Checkpointing
# -------------------------


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    step: int,
    config: dict,
    vocab: Optional[Vocab],
    sentences: Optional[List[List[str]]] = None,
) -> None:
    ckpt = {
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict(),
        "step": step,
        "config": config,
    }
    if vocab is not None:
        ckpt["vocab"] = vocab_to_state(vocab)
    if sentences is not None:
        ckpt["sentences"] = sentences
    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)


def load_checkpoint(
    ckpt_path: Path,
    device: torch.device,
    data_path_for_vocab: Optional[Path] = None,
) -> Tuple[ArrowTokenLM, Vocab, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt.get("config", {})
    d_model = int(config.get("d_model", 256))

    vocab_state = ckpt.get("vocab", None)
    if vocab_state is None:
        # allow rebuild from --data
        if data_path_for_vocab is None:
            raise RuntimeError(
                "Checkpoint missing vocab. Provide --data to rebuild vocab deterministically."
            )
        original = read_original_sentences(data_path_for_vocab)
        # IMPORTANT: vocab should be built from ORIGINAL (not augmented) so ids stay stable with text.
        vocab = build_vocab_from_sentences(original)
    else:
        vocab = vocab_from_state(vocab_state)

    model = ArrowTokenLM(vocab_size=len(vocab.itos), d_model=d_model)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, vocab, ckpt


# -------------------------
# Data reading
# -------------------------


def read_original_sentences(txt_path: Path) -> List[List[str]]:
    original: List[List[str]] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            ws = normalize_line_to_words(line)
            if ws:
                original.append(ws)
    return original


# -------------------------
# Generation (beam search)
# -------------------------


@torch.no_grad()
def step_logits(
    model: ArrowTokenLM, prefix_ids: List[int], device: torch.device, seq_len: int
) -> torch.Tensor:
    """
    Compute next-token logits given prefix (teacher-forcing style).
    We feed prefix into the recurrent model and return logits at last position.
    """
    # truncate to seq_len
    prefix_ids = (prefix_ids or [])[-seq_len:]
    x = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    logits = model(x)  # [1, T, V]
    return logits[0, -1]  # [V]


@torch.no_grad()
def beam_complete(
    model: ArrowTokenLM,
    vocab: Vocab,
    prompt_words: List[str],
    device: torch.device,
    seq_len: int,
    beam_size: int,
    max_new_tokens: int,
    temperature: float,
) -> List[Tuple[List[int], float]]:
    """
    Returns list of (token_ids INCLUDING prompt, score) sorted best-first.
    score is sum logprobs (higher is better).
    """
    prompt_ids = vocab.encode_words(prompt_words)
    beams: List[Tuple[List[int], float]] = [(prompt_ids[:], 0.0)]
    finished: List[Tuple[List[int], float]] = []

    for _ in range(max_new_tokens):
        new_beams: List[Tuple[List[int], float]] = []
        for ids, score in beams:
            if ids and ids[-1] == vocab.eos_id:
                finished.append((ids, score))
                continue

            logits = step_logits(model, ids, device, seq_len)
            if temperature and temperature != 1.0:
                logits = logits / float(temperature)
            logp = F.log_softmax(logits, dim=-1)

            topk = torch.topk(logp, k=min(beam_size, logp.numel()))
            for lp, tok in zip(topk.values.tolist(), topk.indices.tolist()):
                new_ids = ids + [int(tok)]
                new_beams.append((new_ids, score + float(lp)))

        if not new_beams:
            break
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

        if all(b and b[-1] == vocab.eos_id for b, _ in beams):
            finished.extend(beams)
            break

    finished.extend(beams)
    seen = set()
    uniq: List[Tuple[List[int], float]] = []
    for ids, sc in sorted(finished, key=lambda x: x[1], reverse=True):
        t = tuple(ids)
        if t in seen:
            continue
        seen.add(t)
        uniq.append((ids, sc))
    return uniq


# -------------------------
# Retrieval: longest suffix completion from memorized sentences
# -------------------------


def _find_occurrences(haystack: List[str], needle: List[str]) -> List[int]:
    """Return all start indices where needle occurs contiguously in haystack."""
    if not needle or not haystack or len(needle) > len(haystack):
        return []
    starts: List[int] = []
    m = len(needle)
    for i in range(0, len(haystack) - m + 1):
        if haystack[i : i + m] == needle:
            starts.append(i)
    return starts


@torch.no_grad()
def score_suffix_continuation(
    model: ArrowTokenLM,
    vocab: Vocab,
    suffix_words: List[str],
    qlen: int,
    device: torch.device,
    seq_len: int,
) -> float:
    """
    Score log P(continuation | query) for a suffix that begins with the query words.

    suffix_words: full suffix starting at the match position (includes query words).
    qlen: number of words in the query (prefix inside suffix_words).
    We score tokens AFTER the query, including EOS.
    """
    if not suffix_words:
        return -1e9
    ids = vocab.encode_words(suffix_words) + [vocab.eos_id]
    qlen = max(1, int(qlen))

    if len(ids) > seq_len:
        chop = len(ids) - seq_len
        ids = ids[chop:]
        qlen = max(1, qlen - chop)
        if qlen >= len(ids):
            qlen = max(1, len(ids) - 1)

    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    logits = model(x)
    logp = F.log_softmax(logits, dim=-1)

    start = max(0, qlen - 1)
    lp = logp[0, start:, :].gather(1, y[0, start:].unsqueeze(1)).squeeze(1)
    return float(lp.sum().item())


@torch.no_grad()
def retrieve_best_suffixes(
    model: ArrowTokenLM,
    vocab: Vocab,
    sentences: List[List[str]],
    query_words: List[str],
    device: torch.device,
    seq_len: int,
    top_k: int = 4,
) -> List[Tuple[List[str], float, int, int, int]]:
    """
    Return top_k suffixes that start at an occurrence of query_words inside any memorized sentence,
    ranked by continuation score (higher better).

    Returns:
      (suffix_words, score, line_no_1based, start_idx_0based, end_idx_0based)
    """
    if not query_words:
        return []
    cands: List[Tuple[List[str], float, int, int, int]] = []

    qlen = len(query_words)
    for line_idx, s in enumerate(sentences):
        for start in _find_occurrences(s, query_words):
            suffix = s[start:]
            sc = score_suffix_continuation(model, vocab, suffix, qlen, device, seq_len)
            end = start + qlen - 1
            cands.append((suffix, sc, line_idx + 1, start, end))

    cands.sort(key=lambda t: (t[1], len(t[0])), reverse=True)

    out: List[Tuple[List[str], float, int, int, int]] = []
    # Deduplicate by the actual suffix text, not by its (line,start,end) location.
    # This ensures we only return multiple answers if they are not duplicates.
    seen = set()
    for suf, sc, line_no, start, end in cands:
        key = tuple(suf)
        if key in seen:
            continue
        seen.add(key)
        out.append((suf, sc, line_no, start, end))
        if len(out) >= top_k:
            break
    return out


# -------------------------
# Readline history (GNU-like editing + persistent history)
# -------------------------


def setup_history(history_file: str, max_len: int = 2000):
    """
    Enable readline editing + persistent history if `readline` is available.
    Loads history at startup and writes it on exit.

    Returns readline module if available, else None.
    """
    try:
        import readline  # type: ignore
    except Exception:
        print("Note: readline not available; REPL will not have line editing/history.")
        return None

    hist_path = Path(history_file).expanduser()
    hist_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        readline.set_history_length(int(max_len))
    except Exception:
        pass

    if hist_path.exists():
        try:
            readline.read_history_file(str(hist_path))
        except Exception:
            pass

    def _save():
        try:
            readline.write_history_file(str(hist_path))
        except Exception:
            pass

    atexit.register(_save)
    return readline


# -------------------------
# REPL
# -------------------------


def repl_generate(
    model: ArrowTokenLM,
    vocab: Vocab,
    device: torch.device,
    seq_len: int,
    beam_size: int,
    max_new_tokens: int,
    temperature: float,
    num_return: int = 4,
    sentences: Optional[List[List[str]]] = None,
    use_retrieval: bool = True,
    show_meta: bool = True,
    history_file: str = "history.txt",
    history_len: int = 2000,
) -> None:
    # Enable GNU-like editing/history if available
    setup_history(history_file=history_file, max_len=history_len)

    print("REPL: type a cue (words); empty line or Ctrl-D to exit.")
    while True:
        try:
            line = input("> ")
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            print()
            return

        line = line.strip()
        if not line:
            return

        prompt_words = normalize_line_to_words(line)

        # Retrieval-first: return suffixes to end-of-sentence when possible.
        if use_retrieval and sentences is not None and prompt_words:
            hits = retrieve_best_suffixes(
                model, vocab, sentences, prompt_words, device, seq_len, top_k=num_return
            )
            if hits:
                for i, (suf_words, sc, line_no, start, end) in enumerate(hits, 1):
                    if show_meta:
                        print(
                            f"[{i}] (line {line_no}, span {start}-{end}, score {sc:.3f}) "
                            f"{' '.join(suf_words)}"
                        )
                    else:
                        print(f"[{i}] {' '.join(suf_words)}")
                continue

        # Fallback to free beam completion (still useful when cue isn't present).
        beams = beam_complete(
            model,
            vocab,
            prompt_words,
            device,
            seq_len,
            beam_size,
            max_new_tokens,
            temperature,
        )
        for i, (ids, sc) in enumerate(beams[:num_return], 1):
            text = vocab.decode_text(ids, stop_at_eos=True)
            if show_meta:
                print(f"[{i}] (beam score {sc:.3f}) {text}")
            else:
                print(f"[{i}] {text}")


# -------------------------
# Training
# -------------------------


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read originals, write Prolog for originals only
    original = read_original_sentences(data_path)
    if not original:
        raise RuntimeError("No non-empty sentences found in data file.")
    pl_path = write_prolog_file(data_path, original)
    print(f"Wrote Prolog sentences to: {pl_path}")

    # Build augmented unique segments (global dedup)
    augmented = build_augmented_sentences(original)
    print(f"Original sentences: {len(original)}")
    print(f"Augmented unique segments: {len(augmented)}")

    # Build vocab from ORIGINAL sentences (deterministic + stable)
    vocab = build_vocab_from_sentences(original)
    print(
        f"Vocab size: {len(vocab.itos)} (pad={vocab.pad_id}, eos={vocab.eos_id}, unk={vocab.unk_id})"
    )

    # Encode augmented segments
    encoded = [vocab.encode_words(ws) for ws in augmented]
    encoded = [e for e in encoded if len(e) > 0]

    # Create / resume model
    config = {"d_model": args.d_model, "seq_len": args.seq_len}

    model = ArrowTokenLM(vocab_size=len(vocab.itos), d_model=args.d_model).to(device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    start_step = 0
    if args.resume:
        ckpt_path = Path(args.resume)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        if "optim_state" in ckpt and args.resume_optim:
            try:
                optim.load_state_dict(ckpt["optim_state"])
            except Exception as e:
                print(
                    f"Warning: could not load optimizer state ({e}); continuing with fresh optimizer."
                )
        start_step = int(ckpt.get("step", 0))
        if "vocab" in ckpt:
            cv = vocab_from_state(ckpt["vocab"])
            if cv.itos != vocab.itos:
                print(
                    "WARNING: Vocab reconstructed from data differs from vocab in checkpoint."
                )
                print(
                    "         Use the same data file and checkpoint pair for strict consistency."
                )
        print(f"Resumed from {ckpt_path} at step {start_step}")

    model.train()
    t0 = time.time()
    best_loss = float("inf")
    ckpt_latest = out_dir / "ckpt_latest.pt"
    ckpt_best = out_dir / "ckpt_best.pt"

    for step in range(start_step, args.max_steps):
        x, y, m = make_batch(
            encoded, vocab.pad_id, vocab.eos_id, args.seq_len, args.batch_size, device
        )
        logits = model(x)
        loss = masked_ce_loss(logits, y, m)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

        if (step + 1) % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step+1:6d} | loss {loss.item():.4f} | {dt:.1f}s")
            t0 = time.time()

        if (step + 1) % args.save_every == 0:
            save_checkpoint(
                ckpt_latest, model, optim, step + 1, config, vocab, sentences=original
            )
            print(f"Saved: {ckpt_latest}")

        if args.save_best and loss.item() < best_loss:
            best_loss = loss.item()
            save_checkpoint(
                ckpt_best, model, optim, step + 1, config, vocab, sentences=original
            )
            if (step + 1) % args.log_every != 0:
                print(f"New best loss {best_loss:.4f} -> saved {ckpt_best}")

    save_checkpoint(
        ckpt_latest, model, optim, args.max_steps, config, vocab, sentences=original
    )
    print(f"Training done. Saved final: {ckpt_latest}")


# -------------------------
# Generate
# -------------------------


def generate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt)
    data_for_vocab = Path(args.data) if args.data else None
    model, vocab, ckpt = load_checkpoint(
        ckpt_path, device, data_path_for_vocab=data_for_vocab
    )

    sentences = ckpt.get("sentences", None)
    if sentences is None and args.data:
        sentences = read_original_sentences(Path(args.data))

    cfg = ckpt.get("config", {})
    seq_len = int(cfg.get("seq_len", args.seq_len))
    d_model = int(cfg.get("d_model", args.d_model))
    print(f"Loaded ckpt: {ckpt_path}")
    print(f"Model: d_model={d_model}, seq_len={seq_len}, vocab_size={len(vocab.itos)}")

    if args.repl:
        repl_generate(
            model,
            vocab,
            device,
            seq_len,
            args.beam_size,
            args.max_new_tokens,
            args.temperature,
            num_return=args.num_return,
            sentences=sentences,
            use_retrieval=(not args.no_retrieve),
            history_file=args.history_file,
            history_len=args.history_len,
        )
        return

    # Retrieval mode for one-shot prompt (returns suffixes to end of sentence)
    if (not args.no_retrieve) and sentences is not None and args.prompt:
        q = normalize_line_to_words(args.prompt)
        hits = retrieve_best_suffixes(
            model, vocab, sentences, q, device, seq_len, top_k=args.num_return
        )
        if hits:
            for i, (suf_words, sc, line_no, start, end) in enumerate(hits, 1):
                print(
                    f"[{i}] (line {line_no}, span {start}-{end}, score {sc:.3f}) {' '.join(suf_words)}"
                )
            return

    prompt_words = normalize_line_to_words(args.prompt or "")
    beams = beam_complete(
        model,
        vocab,
        prompt_words,
        device,
        seq_len,
        args.beam_size,
        args.max_new_tokens,
        args.temperature,
    )
    for i, (ids, sc) in enumerate(beams[: args.num_return], 1):
        print(f"[{i}] {vocab.decode_text(ids, stop_at_eos=True)}")


# -------------------------
# CLI
# -------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arrow.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser(
        "train", help="train model on sentence lines (+ all contiguous subsequences)"
    )
    pt.add_argument(
        "--data", required=True, help="path to <file.txt> with one sentence per line"
    )
    pt.add_argument("--out_dir", required=True, help="output directory for checkpoints")
    pt.add_argument("--resume", default=None, help="resume from checkpoint")
    pt.add_argument(
        "--resume_optim", action="store_true", help="also resume optimizer state"
    )
    pt.add_argument("--d_model", type=int, default=256)
    pt.add_argument("--seq_len", type=int, default=64)
    pt.add_argument("--batch_size", type=int, default=32)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--weight_decay", type=float, default=0.01)
    pt.add_argument("--grad_clip", type=float, default=1.0)
    pt.add_argument("--max_steps", type=int, default=5000)
    pt.add_argument("--log_every", type=int, default=100)
    pt.add_argument("--save_every", type=int, default=500)
    pt.add_argument("--save_best", action="store_true")
    pt.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    pg = sub.add_parser("generate", help="generate / REPL from checkpoint")
    pg.add_argument("--ckpt", required=True, help="checkpoint path")
    pg.add_argument(
        "--data",
        default=None,
        help="ONLY needed if ckpt lacks vocab; used to rebuild vocab",
    )
    pg.add_argument("--prompt", default="", help="prompt prefix (one-shot mode)")
    pg.add_argument("--repl", action="store_true", help="interactive mode")
    pg.add_argument("--beam_size", type=int, default=8)
    pg.add_argument("--num_return", type=int, default=4)
    pg.add_argument("--max_new_tokens", type=int, default=32)
    pg.add_argument("--temperature", type=float, default=1.0)
    pg.add_argument(
        "--no_retrieve",
        action="store_true",
        help="disable retrieval; use free beam completion only",
    )
    # REPL history / line-editing
    pg.add_argument(
        "--history_file",
        type=str,
        default="history.txt",
        help="REPL history file (read/write) when readline is available.",
    )
    pg.add_argument(
        "--history_len",
        type=int,
        default=2000,
        help="Max number of history entries to keep in memory.",
    )
    # fallback defaults if ckpt lacks config
    pg.add_argument("--d_model", type=int, default=256)
    pg.add_argument("--seq_len", type=int, default=64)
    pg.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "generate":
        generate(args)
    else:
        raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()
