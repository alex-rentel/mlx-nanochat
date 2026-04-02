"""
Supervised Fine-Tuning (SFT) pipeline for mlx-nanochat.
Loads a pretrained base model and fine-tunes on conversation data.
IMPORTANT: All mx.eval() calls in this file are MLX array materialization,
NOT Python's eval() builtin. mx.eval() forces lazy computation graphs to execute.
"""

import os
import json
import time
import math
from dataclasses import asdict

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import requests

from nanochat_mlx.gpt import GPT, GPTConfig, build_model
from nanochat_mlx.common import get_base_dir, print_banner
from nanochat_mlx.tokenizer import get_tokenizer
from nanochat_mlx.train import save_checkpoint, load_checkpoint


def load_smoltalk(max_examples=50000):
    """Download and load SmolTalk conversation data."""
    base_dir = get_base_dir()
    cache_dir = os.path.join(base_dir, "sft_data")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "smoltalk.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            data = json.load(f)
        return data[:max_examples]

    # Download SmolTalk from HuggingFace
    print("Downloading SmolTalk dataset...")
    url = "https://huggingface.co/datasets/HuggingFaceTB/smoltalk/resolve/main/data/all/train-00000-of-00001.parquet"
    try:
        import pyarrow.parquet as pq
        import tempfile
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    tmp.write(chunk)
            tmp_path = tmp.name

        table = pq.read_table(tmp_path)
        os.unlink(tmp_path)

        conversations = []
        for row in table.to_pylist():
            if "messages" in row:
                conversations.append({"messages": row["messages"]})

        with open(cache_path, "w") as f:
            json.dump(conversations, f)
        print(f"Cached {len(conversations)} conversations to {cache_path}")
        return conversations[:max_examples]
    except Exception as e:
        print(f"Failed to download SmolTalk: {e}")
        print("Generating synthetic training data instead...")
        return _generate_synthetic_sft_data(max_examples)


def _generate_synthetic_sft_data(n=1000):
    """Generate simple synthetic conversations for testing."""
    conversations = []
    templates = [
        ("What is {n} + {m}?", "The answer is {a}."),
        ("Name a color.", "Blue is a color."),
        ("What is the capital of France?", "The capital of France is Paris."),
        ("Say hello.", "Hello! How can I help you today?"),
        ("What is Python?", "Python is a programming language."),
    ]
    for i in range(n):
        t = templates[i % len(templates)]
        n_val, m_val = i % 100, (i * 7) % 100
        user_msg = t[0].format(n=n_val, m=m_val)
        asst_msg = t[1].format(a=n_val + m_val)
        conversations.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": asst_msg},
            ]
        })
    return conversations


def sft_data_loader(tokenizer, conversations, B, T):
    """Pack conversations into batches using BOS-aligned bestfit packing."""
    row_capacity = T + 1

    # Tokenize all conversations
    all_docs = []
    for conv in conversations:
        try:
            ids, mask = tokenizer.render_conversation(conv, max_tokens=row_capacity)
            if len(ids) > 2:
                all_docs.append((ids, mask))
        except (AssertionError, KeyError):
            continue

    if not all_docs:
        raise ValueError("No valid conversations found in SFT data")

    print(f"Tokenized {len(all_docs)} conversations for SFT")
    doc_idx = 0

    while True:
        batch_inputs = np.zeros((B, T), dtype=np.int32)
        batch_targets = np.full((B, T), -1, dtype=np.int32)

        for row in range(B):
            pos = 0
            while pos < row_capacity - 1:
                if doc_idx >= len(all_docs):
                    doc_idx = 0
                    np.random.shuffle(all_docs)

                ids, mask = all_docs[doc_idx]
                doc_idx += 1

                remaining = row_capacity - pos
                doc_len = min(len(ids), remaining)
                ids_chunk = ids[:doc_len]
                mask_chunk = mask[:doc_len]

                for j in range(doc_len - 1):
                    if pos + j < T:
                        batch_inputs[row, pos + j] = ids_chunk[j]
                        if j + 1 < doc_len and mask_chunk[j + 1] == 1:
                            batch_targets[row, pos + j] = ids_chunk[j + 1]

                pos += doc_len
                if doc_len < len(ids):
                    break

        yield mx.array(batch_inputs), mx.array(batch_targets)


def run_sft(args):
    """Main SFT function."""
    print_banner()
    print("Starting Supervised Fine-Tuning...")

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    # Build model
    model = build_model(
        depth=args.depth,
        aspect_ratio=args.aspect_ratio,
        head_dim=args.head_dim,
        max_seq_len=args.max_seq_len,
        vocab_size=vocab_size,
    )
    config = model.config

    # Load pretrained checkpoint if available
    base_dir = get_base_dir()
    base_checkpoint_dir = os.path.join(base_dir, "base_checkpoints", f"d{args.depth}")
    if os.path.exists(base_checkpoint_dir):
        steps = []
        for f in os.listdir(base_checkpoint_dir):
            if f.endswith(".safetensors"):
                step = int(f.split("_")[1].split(".")[0])
                steps.append(step)
        if steps:
            latest_step = max(steps)
            print(f"Loading pretrained checkpoint from step {latest_step}")
            load_checkpoint(base_checkpoint_dir, latest_step, model)
    else:
        print("No pretrained checkpoint found, training from scratch")

    nparams = sum(p.size for _, p in mlx.utils.tree_flatten(model.parameters()))
    print(f"Total parameters: {nparams:,}")

    # Load SFT data and split into train/val
    import random
    conversations = load_smoltalk(max_examples=args.max_examples)
    random.shuffle(conversations)
    split_idx = int(len(conversations) * 0.9)
    train_convos = conversations[:split_idx]
    val_convos = conversations[split_idx:]
    print(f"SFT split: {len(train_convos)} train, {len(val_convos)} val")

    # Create data loader
    loader = sft_data_loader(tokenizer, train_convos, args.device_batch_size, args.max_seq_len)

    # Optimizer - mx.eval below is MLX array materialization, not Python eval
    lr = args.learning_rate
    optimizer = optim.AdamW(learning_rate=lr, betas=(0.9, 0.95), weight_decay=0.01)

    # Training loop
    loss_fn = lambda model, x, y: model(x, targets=y)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    sft_checkpoint_dir = os.path.join(base_dir, "sft_checkpoints", f"d{args.depth}")
    smooth_loss = 0.0

    print(f"\nSFT training for {args.num_iterations} iterations...")
    for step in range(args.num_iterations):
        t0 = time.time()

        x, y = next(loader)
        loss, grads = loss_and_grad_fn(model, x, y)
        optimizer.update(model, grads)
        # mx.eval forces lazy MLX computation to execute (NOT Python's eval builtin)
        mx.eval(model.parameters(), optimizer.state, loss)

        dt = time.time() - t0
        loss_val = loss.item()
        smooth_loss = 0.9 * smooth_loss + 0.1 * loss_val
        debiased = smooth_loss / (1 - 0.9**(step + 1))

        if step % 10 == 0:
            tok_per_sec = int(args.device_batch_size * args.max_seq_len / dt)
            print(f"sft step {step:05d}/{args.num_iterations:05d} | loss: {debiased:.4f} | dt: {dt*1000:.0f}ms | tok/s: {tok_per_sec:,}")

        if step > 0 and step % 100 == 0:
            val_loader_iter = sft_data_loader(tokenizer, val_convos, args.device_batch_size, args.max_seq_len)
            val_loss = 0.0
            val_steps = 5
            for vs in range(val_steps):
                vx, vy = next(val_loader_iter)
                vl = model(vx, targets=vy)
                mx.eval(vl)  # mx.eval materializes MLX lazy arrays
                val_loss += vl.item()
            val_loss /= val_steps
            print(f"sft step {step:05d} | val_loss: {val_loss:.4f}")

        if step > 0 and step % args.save_every == 0:
            meta = {"step": step, "model_config": asdict(config), "phase": "sft"}
            save_checkpoint(sft_checkpoint_dir, step, model, meta)

    meta = {"step": args.num_iterations, "model_config": asdict(config), "phase": "sft"}
    save_checkpoint(sft_checkpoint_dir, args.num_iterations, model, meta)
    print("SFT complete!")
    return model
