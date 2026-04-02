"""
Interactive chat with a trained model. Run as:
    python -m scripts.chat --depth=4
"""

import argparse
import mlx.core as mx
from nanochat_mlx.gpt import build_model
from nanochat_mlx.engine import Engine
from nanochat_mlx.tokenizer import get_tokenizer
from nanochat_mlx.train import load_checkpoint
from nanochat_mlx.common import get_base_dir
import os

parser = argparse.ArgumentParser(description='Chat with the model')
parser.add_argument('--depth', type=int, default=20, help='Model depth')
parser.add_argument('--aspect-ratio', type=int, default=64, help='Aspect ratio')
parser.add_argument('--head-dim', type=int, default=128, help='Head dimension')
parser.add_argument('--max-seq-len', type=int, default=2048, help='Max sequence length')
parser.add_argument('--source', type=str, default="sft", choices=["base", "sft"], help="Model source")
parser.add_argument('--prompt', type=str, default='', help='Single prompt (non-interactive)')
parser.add_argument('--temperature', type=float, default=0.6, help='Sampling temperature')
parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
parser.add_argument('--top-p', type=float, default=None, help='Nucleus (top-p) sampling')
args = parser.parse_args()

tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()

model = build_model(
    depth=args.depth,
    aspect_ratio=args.aspect_ratio,
    head_dim=args.head_dim,
    max_seq_len=args.max_seq_len,
    vocab_size=vocab_size,
)

# Load checkpoint
base_dir = get_base_dir()
if args.source == "sft":
    checkpoint_dir = os.path.join(base_dir, "sft_checkpoints", f"d{args.depth}")
else:
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", f"d{args.depth}")

if os.path.exists(checkpoint_dir):
    steps = []
    for f in os.listdir(checkpoint_dir):
        if f.endswith(".safetensors"):
            step = int(f.split("_")[1].split(".")[0])
            steps.append(step)
    if steps:
        latest_step = max(steps)
        print(f"Loading checkpoint from step {latest_step}")
        load_checkpoint(checkpoint_dir, latest_step, model)
else:
    print(f"No checkpoint found at {checkpoint_dir}, using random weights")

# Special tokens
bos = tokenizer.get_bos_token_id()
user_start = tokenizer.encode_special("<|user_start|>")
user_end = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")
assistant_end = tokenizer.encode_special("<|assistant_end|>")

engine = Engine(model, tokenizer)

print("\nNanoChat MLX Interactive Mode")
print("-" * 50)
print("Type 'quit' or 'exit' to end")
print("Type 'clear' to start new conversation")
print("-" * 50)

conversation_tokens = [bos]

while True:
    if args.prompt:
        user_input = args.prompt
    else:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    if user_input.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break
    if user_input.lower() == 'clear':
        conversation_tokens = [bos]
        print("Conversation cleared.")
        continue
    if not user_input:
        continue

    # Add user message
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(user_input))
    conversation_tokens.append(user_end)
    conversation_tokens.append(assistant_start)

    # Generate response
    response_tokens = []
    print("\nAssistant: ", end="", flush=True)
    for token_column, token_masks in engine.generate(
        conversation_tokens, num_samples=1,
        max_tokens=256, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p
    ):
        token = token_column[0]
        response_tokens.append(token)
        print(tokenizer.decode([token]), end="", flush=True)
    print()

    if not response_tokens or response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)

    if args.prompt:
        break
