#!/bin/bash
# Launch RL training for openai/gpt-oss-20b on Evil Agent Dojo
# 2xH200 setup: 1 GPU for training, 1 GPU for inference

set -e

echo "Starting Evil Agent Dojo training..."
echo "Model: openai/gpt-oss-20b"
echo "Suites: all (workspace, banking, slack, travel)"
echo "Attack type: ignore_previous"
echo ""

# Launch training with GPU allocation
uv run rl \
  --trainer @ train.toml \
  --orchestrator @ orch.toml \
  --inference @ infer.toml \
  --trainer-gpus 1 \
  --inference-gpus 1 \
  --wandb.project evil-agent-dojo \
  --wandb.name gpt-oss-20b-all-suites-v1

echo ""
echo "Training launched! Monitor at https://wandb.ai/mcilieg/evil-agent-dojo"
