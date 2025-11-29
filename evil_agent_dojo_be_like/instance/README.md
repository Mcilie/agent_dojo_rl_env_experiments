# Training Configuration for openai/gpt-oss-20b

## Hardware Setup
- 2xH200 GPUs
- 1 GPU for training, 1 GPU for inference

## Configuration Files

### train.toml
- **max_steps**: 1000 training steps
- **model.name**: openai/gpt-oss-20b (20B parameter model)
- **optim.lr**: 1e-6 learning rate

### orch.toml
- **max_steps**: 1000
- **seq_len**: 8192 tokens
- **batch_size**: 256 rollouts per batch
- **rollouts_per_example**: 8 rollouts per task
- **environment.id**: evil-agent-dojo-be-like
- **environment.args**:
  - suites: ["all"] - All AgentDojo suites
  - attack_type: "ignore_previous"
  - max_turns: 20
- **sampling**:
  - max_tokens: 2048 (increased from default 512)
  - temperature: 1.0 (high exploration)

### infer.toml
- **model.name**: openai/gpt-oss-20b

## Launch Training

```bash
cd instance
bash launch_training.sh
```

Or manually:
```bash
uv run rl \
  --trainer @ train.toml \
  --orchestrator @ orch.toml \
  --inference @ infer.toml \
  --trainer-gpus 1 \
  --inference-gpus 1 \
  --wandb.project evil-agent-dojo \
  --wandb.name gpt-oss-20b-all-suites-v1
```

## Monitor Training

W&B Dashboard: https://wandb.ai/mcilieg/evil-agent-dojo

## Expected Training Time

With 2xH200 and the config above:
- 1000 steps × 256 rollouts/batch = 256,000 rollouts
- Estimated: 12-24 hours depending on task completion times

## Key Metrics to Watch

- `reward/mean` - Should increase toward 1.0
- `environment/utility_rate` - Task completion rate
- `environment/security_rate` - Prompt injection success rate
