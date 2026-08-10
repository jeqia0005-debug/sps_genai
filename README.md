# SPS GenAI API

FastAPI service collecting the models built for the Applied Generative AI
assignments: bigram text generation, word embeddings, a CIFAR-10 classifier, a
GAN, an energy-based model, a diffusion model, and a GPT-2 question-answering
model post-trained with reinforcement learning.

## GPT-2 question answering and RL post-training

`openai-community/gpt2` is fine-tuned on SQuAD and then post-trained with a
policy gradient so that every response follows the format

```
<reasoning> ... </reasoning> <answer> ... </answer>
```

The RL stage sees no labelled example of that format, only the scalar reward
assigned to completions the model generated itself.

### Pipeline

| Stage | Script | Artefact | Description |
| --- | --- | --- | --- |
| `base` | - | downloaded from HuggingFace | pretrained GPT-2 (124M) |
| `sft` | `train_llm.py` | `gpt2_squad_lora.pt` | supervised fine-tuning on SQuAD |
| `rl` | `train_rl.py` | `gpt2_rl_lora.pt` | policy gradient post-training for the response format |

Each stage trains a LoRA adapter rather than the full network, so a stage is
about 3MB on disk instead of the ~500MB of a full GPT-2 checkpoint.  The RL
adapter sits on top of the merged supervised model, so disabling it recovers the
supervised policy.

### Formulation

Generation is treated as a Markov decision process:

- **state** - the prompt plus the tokens generated so far,
- **action** - the next token drawn from the vocabulary,
- **episode** - one completion, ending at the end-of-text token,
- **reward** - the shaped score of the finished completion,
- **policy** - GPT-2 with the trainable LoRA adapter.

Each epoch samples a batch of episodes from the current policy, weights every
action by the return of the episode it belongs to, and takes one gradient step
on

```
loss = -( log pi(a_t | s_t) * R(tau) ).mean()
```

`--kl-coefficient` adds a KL penalty against the supervised policy; it is 0.0 by
default and can be raised if the policy drifts into degenerate text.

### Reward

| Component | Value |
| --- | --- |
| each of the four tags appearing exactly once | +0.25 (max 1.0) |
| the whole completion matching the layout | +1.0 |
| token F1 of the answer section against the SQuAD gold answer | +2.0 x F1 |
| answer longer than 12 words | -0.5 |
| empty reasoning section | -0.5 |
| generation hit the token budget without stopping | -0.5 |

Formatting credit is graded rather than binary, so completions that are close to
the layout score above ones that ignore it.  `train_llm.py` also writes a
fraction of its supervised targets in the structured layout
(`--structured-ratio`, 0.25 by default), which leaves the model sampling that
layout occasionally when the RL stage starts.

## Training

Both scripts download what they need on first run: GPT-2 from HuggingFace and
SQuAD v1.1 into `data/squad/`.

```bash
# Stage 1 - supervised fine-tuning on SQuAD
uv run python train_llm.py --examples 4000 --epochs 1

# Stage 2 - policy gradient post-training for the response format
uv run python train_rl.py --epochs 200
```

`train_rl.py` measures format compliance and answer F1 on held-out SQuAD dev
questions before and after training, and writes the return curve to
`rl_training_log.json`, which the API serves.

Both scripts select `cuda`, `mps` or `cpu` automatically; pass `--device cpu` to
override.  Lower `--examples` and `--epochs` for a shorter run.

## Running the API

```bash
uv run fastapi dev main.py
```

`gpt2_squad_lora.pt`, `gpt2_rl_lora.pt` and `rl_training_log.json` have to exist
before the question-answering endpoints work; the other endpoints do not depend
on them.  Models load on first request, and a missing adapter returns a 503
naming the script that produces it.

### Endpoints

| Method | Path | Description |
| --- | --- | --- |
| POST | `/generate` | bigram text generation |
| POST | `/embedding` | spaCy word embedding |
| POST | `/similarity` | word similarity |
| POST | `/predict` | CIFAR-10 image classification |
| GET | `/generate-digit` | MNIST digit from the GAN |
| GET | `/generate-energy` | CIFAR-10 sample from the energy-based model |
| GET | `/generate-diffusion` | CIFAR-10 sample from the diffusion model |
| POST | `/generate_with_llm` | answer a question with the SQuAD fine-tuned GPT-2 |
| POST | `/generate_with_rl` | answer with the RL post-trained model |
| POST | `/rl/compare` | both stages on the same question, side by side |
| POST | `/rl/reward` | score any completion with the training reward |
| GET | `/rl/format` | the target format and the reward shaping |
| GET | `/rl/training-curve` | return curve and before/after evaluation |

```bash
curl -X POST http://localhost:8000/generate_with_rl \
  -H "Content-Type: application/json" \
  -d '{
        "question": "Who wrote The Count of Monte Cristo?",
        "context": "The Count of Monte Cristo is an adventure novel written by Alexandre Dumas.",
        "gold_answer": "Alexandre Dumas"
      }'
```

```json
{
  "stage": "rl",
  "completion": " <reasoning> The Count of Monte Cristo is an adventure novel written by Alexandre Dumas. </reasoning> <answer> Alexandre Dumas </answer>",
  "reasoning": "The Count of Monte Cristo is an adventure novel written by Alexandre Dumas.",
  "answer": "Alexandre Dumas",
  "format_compliant": true,
  "finished": true,
  "reward": { "total": 4.0, "answer_f1": 1.0, "format_compliant": true }
}
```

## Docker

The image bakes the pretrained GPT-2 weights in at build time, so the container
needs no network at runtime.  The build copies the output of the two training
scripts, which therefore have to run first.

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

## Tests

```bash
uv run python tests/test_rl_pipeline.py   # or: uv run pytest tests/
```

Covers the reward shaping, the LoRA adapter (no-op at initialisation, disabling,
merging, checkpoint round-trip) and the policy gradient loss.

## Layout

```
app/
  squad_data.py      SQuAD loading and the two response layouts
  llm_model.py       GPT-2 loading, prompting, sampling, log probabilities
  lora.py            low-rank adapters: inject, disable, merge, save/load
  rl_reward.py       response parsing and the shaped reward
  rl_trainer.py      policy gradient trainer and evaluation
  bigram_model.py    embedding_model.py  cnn_model.py
  gan_model.py       energy_model.py     diffusion_model.py  image_utils.py
train_llm.py         stage 1 - supervised fine-tuning on SQuAD
train_rl.py          stage 2 - policy gradient post-training
main.py              FastAPI application
tests/               tests for the RL components
```

## Local dependencies

`pyproject.toml` pins the FastAPI and spaCy stack.  PyTorch and transformers are
installed separately, the same way the Dockerfile does it:

```bash
uv sync
uv pip install torch torchvision transformers pillow
```
