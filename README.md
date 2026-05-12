# Implicit Ethical Alignment Pilot

This folder contains a practical pilot experiment for the proposal:

**Implicit Ethical Alignment in Large Language Models: A Representation-Level Analysis of Policy Choice**

## Files

- `framework_prompts.csv`: 30 framework-signature prompts, 10 per framework.
- `policy_scenarios.csv`: 30 policy scenarios. Scenarios 1-10 are for calibration, 11-30 are held-out evaluation.
- `run_activation_pilot.py`: ready-to-run Python script using Hugging Face Transformers.
- `requirements.txt`: Python dependencies.

## Recommended model

Use an instruction-tuned open-weight model that supports hidden-state extraction. Example:

```bash
python run_activation_pilot.py --model_name Qwen/Qwen2.5-1.5B-Instruct
```

If memory is limited, use a smaller instruction model. If you have enough GPU memory, use a stronger 3B or 7B model.

## What the script does

1. Loads framework prompts.
2. Extracts hidden states from the same model.
3. Builds one vector per framework per layer.
4. Loads held-out policy scenarios.
5. Runs the model on each scenario.
6. Captures decision activations.
7. Computes cosine similarity between decision activations and framework vectors.
8. Calculates decision-framework consistency.
9. Runs random-vector and shuffled-label controls.
10. Saves results to CSV.

## Main output

The main result is:

```text
Consistency = cases where predicted internal framework matches output framework / total held-out cases
```

Chance level for three frameworks is approximately 33%.
