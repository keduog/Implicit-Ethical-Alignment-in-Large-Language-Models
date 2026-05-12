# Practical Step-by-Step Experiment

## Step 1: Define ethical frameworks

Input file:
- `framework_prompts.csv`

What is done:
- Three frameworks are used: utilitarianism, justice_as_fairness, categorical_imperative.
- Each framework has 10 signature prompts.

Output:
- Framework-specific activation examples.

Report:
- Number of frameworks.
- Number of prompts per framework.

---

## Step 2: Construct policy scenarios

Input file:
- `policy_scenarios.csv`

What is done:
- 30 policy scenarios are created.
- Scenarios 1–10 are calibration scenarios.
- Scenarios 11–30 are held-out evaluation scenarios.

Output:
- A scenario set for policy-choice testing.

Report:
- Number of scenarios.
- Domains covered.
- Held-out split.

---

## Step 3: Design policy options

Input file:
- `policy_scenarios.csv`

What is done:
- Each scenario has three options.
- One option is labeled utilitarianism.
- One option is labeled justice_as_fairness.
- One option is labeled categorical_imperative.
- The labels are hidden from the model.
- Option order is randomized during the experiment.

Output:
- Labeled policy-choice dataset.

Report:
- Label balance.
- Randomization method.

---

## Step 4: Run model decision task

Script:
- `run_activation_pilot_v2.py`

Two conditions are used:

### Condition 1: answer_only
The model is asked to answer only A, B, or C.

### Condition 2: structured_justification
The model gives only a brief two-sentence justification and then a final answer.

Important:
- This is not unrestricted chain-of-thought.
- The prompt says "brief justification", not "think step by step".

Output:
- Model choice.
- Model response.
- Decision activation.

Report:
- Choice distribution by framework.
- Invalid response rate.

---

## Step 5: Capture internal activations

What is done:
- The script extracts hidden states using `output_hidden_states=True`.

For answer_only:
- The activation is captured at the last prompt token before generation.

For structured_justification:
- The model generates a brief justification.
- The script captures the activation immediately before the phrase `Final answer`.

Output:
- One decision vector per layer for each scenario.

Report:
- Model name.
- Layer analyzed.
- Token position used for activation capture.

---

## Step 6: Construct framework vectors inside the same model

What is done:
- Each framework prompt is passed through the same model.
- Hidden states are extracted.
- Prompts are averaged to form one vector per framework per layer.

Output:
- `v_utilitarianism`
- `v_justice_as_fairness`
- `v_categorical_imperative`

Report:
- Vector construction method.
- Number of prompts used per vector.

---

## Step 7: Measure representation-level alignment

What is done:
- Cosine similarity is computed between the decision vector and each framework vector.

Output columns:
- `score_utilitarianism`
- `score_justice_as_fairness`
- `score_categorical_imperative`
- `internal_framework_prediction`

Report:
- Average score per framework.
- Highest-score framework per scenario.

---

## Step 8: Analyze decision-framework consistency

What is done:
- The selected option is mapped to its hidden framework label.
- The internal framework prediction is compared with the output framework.

Output:
- `match = True/False`

Report:
- Consistency score.

Formula:
Consistency = matches / valid held-out cases

---

## Step 9: Perform layer-wise analysis

What is done:
- You can rerun the script with different layer values.

Examples:
- `--layer -1` final layer
- `--layer 10` middle/early layer
- `--layer 20` middle layer

Output:
- Results file per layer if you change the output name manually or move files.

Report:
- Which layer gives the strongest consistency.
- Whether alignment appears earlier or later.

---

## Step 10: Add controls

Controls included:
1. Random-vector control.
2. Shuffled-label control.
3. Option-order randomization.
4. Equal-length design attempt.
5. Answer-only baseline.
6. Structured-justification comparison.

Output:
- Main consistency.
- Random-vector consistency.
- Shuffled-label consistency.

Report:
- Whether real consistency is higher than control consistency.

---

## Step 11: Aggregate results

What is done:
- The script prints a summary.
- The script saves a CSV result file.

Output:
- `results_answer_only_layer-1.csv`
- `results_structured_justification_layer-1.csv`

Report:
- Held-out valid cases.
- Matches.
- Consistency.
- Binomial p-value.
- Control consistency.

---

## Step 12: Interpret findings

Correct interpretation:
- The experiment tests whether decision activations are geometrically closer to one framework vector than others.
- It does not prove that the model has human-like ethical reasoning.

Strong claim:
- "The model's policy choices are associated with detectable internal representation patterns."

Avoid claiming:
- "The model truly understands ethics."
- "The model has moral beliefs."
- "The model reasons exactly like a human philosopher."
