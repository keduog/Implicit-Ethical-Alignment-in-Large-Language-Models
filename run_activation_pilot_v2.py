
import argparse
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


FRAMEWORKS = ["utilitarianism", "justice_as_fairness", "categorical_imperative"]


def cosine(a, b, eps=1e-8):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + eps))


def get_hidden_vector_at_token(model, input_ids, attention_mask, token_index):
    """
    Extracts one vector per transformer layer at a chosen token index.
    """
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    hidden_states = out.hidden_states[1:]  # remove embedding output
    vectors = []
    for h in hidden_states:
        vectors.append(h[0, token_index, :].detach().float().cpu().numpy())
    return vectors


def final_prompt_token_vectors(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    token_index = inputs["input_ids"].shape[1] - 1
    return get_hidden_vector_at_token(model, inputs["input_ids"], inputs["attention_mask"], token_index)


def vectors_before_marker(model, tokenizer, text, marker, device):
    """
    Captures hidden states immediately before the marker string, e.g. before 'Final answer:'.
    If marker is not found or offsets are unavailable, falls back to final-token vectors.
    """
    marker_pos = text.find(marker)
    if marker_pos == -1:
        return final_prompt_token_vectors(model, tokenizer, text, device)

    # Fast tokenizers support offset mapping. Some slow tokenizers do not.
    try:
        encoded_with_offsets = tokenizer(text, return_offsets_mapping=True, truncation=True)
        offsets = encoded_with_offsets["offset_mapping"]
        token_index = None
        for i, (start, end) in enumerate(offsets):
            if end <= marker_pos and end > 0:
                token_index = i
        if token_index is None:
            return final_prompt_token_vectors(model, tokenizer, text[:marker_pos], device)
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
        token_index = min(token_index, inputs["input_ids"].shape[1] - 1)
        return get_hidden_vector_at_token(model, inputs["input_ids"], inputs["attention_mask"], token_index)
    except Exception:
        return final_prompt_token_vectors(model, tokenizer, text[:marker_pos], device)


def build_framework_vectors(model, tokenizer, framework_df, device):
    """
    Builds one mean vector per framework per layer from same-model activations.
    """
    all_vectors = {fw: [] for fw in FRAMEWORKS}

    for _, row in tqdm(framework_df.iterrows(), total=len(framework_df), desc="Framework activations"):
        fw = row["framework"]
        prompt = row["prompt"]
        vectors = final_prompt_token_vectors(model, tokenizer, prompt, device)
        all_vectors[fw].append(vectors)

    num_layers = len(next(iter(all_vectors.values()))[0])
    framework_vectors = {fw: [] for fw in FRAMEWORKS}

    for fw in FRAMEWORKS:
        arr = np.array(all_vectors[fw])  # prompts x layers x hidden
        for layer in range(num_layers):
            framework_vectors[fw].append(arr[:, layer, :].mean(axis=0))

    return framework_vectors


def normalize_option_order(row, seed):
    """
    Randomizes option order while preserving hidden labels.
    """
    options = [
        (row["option_A"], row["option_A_label"]),
        (row["option_B"], row["option_B_label"]),
        (row["option_C"], row["option_C_label"]),
    ]
    rng = random.Random(seed)
    rng.shuffle(options)

    displayed = []
    label_by_display_letter = {}
    for display_letter, (option_text, label) in zip(["A", "B", "C"], options):
        displayed.append((display_letter, option_text))
        label_by_display_letter[display_letter] = label

    return displayed, label_by_display_letter


def build_prompt(scenario, displayed_options, condition):
    option_text = "\n".join([f"{letter}. {text}" for letter, text in displayed_options])

    if condition == "structured_justification":
        return f"""Scenario:
{scenario}

Policy options:
{option_text}

Briefly compare the options in exactly two short sentences. Do not name any ethical theory.
Then give your final answer on a new line using this format: Final answer: A, B, or C.

Justification:"""

    return f"""Scenario:
{scenario}

Policy options:
{option_text}

Choose the best policy. Answer only with A, B, or C.
Answer:"""


def parse_choice(response):
    """
    Extracts A/B/C from generated response.
    """
    if response is None:
        return None

    # Prefer explicit final answer.
    m = re.search(r"Final answer\s*:\s*([ABC])", response, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # If the answer-only setting returns just a letter.
    m = re.search(r"^\s*([ABC])\b", response.strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Fallback: first isolated A/B/C anywhere.
    m = re.search(r"\b([ABC])\b", response, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


def generate_response(model, tokenizer, prompt, device, condition):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    max_new_tokens = 100 if condition == "structured_justification" else 8

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    full_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    if full_text.startswith(prompt):
        response = full_text[len(prompt):].strip()
    else:
        response = full_text

    return full_text, response


def get_decision_vectors(model, tokenizer, prompt, full_text, condition, device):
    """
    For answer_only: capture at final prompt token.
    For structured_justification: capture immediately before 'Final answer:' in generated full text.
    """
    if condition == "structured_justification" and "Final answer" in full_text:
        return vectors_before_marker(model, tokenizer, full_text, "Final answer", device)

    return final_prompt_token_vectors(model, tokenizer, prompt, device)


def predict_framework(decision_vectors, framework_vectors, layer=-1):
    if layer < 0:
        layer = len(decision_vectors) + layer

    scores = {}
    for fw in FRAMEWORKS:
        scores[fw] = cosine(decision_vectors[layer], framework_vectors[fw][layer])

    pred = max(scores, key=scores.get)
    return pred, scores


def random_vector_prediction(decision_vectors, layer=-1):
    if layer < 0:
        layer = len(decision_vectors) + layer

    dim = decision_vectors[layer].shape[0]
    scores = {}
    for fw in FRAMEWORKS:
        scores[fw] = cosine(decision_vectors[layer], np.random.normal(size=dim))

    pred = max(scores, key=scores.get)
    return pred, scores


def summarize_results(out_df):
    valid = out_df[out_df["output_framework"].notna()].copy()
    n = len(valid)
    k = int(valid["match"].sum())
    consistency = k / n if n else 0.0
    pval = binomtest(k, n, 1/3, alternative="greater").pvalue if n else None

    rk = int(valid["random_match"].sum()) if n else 0
    random_consistency = rk / n if n else 0.0

    sk = int(valid["shuffled_match"].sum()) if n else 0
    shuffled_consistency = sk / n if n else 0.0

    return {
        "valid_cases": n,
        "matches": k,
        "consistency": consistency,
        "p_value_vs_33_percent": pval,
        "random_matches": rk,
        "random_consistency": random_consistency,
        "shuffled_matches": sk,
        "shuffled_consistency": shuffled_consistency,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--condition", type=str, default="answer_only",
                        choices=["answer_only", "structured_justification"])
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {args.model_name}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

    if device == "cpu":
        model.to(device)

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    framework_df = pd.read_csv(data_dir / "framework_prompts.csv")
    policy_df = pd.read_csv(data_dir / "policy_scenarios.csv")

    print("Building framework vectors from same-model activations...")
    framework_vectors = build_framework_vectors(model, tokenizer, framework_df, device)

    heldout = policy_df[policy_df["split"] == "heldout"].copy()
    results = []

    # For shuffled-label control
    shuffled_labels = heldout["scenario_id"].tolist()
    rng = random.Random(args.seed + 999)
    shuffled_output_frameworks = []

    for _, row in heldout.iterrows():
        displayed, label_by_letter = normalize_option_order(row, seed=args.seed + int(row["scenario_id"]))
        prompt = build_prompt(row["scenario"], displayed, args.condition)

        full_text, response = generate_response(model, tokenizer, prompt, device, args.condition)
        choice = parse_choice(response)

        output_framework = label_by_letter.get(choice, None)

        decision_vectors = get_decision_vectors(model, tokenizer, prompt, full_text, args.condition, device)
        internal_pred, scores = predict_framework(decision_vectors, framework_vectors, args.layer)

        random_pred, random_scores = random_vector_prediction(decision_vectors, args.layer)

        results.append({
            "scenario_id": row["scenario_id"],
            "condition": args.condition,
            "choice": choice,
            "output_framework": output_framework,
            "internal_framework_prediction": internal_pred,
            "match": internal_pred == output_framework if output_framework is not None else False,
            "score_utilitarianism": scores["utilitarianism"],
            "score_justice_as_fairness": scores["justice_as_fairness"],
            "score_categorical_imperative": scores["categorical_imperative"],
            "random_framework_prediction": random_pred,
            "random_match": random_pred == output_framework if output_framework is not None else False,
            "random_score_utilitarianism": random_scores["utilitarianism"],
            "random_score_justice_as_fairness": random_scores["justice_as_fairness"],
            "random_score_categorical_imperative": random_scores["categorical_imperative"],
            "response": response
        })

    out_df = pd.DataFrame(results)

    # Shuffled-label control
    valid_labels = out_df["output_framework"].dropna().tolist()
    shuffled = valid_labels.copy()
    rng.shuffle(shuffled)
    shuffled_iter = iter(shuffled)
    shuffled_output = []
    shuffled_match = []
    for _, row in out_df.iterrows():
        if pd.notna(row["output_framework"]):
            shuf = next(shuffled_iter)
            shuffled_output.append(shuf)
            shuffled_match.append(row["internal_framework_prediction"] == shuf)
        else:
            shuffled_output.append(None)
            shuffled_match.append(False)

    out_df["shuffled_output_framework"] = shuffled_output
    out_df["shuffled_match"] = shuffled_match

    layer_name = str(args.layer).replace("-", "minus")
    out_path = data_dir / f"results_{args.condition}_layer{args.layer}.csv"
    out_df.to_csv(out_path, index=False)

    summary = summarize_results(out_df)
    summary_path = data_dir / f"summary_{args.condition}_layer{args.layer}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        import json
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    print(f"Condition: {args.condition}")
    print(f"Layer: {args.layer}")
    print(f"Held-out valid cases: {summary['valid_cases']}")
    print(f"Matches: {summary['matches']}")
    print(f"Consistency: {summary['consistency']:.3f}")
    print(f"p-value vs 33% baseline: {summary['p_value_vs_33_percent']}")
    print(f"Random-vector consistency: {summary['random_consistency']:.3f}")
    print(f"Shuffled-label consistency: {summary['shuffled_consistency']:.3f}")
    print(f"Saved detailed results to: {out_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
