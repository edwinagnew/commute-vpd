"""Functions copied verbatim from Wang et al.'s circuit notebook (venv 2): FinetuneCircuits at fdaeeaac63d5,
2_arithmetic_operations_100/find_circuit_100/100_2arithmetic_opeartions_circuit.ipynb, cells 2 (collate_EAP,
get_logit_positions, kl_divergence) and 8 (calculate_faithfulness). They are defined in the notebook, not in `eap/`.
Only the imports above them are gathered here; the bodies are the notebook's, character for character.
"""
# pyright: reportMissingImports=false
# ruff: noqa
import torch
import torch.nn.functional as F
from eap.evaluate import evaluate_graph, evaluate_baseline


def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    return clean, corrupted, labels


def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    return logits


def kl_divergence(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=True):
    logits = get_logit_positions(logits, input_length)
    clean_logits = get_logit_positions(clean_logits, input_length)

    probs = torch.softmax(logits, dim=-1)
    clean_probs = torch.softmax(clean_logits, dim=-1)

    results = F.kl_div(probs.log(), clean_probs.log(), log_target=True, reduction='none').mean(-1)
    return results.mean() if mean else results


# Function to calculate faithfulness and percentage of model performance
def calculate_faithfulness(model, g, dataloader, metric_fn):
    # Evaluate baseline (full model performance)
    baseline_performance = evaluate_baseline(model, dataloader, metric_fn).mean().item()

    # Evaluate the discovered circuit's performance
    circuit_performance = evaluate_graph(model, g, dataloader, metric_fn).mean().item()

    # Calculate the absolute difference (faithfulness)
    faithfulness = abs(baseline_performance - circuit_performance)

    # Calculate the percentage of model performance achieved by the circuit
    percentage_performance = (1 - faithfulness / baseline_performance) * 100

    print(f"Baseline performance: {baseline_performance}")
    print(f"Circuit performance: {circuit_performance}")
    print(f"Faithfulness: {faithfulness}")
    print(f"Percentage of model performance achieved by the circuit: {percentage_performance:.2f}%")

    return faithfulness, percentage_performance
