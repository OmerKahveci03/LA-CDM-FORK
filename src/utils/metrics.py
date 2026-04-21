import numpy as np

def classification_metrics_flat(
    predictions: list[str | None],
    labels: list[str],
    classes: list[str],
    count_none_as_wrong: bool = True,
) -> dict[str, float | int]:
    """Compute per-class and aggregated classification metrics.

    Args:
        predictions: Predicted class labels (may contain ``None``).
        labels: Ground-truth class labels.
        classes: List of valid class names.
        count_none_as_wrong: If ``True``, ``None`` predictions count as
            incorrect. If ``False``, they are excluded from evaluation.

    Returns:
        Dict with ``none_fraction``, per-class precision/recall/F1/support,
        micro/macro precision/recall/F1, and ``accuracy``.
    """
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must be same length")
    N = len(predictions)

    # Fraction of None predictions
    none_count = sum(1 for p in predictions if p is None)
    none_fraction = none_count / N

    # Build evaluation pairs
    if count_none_as_wrong:
        eval_pairs = list(zip(predictions, labels))
    else:
        eval_pairs = [(p, l) for p, l in zip(predictions, labels) if p is not None]
    M = len(eval_pairs)
    if M == 0:
        # No predictions to evaluate, return all metrics as 0.0
        results = {'none_fraction': none_fraction}
        for c in classes:
            results[f'precision_{c}'] = 0.0
            results[f'recall_{c}'] = 0.0
            results[f'f1_{c}'] = 0.0
            results[f'support_{c}'] = 0
        results['micro_precision'] = 0.0
        results['micro_recall'] = 0.0
        results['micro_f1'] = 0.0
        results['macro_precision'] = 0.0
        results['macro_recall'] = 0.0
        results['macro_f1'] = 0.0
        results['accuracy'] = 0.0
        return results

    results = {'none_fraction': none_fraction}
    total_TP = total_FP = total_FN = 0

    # Per-class counts
    for c in classes:
        TP = sum(1 for p, l in eval_pairs if p == c and l == c)
        FP = sum(1 for p, l in eval_pairs if p == c and l != c)
        FN = sum(1 for p, l in eval_pairs if p != c and l == c)
        support = sum(1 for _, l in eval_pairs if l == c)

        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1   = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0

        results[f'precision_{c}'] = prec
        results[f'recall_{c}']    = rec
        results[f'f1_{c}']        = f1
        results[f'support_{c}']   = support

        total_TP += TP
        total_FP += FP
        total_FN += FN

    # Micro-averaged metrics
    results['micro_precision'] = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    results['micro_recall']    = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    results['micro_f1']        = (2 * total_TP) / (2 * total_TP + total_FP + total_FN) \
                                 if (2 * total_TP + total_FP + total_FN) > 0 else 0.0

    # Macro-averaged metrics
    precisions = [results[f'precision_{c}'] for c in classes]
    recalls    = [results[f'recall_{c}']    for c in classes]
    f1s        = [results[f'f1_{c}']        for c in classes]

    results['macro_precision'] = sum(precisions) / len(classes)
    results['macro_recall']    = sum(recalls)    / len(classes)
    results['macro_f1']        = sum(f1s)        / len(classes)

    # Accuracy
    correct = sum(1 for p, l in eval_pairs if p == l)
    results['accuracy'] = correct / M

    return results


def calculate_classification_metrics(
    predictions: list[str | None],
    labels: list[str],
    ece_triplets: list[tuple],
    classes: list[str] = ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis'],
) -> dict[str, float]:
    """Compute classification metrics and Expected Calibration Error (ECE).

    Combines per-class and aggregate metrics under two settings (counting
    ``None`` predictions as wrong vs. ignoring them), plus hypothesis
    accuracy, unknown fraction, and ECE from confidence calibration triplets.

    Args:
        predictions: Predicted diagnosis labels (may contain ``None``).
        labels: Ground-truth diagnosis labels.
        ece_triplets: List of ``(hypothesis, confidence, label)`` tuples
            for ECE computation.
        classes: Valid class names.

    Returns:
        Flat dict with ``all/*`` and ``ignore/*`` metric variants, ``ece``,
        ``hypothesis_accuracy``, and ``unknown_hypothesis_fraction``.
    """
    ece = 0.0
    # Hypothesis accuracy calculation
    hypothesis_accuracy = 0.0  # accuracy of predictions contained in ece_triplets
    unknown_fraction = 0.0   # fraction of ece_triplet predictions that are 'unknown'
    if len(ece_triplets) != 0:
        # Calculate accuracy of the predictions inside ece_triplets
        correct_hypothesis_preds = sum(1 for pred, _, lbl in ece_triplets if pred == lbl)
        hypothesis_accuracy = correct_hypothesis_preds / len(ece_triplets)

        # Calculate fraction of predictions that are 'unknown'
        unknown_fraction = sum(1 for pred, _, _ in ece_triplets if isinstance(pred, str) and pred.lower() == 'unknown') / len(ece_triplets)

    if len(ece_triplets) != 0:
        # Calculate ECE
        bin_boundaries = np.array([-0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05])
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        # ECE triplet is a list of tuples (hypothesis, confidence, label)
        # Only keep triplets where the confidence is not None and hypothesis is not None and not "Unknown"
        ece_triplets = [t for t in ece_triplets if t[1] is not None and t[0] is not None and t[0] != "unknown"]
        # Only keep triplets where the confidence is a float between 0 and 1
        ece_triplets = [t for t in ece_triplets if t[1] >= 0 and t[1] <= 1]
        if len(ece_triplets) != 0:
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = [(bin_lower <= conf[1] < bin_upper) for conf in ece_triplets]
                prop_in_bin = np.mean(in_bin)
                if prop_in_bin > 0:
                    bin_accuracy = np.mean([ece_triplets[i][0].lower() == ece_triplets[i][2] for i in range(len(ece_triplets)) if in_bin[i]])
                    avg_confidence_in_bin = np.mean([ece_triplets[i][1] for i in range(len(ece_triplets)) if in_bin[i]])
                    ece += np.abs(avg_confidence_in_bin - bin_accuracy) * prop_in_bin


    all_metrics    = classification_metrics_flat(predictions, labels, classes, True)
    ignore_metrics = classification_metrics_flat(predictions, labels, classes, False)

    combined = {'none_fraction': all_metrics['none_fraction']}
    # suffix keys for "all"
    for k, v in all_metrics.items():
        if k != 'none_fraction':
            combined[f'all/{k}'] = v
    # suffix keys for "ignore"
    for k, v in ignore_metrics.items():
        if k != 'none_fraction':
            combined[f'ignore/{k}'] = v

    combined['ece'] = ece
    # NEW METRICS ADDED TO OUTPUT
    combined['hypothesis_accuracy'] = hypothesis_accuracy
    combined['unknown_hypothesis_fraction'] = unknown_fraction
    return combined
