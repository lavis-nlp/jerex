from typing import List, Tuple

from sklearn.metrics import precision_recall_fscore_support as prfs

METRIC_LABELS = ['prec_micro', 'rec_micro', 'f1_micro', 'prec_macro', 'rec_macro', 'f1_macro']


def score(gt: List[List[Tuple]], pred: List[List[Tuple]], type_idx=None, print_results: bool = False):
    assert len(gt) == len(pred)

    gt_flat = []
    pred_flat = []
    labels = set()

    for (sample_gt, sample_pred) in zip(gt, pred):
        union = []
        for s in sample_gt:
            if s not in union:
                union.append(s)

        for s in sample_pred:
            if s not in union:
                union.append(s)

        for s in union:
            if s in sample_gt:
                if type_idx is not None:
                    t = s[type_idx]
                    gt_flat.append(t.index)
                    labels.add(t)
                else:
                    gt_flat.append(0)
            else:
                gt_flat.append(-1)

            if s in sample_pred:
                if type_idx is not None:
                    t = s[type_idx]
                    pred_flat.append(t.index)
                    labels.add(t)
                else:
                    pred_flat.append(0)
            else:
                pred_flat.append(-1)

    if type_idx is not None:
        labels, labels_str = zip(*[(l.index, l.short_name) for l in labels])
    else:
        labels, labels_str = [0], ['Binary']

    metrics = _compute_metrics(gt_flat, pred_flat, labels, labels_str, print_results)
    return metrics


def score_single(gt: List[Tuple], pred: List[Tuple], type_idx=None, print_results: bool = False):
    return score([gt], [pred], type_idx=type_idx, print_results=print_results)


def _compute_metrics(gt_all, pred_all, labels, labels_str, print_results: bool = False):
    per_type = prfs(gt_all, pred_all, labels=labels, average=None, zero_division=0)
    micro = prfs(gt_all, pred_all, labels=labels, average='micro', zero_division=0)[:-1]
    macro = prfs(gt_all, pred_all, labels=labels, average='macro', zero_division=0)[:-1]
    total_support = sum(per_type[-1])

    if print_results:
        _print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], labels_str)

    metrics = [m * 100 for m in micro + macro]
    return dict(zip(METRIC_LABELS, metrics))


def _print_results(per_type: List, micro: List, macro: List, types: List):
    columns = ('type', 'precision', 'recall', 'f1-score', 'support')

    row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
    results = [row_fmt % columns, '\n']

    metrics_per_type = []
    for i, t in enumerate(types):
        metrics = []
        for j in range(len(per_type)):
            metrics.append(per_type[j][i])
        metrics_per_type.append(metrics)

    for m, t in zip(metrics_per_type, types):
        results.append(row_fmt % _get_row(m, t))
        results.append('\n')

    results.append('\n')

    # micro
    results.append(row_fmt % _get_row(micro, 'micro'))
    results.append('\n')

    # macro
    results.append(row_fmt % _get_row(macro, 'macro'))

    results_str = ''.join(results)
    print(results_str)


def _get_row(data, label):
    row = [label]
    for i in range(len(data) - 1):
        row.append("%.2f" % (data[i] * 100))
    row.append(data[3])
    return tuple(row)
