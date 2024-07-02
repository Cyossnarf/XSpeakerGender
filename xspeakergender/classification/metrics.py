# coding: utf-8

import os
import sys

from confidence_intervals import evaluate_with_conf_int
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score

# We include the path of the toplevel package in the system path,
# so we can always use absolute imports within the package.
toplevel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if toplevel_path not in sys.path:
    sys.path.insert(1, toplevel_path)

import utils.constants as cst
import utils.util as utl


def p4_aux(prec_0, rec_0, prec_1, rec_1):
    p4 = 4 / (1 / prec_0 + 1 / rec_0 + 1 / prec_1 + 1 / rec_1)
    return p4


def p4_score(gold_labels, pred_labels):
    prec_1 = precision_score(gold_labels, pred_labels, pos_label=1)
    rec_1 = recall_score(gold_labels, pred_labels, pos_label=1)
    prec_0 = precision_score(gold_labels, pred_labels, pos_label=0)
    rec_0 = recall_score(gold_labels, pred_labels, pos_label=0)
    p4 = p4_aux(prec_0, rec_0, prec_1, rec_1)
    return p4


def binary_classification_scores(gold_labels, pred_labels, stats_file_path=None, model_name=None, model_dir_path=None,
                                 test_file_path=None, decimal=None, str_pos=cst.POS, str_neg=cst.NEG,
                                 group_name=cst.WHOLE, hyperparams="", ci=False, n_bootstraps=cst.N_BOOTSTRAPS,
                                 alpha=cst.CI_ALPHA, conditions=None):
    # Calculate Distribution
    n_examples = len(gold_labels)
    n_gold_lbl_pos = sum(gold_labels)
    n_gold_lbl_neg = n_examples - n_gold_lbl_pos
    p_gold_lbl_pos = n_gold_lbl_pos / n_examples
    p_gold_lbl_neg = n_gold_lbl_neg / n_examples
    print("  Gold %s class: %.3f (%d / %d)" % (str_pos, p_gold_lbl_pos, n_gold_lbl_pos, n_examples))
    print("  Gold %s class: %.3f (%d / %d)" % (str_neg, p_gold_lbl_neg, n_gold_lbl_neg, n_examples))
    n_pred_lbl_pos = sum(pred_labels)
    n_pred_lbl_neg = n_examples - n_pred_lbl_pos
    p_pred_lbl_pos = n_pred_lbl_pos / n_examples
    p_pred_lbl_neg = n_pred_lbl_neg / n_examples
    print("  Pred %s class: %.3f (%d / %d)" % (str_pos, p_pred_lbl_pos, n_pred_lbl_pos, n_examples))
    print("  Pred %s class: %.3f (%d / %d)" % (str_neg, p_pred_lbl_neg, n_pred_lbl_neg, n_examples))

    # Calculate Accuracy
    if ci:
        metric = accuracy_score
        accuracy, accuracy_ci = evaluate_with_conf_int(pred_labels, metric, labels=gold_labels,
                                                       conditions=conditions, num_bootstraps=n_bootstraps, alpha=alpha)
        print("  Accuracy: %.3f (%.3f, %.3f)" % (accuracy, accuracy_ci[0], accuracy_ci[1]))
        accuracy = str((accuracy, accuracy_ci))
    else:
        accuracy = accuracy_score(gold_labels, pred_labels)
        print("  Accuracy: %.3f" % accuracy)

    # Calculate MCC
    if ci:
        metric = matthews_corrcoef
        mcc, mcc_ci = evaluate_with_conf_int(pred_labels, metric, labels=gold_labels,
                                             conditions=conditions, num_bootstraps=n_bootstraps, alpha=alpha)
        print("  MCC: %.3f (%.3f, %.3f)" % (mcc, mcc_ci[0], mcc_ci[1]))
        mcc = str((mcc, mcc_ci))
    else:
        mcc = matthews_corrcoef(gold_labels, pred_labels)
        print("  MCC: %.3f" % mcc)

    # Calculate F1 (not symmetric)
    precision_pos = precision_score(gold_labels, pred_labels, pos_label=1)
    recall_pos = recall_score(gold_labels, pred_labels, pos_label=1)
    precision_neg = precision_score(gold_labels, pred_labels, pos_label=0)
    recall_neg = recall_score(gold_labels, pred_labels, pos_label=0)
    if ci:
        def metric(g_lbls, p_lbls): return f1_score(g_lbls, p_lbls, pos_label=1)
        f1_pos, f1_pos_ci = evaluate_with_conf_int(pred_labels, metric, labels=gold_labels,
                                                   conditions=conditions, num_bootstraps=n_bootstraps, alpha=alpha)
        print("  F1(%s): %.3f (%.3f, %.3f) (prec.: %.3f, rec.: %.3f)" %
              (str_pos, f1_pos, f1_pos_ci[0], f1_pos_ci[1], precision_pos, recall_pos))
        f1_pos = str((f1_pos, f1_pos_ci))

        def metric(g_lbls, p_lbls): return f1_score(g_lbls, p_lbls, pos_label=0)
        f1_neg, f1_neg_ci = evaluate_with_conf_int(pred_labels, metric, labels=gold_labels,
                                                   conditions=conditions, num_bootstraps=n_bootstraps, alpha=alpha)
        print("  F1(%s): %.3f (%.3f, %.3f) (prec.: %.3f, rec.: %.3f)" %
              (str_neg, f1_neg, f1_neg_ci[0], f1_neg_ci[1], precision_neg, recall_neg))
        f1_neg = str((f1_neg, f1_neg_ci))

        metric = p4_score
        p4, p4_ci = evaluate_with_conf_int(pred_labels, metric, labels=gold_labels,
                                           conditions=conditions, num_bootstraps=n_bootstraps, alpha=alpha)
        print("  P4: %.3f (%.3f, %.3f)" % (p4, p4_ci[0], p4_ci[1]))
        p4 = str((p4, p4_ci))
    else:
        f1_pos = f1_score(gold_labels, pred_labels, pos_label=1)
        print("  F1(%s): %.3f (prec.: %.3f, rec.: %.3f)" % (str_pos, f1_pos, precision_pos, recall_pos))
        f1_neg = f1_score(gold_labels, pred_labels, pos_label=0)
        print("  F1(%s): %.3f (prec.: %.3f, rec.: %.3f)" % (str_neg, f1_neg, precision_neg, recall_neg))
        p4 = p4_aux(precision_neg, recall_neg, precision_pos, recall_pos)
        print("  P4: %.3f" % p4)

    data = {p: list() for p in cst.get_eval_columns(str_pos=str_pos, str_neg=str_neg)}

    data[cst.MODEL_NAME].append(model_name)
    data[cst.MODEL_DIR].append(model_dir_path)
    data[cst.SET_PATH].append(test_file_path)
    data[cst.HYPERPARAMS].append(hyperparams)
    data[cst.GROUP].append(group_name)
    data[cst.N_EXAMPLES].append(n_examples)
    data[cst.get_n_examples_pos(str_pos)].append(n_gold_lbl_pos)
    data[cst.get_n_preds_pos(str_pos)].append(n_pred_lbl_pos)
    data[cst.get_p_examples_neg(str_neg)].append(p_gold_lbl_neg)
    data[cst.get_p_examples_pos(str_pos)].append(p_gold_lbl_pos)
    data[cst.get_p_preds_neg(str_neg)].append(p_pred_lbl_neg)
    data[cst.get_p_preds_pos(str_pos)].append(p_pred_lbl_pos)
    data[cst.ACCURACY].append(accuracy)
    data[cst.MCC].append(mcc)
    data[cst.P4].append(p4)
    data[cst.get_f1_neg(str_neg)].append(f1_neg)
    data[cst.get_precision_neg(str_neg)].append(precision_neg)
    data[cst.get_recall_neg(str_neg)].append(recall_neg)
    data[cst.get_f1_pos(str_pos)].append(f1_pos)
    data[cst.get_precision_pos(str_pos)].append(precision_pos)
    data[cst.get_recall_pos(str_pos)].append(recall_pos)

    df = pd.DataFrame(data)

    if stats_file_path is not None:
        utl.robust_write(stats_file_path, df, decimal=decimal)

    return df
