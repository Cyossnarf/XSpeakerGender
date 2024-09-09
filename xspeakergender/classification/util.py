import os
import sys

# We include the path of the toplevel package in the system path,
# so we can always use absolute imports within the package.
toplevel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if toplevel_path not in sys.path:
    sys.path.insert(1, toplevel_path)

import classification.metrics as mtr
import utils.constants as cst


def coherence(model_name, model_dir_path, test_file_path, df, attr_scores, hp="", rm_neutral=False,
              stats_file_path=None, decimal=".", ci=False, n_bootstraps=cst.N_BOOTSTRAPS, alpha=cst.CI_ALPHA,
              cond_column=None):

    df[cst.SENT_ATTR_REL_SCORE] = attr_scores
    df[cst.SENT_ATTR_LABEL] = df[cst.SENT_ATTR_REL_SCORE] < 0
    df[cst.SPEAKER_GENDER_PRED] -= 1
    df[cst.SPEAKER_GENDER] -= 1

    # Remove neutral examples (e.g. examples that do not contain any word from the vocabulary)
    if rm_neutral:
        n_examples_init = df.shape[0]
        df = df.loc[df[cst.SENT_ATTR_REL_SCORE] != 0]
        n_examples_excluded = n_examples_init - df.shape[0]
        print("Number of excluded neutral examples:")
        print(n_examples_excluded, "(%.2f%%)" % (100 * n_examples_excluded / n_examples_init))

    str_pos = cst.FEM
    str_neg = cst.MAL

    gold_labels = df[cst.SPEAKER_GENDER_PRED].to_numpy()
    gold_labels2 = df[cst.SPEAKER_GENDER].to_numpy()
    pred_labels = df[cst.SENT_ATTR_LABEL].to_numpy()
    conditions = None if cond_column is None else df[cond_column].to_numpy()

    hyperparams = "ref: %s" % cst.SPEAKER_GENDER_PRED
    print(hyperparams)
    hyperparams += hp
    mtr.binary_classification_scores(gold_labels, pred_labels, stats_file_path=stats_file_path, model_name=model_name,
                                     model_dir_path=model_dir_path, test_file_path=test_file_path, decimal=decimal,
                                     str_pos=str_pos, str_neg=str_neg, hyperparams=hyperparams, ci=ci,
                                     n_bootstraps=n_bootstraps, alpha=alpha, conditions=conditions)
    hyperparams = "ref: %s" % cst.SPEAKER_GENDER
    print(hyperparams)
    hyperparams += hp
    mtr.binary_classification_scores(gold_labels2, pred_labels, stats_file_path=stats_file_path, model_name=model_name,
                                     model_dir_path=model_dir_path, test_file_path=test_file_path, decimal=decimal,
                                     str_pos=str_pos, str_neg=str_neg, hyperparams=hyperparams, ci=ci,
                                     n_bootstraps=n_bootstraps, alpha=alpha, conditions=conditions)
