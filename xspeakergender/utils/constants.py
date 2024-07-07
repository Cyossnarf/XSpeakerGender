# coding: utf-8

import os


TOP_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

SEED = 42

LOGGING_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
DEFAULT_LOG_LVL = "WARNING"

N_ACCESS_TRIALS = 100

N_BOOTSTRAPS = 5000
CI_ALPHA = 5

MALE = 1
FEMALE = 2
GENDERS = ("neutral", "male", "female")  # must agree with lines before

MALE_INTERNAL = 0
FEMALE_INTERNAL = 1
GENDERS_INTERNAL = ("male", "female")  # must agree with lines before

N_SPEAKERS_M = "n_speakers_M"
N_SPEAKERS_F = "n_speakers_F"

SPEAKER_GENDER = "speaker_gender"
SPEAKER_ID = "speaker_id"
SENTENCE = "sentence"
SPEAKER_GENDER_PRED = "speaker_gender_pred"
SPK_GENDER_PROBA = "spk_gender_proba"
SPK_GENDER_PRED_PROBA = "spk_gender_pred_proba"
ATTRIBUTIONS = "attributions"

MODEL_NAME = "model_name"
MODEL_DIR = "model_dir"
SET_PATH = "set_path"
HYPERPARAMS = "hyperparameters"
GROUP = "group"
N_EXAMPLES = "n_examples"
N_PREDS = "n_preds"
P_EXAMPLES = "p_examples"
P_PREDS = "p_preds"
ACCURACY = "accuracy"
MCC = "mcc"
P4 = "p4"
F1 = "f1"
PRECISION = "precision"
RECALL = "recall"
POS = "P"
NEG = "N"
FEM = "F"
MAL = "M"
PRED = "pred"
PROBA = "proba"

WHOLE = "_whole"


def get_pred_column(gold_column, gold_pos_value):
    return "_".join([gold_column, str(gold_pos_value), PRED])


def get_gold_proba_column(gold_column, gold_pos_value):
    return "_".join([gold_column, str(gold_pos_value), PROBA])


def get_pred_proba_column(gold_column, gold_pos_value):
    return "_".join([gold_column, str(gold_pos_value), PRED, PROBA])


def get_n_examples_pos(str_pos=POS):
    return "_".join((N_EXAMPLES, str_pos))


def get_n_preds_pos(str_pos=POS):
    return "_".join((N_PREDS, str_pos))


def get_p_examples_neg(str_neg=NEG):
    return "_".join((P_EXAMPLES, str_neg))


def get_p_examples_pos(str_pos=POS):
    return "_".join((P_EXAMPLES, str_pos))


def get_p_preds_neg(str_neg=NEG):
    return "_".join((P_PREDS, str_neg))


def get_p_preds_pos(str_pos=POS):
    return "_".join((P_PREDS, str_pos))


def get_f1_neg(str_neg=NEG):
    return "_".join((F1, str_neg))


def get_precision_neg(str_neg=NEG):
    return "_".join((PRECISION, str_neg))


def get_recall_neg(str_neg=NEG):
    return "_".join((RECALL, str_neg))


def get_f1_pos(str_pos=POS):
    return "_".join((F1, str_pos))


def get_precision_pos(str_pos=POS):
    return "_".join((PRECISION, str_pos))


def get_recall_pos(str_pos=POS):
    return "_".join((RECALL, str_pos))


TOKEN_ID = "token_id"
TOKEN = "token"
N_SENTENCES = "n_sentences"
N_OCCURRENCES = "n_occurrences"
F_PROBA_DIFF_PLUS = "F_proba_diff+"
M_PROBA_DIFF_PLUS = "M_proba_diff+"
F_PROBA_REL_DIFF = "F_proba_rel_diff"
M2F_LBL_CHANGE = "M2F_lbl_change"
F2M_LBL_CHANGE = "F2M_lbl_change"
M2F_LBL_REL_CHANGE = "M2F_lbl_rel_change"
F_PROBA_REL_SCORE = "F_proba_rel_score"
M2F_LBL_REL_SCORE = "M2F_lbl_rel_score"
M_ATTR_PLUS = "M_attr+"
F_ATTR_PLUS = "F_attr+"
ATTR_REL = "attr"
ATTR_REL_SCORE = "attr_score"
SENT_ATTR_REL_SCORE = "sentence_attr_score"
SENT_ATTR_LABEL = "sentence_attr_label"

N_OCCUR_M = "n_occur_M"
N_OCCUR_F = "n_occur_F"
N_EXPECT_M = "n_expect_M"
N_EXPECT_F = "n_expect_F"
CHI_SQUARE = "chi_square"
SIGNED_CHI_SQUARE = "signed_chi_square"

RANDOM = "random"

OUTPUT_COLUMNS = [
    SPEAKER_GENDER,
    SPEAKER_GENDER_PRED,
    SPK_GENDER_PROBA,
    SPK_GENDER_PRED_PROBA,
    SENTENCE
]


def get_eval_columns(str_pos=POS, str_neg=NEG):
    eval_columns = [
        MODEL_NAME,
        MODEL_DIR,
        SET_PATH,
        HYPERPARAMS,
        GROUP,
        N_EXAMPLES,
        "_".join((N_EXAMPLES, str_pos)),
        "_".join((N_PREDS, str_pos)),
        "_".join((P_EXAMPLES, str_neg)),
        "_".join((P_EXAMPLES, str_pos)),
        "_".join((P_PREDS, str_neg)),
        "_".join((P_PREDS, str_pos)),
        ACCURACY,
        MCC,
        P4,
        "_".join((F1, str_neg)),
        "_".join((PRECISION, str_neg)),
        "_".join((RECALL, str_neg)),
        "_".join((F1, str_pos)),
        "_".join((PRECISION, str_pos)),
        "_".join((RECALL, str_pos))
    ]
    return eval_columns


EXTRACT_VOC_COLUMNS = [
    TOKEN,
    N_SENTENCES,
    F_PROBA_DIFF_PLUS,
    M_PROBA_DIFF_PLUS,
    F_PROBA_REL_DIFF,
    M2F_LBL_CHANGE,
    F2M_LBL_CHANGE,
    M2F_LBL_REL_CHANGE,
    F_PROBA_REL_SCORE,
    M2F_LBL_REL_SCORE
]

EXTRACT_VOC2_COLUMNS = [
    TOKEN,
    N_OCCURRENCES,
    M_ATTR_PLUS,
    F_ATTR_PLUS,
    ATTR_REL,
    ATTR_REL_SCORE
]
