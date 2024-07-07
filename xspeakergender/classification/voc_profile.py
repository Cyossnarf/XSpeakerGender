# coding: utf-8

import argparse
from collections import Counter
import os
import random
import sys

import numpy as np
import pandas as pd

# We include the path of the toplevel package in the system path,
# so we can always use absolute imports within the package.
toplevel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if toplevel_path not in sys.path:
    sys.path.insert(1, toplevel_path)

import classification.ft_bert as brt
import classification.util as cut
import utils.constants as cst
import utils.util as utl


def freq_profile(model_name, tokenizer_class, data_file_paths, vocab_dir_path, decimal=".", save_vocab=True,
                 save_hd_tl=True, sent_column=cst.SENTENCE):
    print("Loading tokenizer...")
    tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=brt.DO_LOWER_CASE)

    print("Loading dataset (%d file(s))..." % len(data_file_paths))
    dfs = list()
    for data_file_path in data_file_paths:
        print(data_file_path)
        df = pd.read_csv(data_file_path, sep="\t")
        print("Number of examples:")
        print(df.shape[0])
        sentences = df[sent_column].tolist()
        sentence = random.choice(sentences)
        print(" Original: ", sentence)
        print("Tokenized: ", tokenizer.tokenize(sentence))
        print("Token IDs: ", tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence)))
        dfs.append(df)

    if vocab_dir_path:
        os.makedirs(vocab_dir_path, exist_ok=True)

    # Computation as in https://core.ac.uk/download/pdf/376869.pdf (page 13)
    print("Computing chi square values...")
    df = pd.concat(dfs)
    observed = Counter()
    total = 0
    observed_lbl_wise = list()
    total_lbl_wise = list()
    n_speakers_lbl_wise = list()
    labels = cst.MALE, cst.FEMALE
    n_labels = len(labels)
    for label in labels:
        label_df = df.loc[df[cst.SPEAKER_GENDER] == label]
        sentences = label_df[sent_column].tolist()
        speaker_ids = (label_df[cst.SENTENCE_SRC] + label_df[cst.SPEAKER_ID]).tolist()
        label_observed = Counter()
        label_speakers = dict()
        for sentence, speaker_id in zip(sentences, speaker_ids):
            sent_tokens = tokenizer.tokenize(sentence)
            label_observed.update(sent_tokens)
            for token in sent_tokens:
                if token in label_speakers:
                    label_speakers[token].add(speaker_id)
                else:
                    label_speakers[token] = {speaker_id}

        label_total = sum(label_observed.values())
        observed_lbl_wise.append(label_observed)
        total_lbl_wise.append(label_total)
        observed.update(label_observed)
        total += label_total
        for token in label_speakers:
            label_speakers[token] = len(label_speakers[token])
        n_speakers_lbl_wise.append(label_speakers)

    expected_lbl_wise = list()
    tokens = observed.keys()
    n_tokens = len(observed)
    chi_square = np.zeros(n_tokens)
    for i in range(n_labels):
        label_expected = {t: total_lbl_wise[i] * observed[t] / total for t in tokens}
        expected_lbl_wise.append(label_expected)
        chi_square += [(observed_lbl_wise[i][t] - expected_lbl_wise[i][t]) ** 2 / expected_lbl_wise[i][t] for t in tokens]

    # Binary classification: Male +, Female -
    signed_chi_square = [x if observed_lbl_wise[0][t] > expected_lbl_wise[0][t] else -x for x, t in zip(chi_square, tokens)]

    print("  DONE.")

    data = {
        cst.TOKEN_ID: tokenizer.convert_tokens_to_ids(tokens),
        cst.TOKEN: tokens,
        cst.N_OCCUR_M: [observed_lbl_wise[0][t] for t in tokens],
        cst.N_OCCUR_F: [observed_lbl_wise[1][t] for t in tokens],
        cst.N_EXPECT_M: [expected_lbl_wise[0][t] for t in tokens],
        cst.N_EXPECT_F: [expected_lbl_wise[1][t] for t in tokens],
        cst.CHI_SQUARE: chi_square,
        cst.SIGNED_CHI_SQUARE: signed_chi_square,
        cst.N_SPEAKERS_M: [n_speakers_lbl_wise[0][t] if t in n_speakers_lbl_wise[0] else 0 for t in tokens],
        cst.N_SPEAKERS_F: [n_speakers_lbl_wise[1][t] if t in n_speakers_lbl_wise[1] else 0 for t in tokens],
    }
    df = pd.DataFrame(data)
    df = df.set_index(cst.TOKEN_ID)

    if save_vocab:
        vocab_file_path = os.path.join(vocab_dir_path, "vocab.tsv")
        df.to_csv(vocab_file_path, sep="\t", index=True, index_label="", decimal=decimal)

    if save_hd_tl:
        # For convenience
        for n in 1, 3, 9:
            # Dropping lines relative to tokens with low expected frequency
            df = df.loc[(df[cst.N_EXPECT_M] >= n) & (df[cst.N_EXPECT_F] >= n)]

            df = df.sort_values([cst.SIGNED_CHI_SQUARE], ascending=True)
            freq_df = pd.concat([df.head(25), df.tail(25)])
            freq_vocab_file_path = os.path.join(vocab_dir_path, "n%d_freq_vocab.tsv" % n)
            columns = [cst.TOKEN, cst.N_OCCUR_M, cst.N_OCCUR_F, cst.N_EXPECT_M, cst.N_EXPECT_F, cst.SIGNED_CHI_SQUARE,
                       cst.N_SPEAKERS_M, cst.N_SPEAKERS_F]
            freq_df.to_csv(freq_vocab_file_path, sep="\t", index=True, index_label="", decimal=decimal, columns=columns)


def rand_profile(model_name, tokenizer_class, data_file_paths, vocab_dir_path, decimal=".", save_vocab=True,
                 save_hd_tl=True, seed=cst.SEED, sent_column=cst.SENTENCE):
    random.seed(seed)

    print("Loading tokenizer...")
    tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=brt.DO_LOWER_CASE)

    print("Loading dataset (%d file(s))..." % len(data_file_paths))
    dfs = list()
    for data_file_path in data_file_paths:
        print(data_file_path)
        df = pd.read_csv(data_file_path, sep="\t")
        print("Number of examples:")
        print(df.shape[0])
        sentences = df[sent_column].tolist()
        sentence = random.choice(sentences)
        print(" Original: ", sentence)
        print("Tokenized: ", tokenizer.tokenize(sentence))
        print("Token IDs: ", tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence)))
        dfs.append(df)

    if vocab_dir_path:
        os.makedirs(vocab_dir_path, exist_ok=True)

    print("Observing occurrences...")
    df = pd.concat(dfs)
    observed = Counter()
    sentences = df[sent_column].tolist()
    for sentence in sentences:
        observed.update(tokenizer.tokenize(sentence))

    tokens = observed.keys()
    n_tokens = len(tokens)

    print("  DONE.")

    data = {
        cst.TOKEN_ID: tokenizer.convert_tokens_to_ids(tokens),
        cst.TOKEN: tokens,
        cst.N_OCCURRENCES: [observed[t] for t in tokens],
        cst.RANDOM: [random.gauss(0, 1) for _ in range(n_tokens)]
    }
    df = pd.DataFrame(data)
    df = df.set_index(cst.TOKEN_ID)

    if save_vocab:
        vocab_file_path = os.path.join(vocab_dir_path, "vocab.tsv")
        df.to_csv(vocab_file_path, sep="\t", index=True, index_label="", decimal=decimal)

    if save_hd_tl:
        # For convenience
        for n in 1, 3, 9:
            # Dropping lines relative to tokens that appear less than n times in the test file
            df = df.loc[df[cst.N_OCCURRENCES] >= n]

            df = df.sort_values([cst.RANDOM], ascending=True)
            attr_df = pd.concat([df.head(25), df.tail(25)])
            attr_vocab_file_path = os.path.join(vocab_dir_path, "n%d_attr_vocab.tsv" % n)
            columns = [cst.TOKEN, cst.N_OCCURRENCES, cst.RANDOM]
            attr_df.to_csv(attr_vocab_file_path, sep="\t", index=True, decimal=decimal, columns=columns)


def profile_coherence(model_name, tokenizer_class, test_file_path, vocab_file_path, attr_column,
                      stats_file_path=None, decimal=".", hyperparams="", max_len=brt.MAX_SEQ_LEN, criteria_str=None,
                      criteria_str2=None, ci=False, n_bootstraps=cst.N_BOOTSTRAPS, alpha=cst.CI_ALPHA,
                      cond_column=None):
    hp = ", col: %s, %s" % (attr_column, hyperparams) if hyperparams else ", col: %s" % attr_column
    model_dir_path = vocab_file_path

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=brt.DO_LOWER_CASE)

    # Load vocabulary
    print("Loading vocabulary...")
    df = pd.read_csv(vocab_file_path, sep="\t", index_col=0, decimal=decimal)

    # Filter vocabulary according to criteria string
    if criteria_str2 is not None:
        n_examples_init = df.shape[0]
        print("Initial number of tokens:")
        print(n_examples_init)
        df = utl.eval_criteria_string(criteria_str2, df)
        n_examples_excluded = n_examples_init - df.shape[0]
        print("Number of selected tokens:")
        print(df.shape[0], "(%.2f%%)" % (100 * df.shape[0] / n_examples_init))
        print("Number of excluded tokens:")
        print(n_examples_excluded, "(%.2f%%)" % (100 * n_examples_excluded / n_examples_init))

    profile = df[attr_column].to_dict()
    # print("Profile (first 15):", list(profile.items())[:15])  # DEBUG

    # Load dataset
    print("Loading data...")
    df = pd.read_csv(test_file_path, sep="\t")

    # Filter dataset according to criteria string
    if criteria_str is not None:
        n_examples_init = df.shape[0]
        print("Initial number of examples:")
        print(n_examples_init)
        df = utl.eval_criteria_string(criteria_str, df)
        n_examples_excluded = n_examples_init - df.shape[0]
        print("Number of selected examples:")
        print(df.shape[0], "(%.2f%%)" % (100 * df.shape[0] / n_examples_init))
        print("Number of excluded examples:")
        print(n_examples_excluded, "(%.2f%%)" % (100 * n_examples_excluded / n_examples_init))

    input_ids_list, _, _ = brt.load_data(df, tokenizer, "test", max_seq_len=max_len, padding="do_not_pad")

    attr_scores = list()
    for batch in input_ids_list:
        input_ids = batch  # (B=1, S)
        # If the token does not appear in the vocabulary, then use 0 as score (neutral)
        attr_score = sum([profile.get(input_id.item(), 0) for input_id in input_ids[0]])
        attr_scores.append(attr_score)

    # Code shared with attr_coherence
    cut.coherence(model_name, model_dir_path, test_file_path, df, attr_scores, hp=hp, rm_neutral=True,
                  stats_file_path=stats_file_path, decimal=decimal, ci=ci, n_bootstraps=n_bootstraps, alpha=alpha,
                  cond_column=cond_column)


# MAIN  ################################################################################################################

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", type=str, choices=["freq_profile", "rand_profile", "profile_coherence"])

    parser.add_argument("--model_type", "-modt", choices=list(brt.MODEL_TYPES.keys()),
                        help="type that defines model and tokenizer classes")
    parser.add_argument("--model_name", "-modn",
                        help="path to pre-trained model or shortcut name selected in the list: " +
                             ", ".join(brt.MODEL_NAMES))
    parser.add_argument("--data_files", "-datf", nargs="+",
                        help="data files to process")
    parser.add_argument("--vocab_dir", "-vocd",
                        help="directory where the extracted vocabulary is stored")
    parser.add_argument("--vocab_file", "-vocf",
                        help="tsv file that contains the extracted vocabulary")
    parser.add_argument("--test_file", "-tstf",
                        help="tsv file that contains the examples used for evaluation")
    parser.add_argument("--statistics_file", "-staf",
                        help="csv file of gathered statistics")
    parser.add_argument("--decimal", "-d", default=".",
                        help="character to recognize as decimal point")
    parser.add_argument("--hyperparameters", "-hyp",
                        help="string describing the hyperparameters used when training the model (for tracing purpose)")

    parser.add_argument("--dont_save_vocab", "-novoc", action="store_true",
                        help="do not save full vocabulary in vocab_dir")
    parser.add_argument("--dont_save_hd_tl", "-nohdtl", action="store_true",
                        help="do not save head-tail vocabulary in vocab_dir")

    parser.add_argument("--max_seq_length", "-msl", type=int, default=brt.MAX_SEQ_LEN,
                        help="maximum total input sequence length after tokenization"
                             "(sequences longer than this will be truncated, sequences shorter will be padded)")
    parser.add_argument("--attr_column", "-ac", default=cst.ATTR_REL_SCORE,
                        help="name of the column that contains the average attribution scores")
    parser.add_argument("--seed", "-s", type=int, default=cst.SEED,
                        help="random seed for initialization")
    parser.add_argument("--criteria_string", "-cs",
                        help="string describing a filter to apply to the examples before processing")
    parser.add_argument("--criteria_string2", "-cs2",
                        help="string describing a filter to apply to the tokens before processing")

    parser.add_argument("--confidence_intervals", "-ci", action="store_true",
                        help="compute the confidence intervals for evaluation metrics, using bootstrap resampling.")
    parser.add_argument("--n_bootstraps", "-nb", type=int, default=cst.N_BOOTSTRAPS,
                        help="number of bootstrap sets to be created")
    parser.add_argument("--ci_alpha", "-cia", type=float, default=cst.CI_ALPHA,
                        help="level of the interval"
                             "(the confidence interval will be computed between alpha/2 and 100-alpha/2 percentiles)")
    parser.add_argument("--cond_column", "-cc", default=None,
                        help="name of the column that contains the conditions of the samples")

    parser.add_argument("--sent_column", "-sc", default=cst.SENTENCE,
                        help="name of the column that contains the sentences")

    args = parser.parse_args()
    return args


def main(args):
    mode = args.mode

    model_type = args.model_type
    model_class, tokenizer_class = brt.MODEL_TYPES[model_type]
    model_name = args.model_name
    data_file_paths = args.data_files
    vocab_dir_path = args.vocab_dir
    vocab_file_path = args.vocab_file
    test_file_path = args.test_file
    stats_file_path = args.statistics_file
    decimal = args.decimal
    hyperparams = args.hyperparameters

    save_vocab = not args.dont_save_vocab
    save_hd_tl = not args.dont_save_hd_tl

    max_seq_len = args.max_seq_length
    attr_column = args.attr_column
    seed = args.seed
    criteria_str = args.criteria_string
    criteria_str2 = args.criteria_string2

    ci = args.confidence_intervals
    n_bootstraps = args.n_bootstraps
    ci_alpha = args.ci_alpha
    cond_column = args.cond_column

    sent_column = args.sent_column

    if mode == "freq_profile":
        freq_profile(model_name, tokenizer_class, data_file_paths, vocab_dir_path, decimal=decimal,
                     save_vocab=save_vocab, save_hd_tl=save_hd_tl, sent_column=sent_column)

    elif mode == "rand_profile":
        rand_profile(model_name, tokenizer_class, data_file_paths, vocab_dir_path, decimal=decimal,
                     save_vocab=save_vocab, save_hd_tl=save_hd_tl, seed=seed, sent_column=sent_column)

    elif mode == "profile_coherence":
        profile_coherence(model_name, tokenizer_class, test_file_path, vocab_file_path, attr_column,
                          stats_file_path=stats_file_path, decimal=decimal, hyperparams=hyperparams,
                          max_len=max_seq_len, criteria_str=criteria_str, criteria_str2=criteria_str2, ci=ci,
                          n_bootstraps=n_bootstraps, alpha=ci_alpha, cond_column=cond_column)


if __name__ == "__main__":
    main(parse_args())
