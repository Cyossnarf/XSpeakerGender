# coding: utf-8

import argparse
import os
import sys

import pandas as pd

# We include the path of the toplevel package in the system path,
# so we can always use absolute imports within the package.
toplevel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if toplevel_path not in sys.path:
    sys.path.insert(1, toplevel_path)

import utils.constants as cst
import utils.util as utl


# FUNCTIONS ############################################################################################################

def filtre(data_file_path, criteria_str, out_data_file_path=None, sep="\t", decimal="."):
    df = pd.read_csv(data_file_path, sep=sep, decimal=decimal)
    n_examples_init = df.shape[0]
    print("Initial number of examples:")
    print(n_examples_init)

    df = utl.eval_criteria_string(criteria_str, df)

    n_examples_excluded = n_examples_init - df.shape[0]
    print("Number of selected examples:")
    print(df.shape[0], "(%.2f%%)" % (100 * df.shape[0] / n_examples_init))
    print("Number of excluded examples:")
    print(n_examples_excluded, "(%.2f%%)" % (100 * n_examples_excluded / n_examples_init))
    if out_data_file_path is not None:
        df.to_csv(out_data_file_path, sep=sep, index=False, decimal=decimal)


def join(data_file_path, compl_data_file_paths, out_data_file_path, sep="\t", decimal="."):
    df = pd.read_csv(data_file_path, sep=sep, decimal=decimal)
    for compl_data_file_path in compl_data_file_paths:
        compl_df = pd.read_csv(compl_data_file_path, sep=sep, decimal=decimal)
        # https://stackoverflow.com/questions/19125091/pandas-merge-how-to-avoid-duplicating-columns
        df = df.join(compl_df, lsuffix="DROP").filter(regex="^(?!.*DROP)")

    df.to_csv(out_data_file_path, sep=sep, index=False, decimal=decimal)


def balance(data_file_path, out_data_file_path, header, max_ratio, sep="\t", decimal=".", seed=cst.SEED):
    df = pd.read_csv(data_file_path, sep=sep, decimal=decimal)
    groups = df.groupby(header)
    counts = groups.size()
    print("Initial counts for", header, "classes:")
    print(counts)
    min_count = counts.min()
    max_count = int(max_ratio * min_count)
    # https://stackoverflow.com/questions/45839316/pandas-balancing-data
    df = groups.apply(lambda x: x.sample(min(max_count, x.shape[0]), random_state=seed)).reset_index(drop=True)
    counts = df[header].value_counts()
    print("Final counts for", header, "classes:")
    print(counts)
    df.to_csv(out_data_file_path, sep=sep, index=False, decimal=decimal)


def concat(data_file_path, compl_data_file_paths, out_data_file_path, sep="\t", decimal="."):
    df = pd.read_csv(data_file_path, sep=sep, decimal=decimal)
    for compl_data_file_path in compl_data_file_paths:
        compl_df = pd.read_csv(compl_data_file_path, sep=sep, decimal=decimal)
        df = pd.concat([df, compl_df], ignore_index=True, sort=False)

    df.to_csv(out_data_file_path, sep=sep, index=False, decimal=decimal)


# MAIN  ################################################################################################################

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", choices=["filter", "join", "balance", "concat"])

    parser.add_argument("--data_file", "-df",
                        help="csv/tsv file where data is contained")
    parser.add_argument("--complementary_data_files", "-cdf", nargs="+",
                        help="csv/tsv files where complementary data is contained")
    parser.add_argument("--output_data_file", "-odf",
                        help="csv/tsv file where processed data is written")
    parser.add_argument("--criteria_string", "-cs")
    parser.add_argument("--column_header", "-ch")
    parser.add_argument("--max_interclass_ratio", "-mir", type=float, default=1.5)
    parser.add_argument("--seed", "-s", type=int, default=cst.SEED)

    parser.add_argument("--separator", "-sep", choices=["comma", "tab"], default="tab")
    parser.add_argument("--decimal", "-d", default=".",
                        help="character to recognize as decimal point")

    args = parser.parse_args()
    return args


def main(args):
    mode = args.mode

    data_file_path = args.data_file
    compl_data_file_paths = args.complementary_data_files
    out_data_file_path = args.output_data_file
    criteria_str = args.criteria_string
    header = args.column_header
    max_ratio = args.max_interclass_ratio
    seed = args.seed
    sep = "," if args.separator == "comma" else "\t"
    decimal = args.decimal

    if mode == "filter":
        filtre(data_file_path, criteria_str, out_data_file_path=out_data_file_path, sep=sep, decimal=decimal)

    elif mode == "join":
        join(data_file_path, compl_data_file_paths, out_data_file_path, sep=sep, decimal=decimal)

    elif mode == "balance":
        balance(data_file_path, out_data_file_path, header, max_ratio, sep=sep, decimal=decimal, seed=seed)

    elif mode == "concat":
        concat(data_file_path, compl_data_file_paths, out_data_file_path, sep=sep, decimal=decimal)


if __name__ == "__main__":
    main(parse_args())
