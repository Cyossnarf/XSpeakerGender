# coding: utf-8

import datetime
import fcntl
import os
import re
import sys
import time

# We include the path of the toplevel package in the system path,
# so we can always use absolute imports within the package.
toplevel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if toplevel_path not in sys.path:
    sys.path.insert(1, toplevel_path)

import utils.constants as cst


def eval_criteria_string(criteria_str, df):
    criteria_str = re.sub(r"\[(.+?)]", r"df['\1']", criteria_str)
    df = df.loc[eval(criteria_str)]
    return df


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    :param elapsed: time as a float
    :return: time as a formatted string
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def report_progress(step, n_batches, t0, print_every=40, batch_str="Batch"):
    # Progress update every `print_every` batches.
    if step % print_every == 0 and not step == 0:
        # Calculate elapsed time in minutes.
        elapsed = format_time(time.time() - t0)
        # Report progress.
        print("  {:} {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(batch_str, step, n_batches, elapsed))


def robust_write(stats_file_path, df, decimal=None):
    dir_path = os.path.dirname(stats_file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    # https://technicalmasterblog.wordpress.com/2019/08/07/python-file-locking-and-unlocking/
    n = 1
    file_exists = os.path.isfile(stats_file_path)
    while n < cst.N_ACCESS_TRIALS:
        try:
            # try/except in case the file is still locked by another process
            # open the file for editing
            with open(stats_file_path, "a") as stats_file:
                # lock the file
                fcntl.flock(stats_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # edit it
                header = (n == 1) and not file_exists
                df.to_csv(stats_file, index=False, decimal=decimal, header=header, mode="a")
                # and now unlock it so other processed can edit it!
                fcntl.flock(stats_file, fcntl.LOCK_UN)
                break
        except IOError as e:
            # wait before retrying
            time.sleep(0.05)
            n += 1

    if n == cst.N_ACCESS_TRIALS:
        print("Could not manage to write results in")
        print(stats_file_path)
    elif n > 1:
        print("Succeeded in writing results on attempt n°", n)


def write_lines(lines, file_path, newline=True, add=False, make_dir=True, convert=False):
    mode = 'a' if add else 'w'
    if make_dir:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
    if convert:
        lines = map(str, lines)
    if newline:
        lines = map(lambda l: l + '\n', lines)
    with open(file_path, mode) as file:
        file.writelines(lines)
