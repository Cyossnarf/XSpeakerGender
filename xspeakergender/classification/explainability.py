# coding: utf-8

import argparse
import os
import sys
import time

from captum.attr import visualization as viz
from captum.attr import LayerIntegratedGradients, Lime
from captum._utils.models.linear_model import SkLearnLasso
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import \
    BertForSequenceClassification, \
    CamembertForSequenceClassification, \
    FlaubertForSequenceClassification

# We include the path of the toplevel package in the system path,
# so we can always use absolute imports within the package.
toplevel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if toplevel_path not in sys.path:
    sys.path.insert(1, toplevel_path)

import classification.ft_bert as brt
import classification.util as cut
import utils.constants as cst
import utils.util as utl


ATTR_CLASS = cst.MALE_INTERNAL
ATTR_LABEL = cst.GENDERS_INTERNAL[ATTR_CLASS]
N_LIG_STEPS = 600
LIG_BATCH_SIZE = 5
N_LIME_SAMPLES = 200
# Using an alpha too big can result in null attributions: https://github.com/pytorch/captum/issues/1008
LIME_ALPHA = 0.001


def target_layer(model_class, model):
    # todo: find something better
    if model_class is BertForSequenceClassification:
        # https://github.com/huggingface/transformers/blob/v4.33.3/src/transformers/models/bert/modeling_bert.py#L1523
        layer = model.bert.embeddings
    elif model_class is CamembertForSequenceClassification:
        # https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/camembert/modeling_camembert.py#L1037
        layer = model.roberta.embeddings
    elif model_class is FlaubertForSequenceClassification:
        # https://github.com/huggingface/transformers/blob/v4.33.3/src/transformers/models/flaubert/modeling_flaubert.py#L757
        layer = model.transformer.embeddings
    else:
        layer = None

    return layer


def make_baselines(input_ids_list, tokenizer):
    baseline_ids_list = list()
    for input_ids in input_ids_list:  # (B=1, S)
        baseline_ids = (input_ids.shape[1] - 2) * [tokenizer.pad_token_id]
        baseline_ids = tokenizer.build_inputs_with_special_tokens(baseline_ids)
        baseline_ids = torch.tensor([baseline_ids])
        baseline_ids_list.append(baseline_ids)

    return baseline_ids_list


# saved_act = None
def save_act(module, inp, out):
    # global saved_act
    # saved_act = out
    return saved_act


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)

    return attributions


# FUNCTIONS ############################################################################################################

def occlusion(model_class, tokenizer_class, model_dir_path, test_file_path, explain_dir_path, decimal=".",
              max_len=brt.MAX_SEQ_LEN, criteria_str=None, weight_tokens=False, save_exples=True, save_visual=True,
              save_attr=True, save_vocab=True, save_hd_tl=True, attr_file_path=None, sort_exples=True):
    model, tokenizer = brt.load_model(model_class, tokenizer_class, model_dir_path)
    # Put model in evaluation mode
    model.eval()

    # Load dataset.
    df = pd.read_csv(test_file_path, sep="\t")
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

    try:
        assert df.shape[0] != 0
    except AssertionError:
        print("Vocabulary cannot be extracted, for no examples were selected.")
        return

    if explain_dir_path:
        os.makedirs(explain_dir_path, exist_ok=True)

    # Sort examples by pred_label and pred_label_proba
    if sort_exples:
        df = df.sort_values(by=[cst.SPEAKER_GENDER_PRED, cst.SPK_GENDER_PRED_PROBA], ascending=False)

    if save_exples:
        examples_file_path = os.path.join(explain_dir_path, "selected_examples.tsv")
        df.to_csv(examples_file_path, sep="\t", index=True)

    pred_lbl_probas = df[cst.SPK_GENDER_PRED_PROBA].tolist()
    pred_labels = df[cst.SPEAKER_GENDER_PRED].tolist()
    gold_labels = df[cst.SPEAKER_GENDER].tolist()
    pred_f_probas = [proba if lbl == cst.FEMALE else 1 - proba for proba, lbl in zip(pred_lbl_probas, pred_labels)]

    input_ids_list, _, _ = brt.load_data(df, tokenizer, "test", max_seq_len=max_len, padding="do_not_pad")

    # Number of batches
    n_batches = len(input_ids_list)
    assert n_batches == len(pred_labels)
    assert n_batches == len(pred_f_probas)

    # Tracking variables
    vocab = tokenizer.get_vocab()
    data = {
        cst.TOKEN: list(sorted(vocab, key=vocab.get)),
        cst.N_SENTENCES: np.zeros(tokenizer.vocab_size, dtype="int"),
        cst.F_PROBA_DIFF_PLUS: np.zeros(tokenizer.vocab_size),
        cst.M_PROBA_DIFF_PLUS: np.zeros(tokenizer.vocab_size),
        cst.M2F_LBL_CHANGE: np.zeros(tokenizer.vocab_size),
        cst.F2M_LBL_CHANGE: np.zeros(tokenizer.vocab_size)
    }
    assert len(data[cst.TOKEN]) == tokenizer.vocab_size

    # Extraction on test set
    print("Extracting labels for test sentences...")
    proba_diff_vis_data_records = list()
    lbl_change_vis_data_records = list()
    proba_diff_attributions_list = list()
    lbl_change_attributions_list = list()
    # Measure how long the Extraction takes.
    t0 = time.time()

    # Predict
    for step, batch in enumerate(zip(input_ids_list, gold_labels, pred_labels, pred_lbl_probas, pred_f_probas)):
        utl.report_progress(step, n_batches, t0)
        b_input_ids, gold_label, pred_label, pred_lbl_proba, pred_f_proba = batch

        # Add batch to GPU
        b_input_ids = b_input_ids.to(brt.DEVICE)  # (B=1, S)

        # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
            # One batch consists of one sentence, duplicated to the number of unique tokens within,
            # with each duplicate being masked for one unique token.
            unique_ids = b_input_ids.unique()                                # (B=n_unique,)
            # Recap of special tokens for BERT models
            #   bos_token_id, eos_token_id, cls_token_id, sep_token_id, pad_token_id, mask_token_id, (unk_token_id)
            # bert-base-multilingual-uncased
            #   None None 101 102 0 103
            # flaubert/flaubert_base_uncased, nherve/flaubert-oral-asr_nb, nherve/flaubert-oral-mixed
            #   0 None 1 1 2 5
            # camembert-base, camembert/camembert-base-ccnet
            #   5 6 5 6 1 32004
            # cl-tohoku/bert-base-japanese
            #   None None 2 3 0 4 (1)
            # /!\ CAUTION:
            #   (unique_ids != None) * (unique_ids != None) * (unique_ids != 2)
            #     = True * True * <bool tensor> = ((True * True) * <bool tensor>) = 1 * <bool tensor>
            #     = <int tensor>
            #   While:
            #   (unique_ids != 2) * (unique_ids != None) * (unique_ids != None)
            #     = <bool tensor> * True * True = ((<bool tensor> * True) * True) = <bool tensor> * True
            #     = <bool tensor>
            #   Conclusion:
            #   The two first special token ids checked MUST NOT be None
            unique_ids = unique_ids[(unique_ids != tokenizer.pad_token_id) *
                                    (unique_ids != tokenizer.unk_token_id) *
                                    (unique_ids != tokenizer.cls_token_id) *
                                    (unique_ids != tokenizer.sep_token_id) *
                                    (unique_ids != tokenizer.bos_token_id) *
                                    (unique_ids != tokenizer.eos_token_id)]  # (B=n_unique,)
            n_unique_ids = unique_ids.shape[0]
            seq_len = b_input_ids.shape[1]

            indices = b_input_ids[0]  # (S,)
            b_input_ids = b_input_ids.expand(n_unique_ids, -1)    # (B=n_unique, S)

            b_mask_ids = unique_ids.unsqueeze(1).expand(-1, seq_len)  # (B=n_unique, S)
            b_mask = b_input_ids == b_mask_ids                        # (B=n_unique, S)
            b_input_ids = b_input_ids.masked_fill(b_mask, tokenizer.mask_token_id)

            # Perform a forward pass (evaluate the model on this batch).
            outputs = model(b_input_ids, token_type_ids=None)

        b_logits = outputs[0]
        b_soft_preds = b_logits.softmax(dim=1)  # (B=n_unique, K=2)
        # Move to CPU
        indices = indices.detach().cpu().numpy()
        unique_ids = unique_ids.detach().cpu().numpy()
        b_soft_preds = b_soft_preds.detach().cpu().numpy()
        b_mask = b_mask.detach().cpu().numpy()
        # For each sample, pick the label (0 or 1) with the higher score.
        b_pred_labels = np.argmax(b_soft_preds, axis=1)  # (B=n_unique,)
        # stuff
        b_weight = n_unique_ids if weight_tokens else 1
        b_proba_diffs = b_soft_preds[:, cst.FEMALE_INTERNAL] - pred_f_proba  # (B=n_unique,)
        b_proba_diffs = b_weight * b_proba_diffs                             # (B=n_unique,)
        b_lbl_changes = b_pred_labels - (pred_label - 1)                     # (B=n_unique,)
        b_lbl_changes = b_weight * b_lbl_changes                             # (B=n_unique,)

        # Normalizing b_mask
        b_normed_mask_t = b_mask.transpose() / b_mask.sum(axis=1)  # (S, B=n_unique)
        # Compute token attributions
        proba_diff_attributions = np.sum(b_normed_mask_t * b_proba_diffs, axis=1)  # (S,)
        lbl_change_attributions = np.sum(b_normed_mask_t * b_lbl_changes, axis=1)  # (S,)

        proba_diff_attributions_list.append(torch.from_numpy(proba_diff_attributions))
        lbl_change_attributions_list.append(torch.from_numpy(lbl_change_attributions))

        # Necessary only for visualization
        if save_visual:
            tokens = tokenizer.convert_ids_to_tokens(indices)  # (S,)

            vis_data_record = viz.VisualizationDataRecord(proba_diff_attributions,
                                                          pred_lbl_proba,
                                                          cst.GENDERS[pred_label],
                                                          cst.GENDERS[gold_label],
                                                          ATTR_LABEL,
                                                          proba_diff_attributions.sum(),
                                                          tokens,
                                                          None)
            proba_diff_vis_data_records.append(vis_data_record)
            vis_data_record = viz.VisualizationDataRecord(lbl_change_attributions,
                                                          pred_lbl_proba,
                                                          cst.GENDERS[pred_label],
                                                          cst.GENDERS[gold_label],
                                                          ATTR_LABEL,
                                                          lbl_change_attributions.sum(),
                                                          tokens,
                                                          None)
            lbl_change_vis_data_records.append(vis_data_record)

        for unique_id, proba_diff, lbl_change in zip(unique_ids, b_proba_diffs, b_lbl_changes):
            data[cst.N_SENTENCES][unique_id] += 1
            if proba_diff > 0:
                data[cst.F_PROBA_DIFF_PLUS][unique_id] += proba_diff
            else:
                data[cst.M_PROBA_DIFF_PLUS][unique_id] -= proba_diff
            if lbl_change > 0:
                data[cst.M2F_LBL_CHANGE][unique_id] += lbl_change
            else:
                data[cst.F2M_LBL_CHANGE][unique_id] -= lbl_change

    print("  DONE.")

    if save_visual:
        visual = viz.visualize_text(proba_diff_vis_data_records)
        visual_file_path = os.path.join(explain_dir_path, "proba_diff_visual.html")
        with open(visual_file_path, "w") as file:
            file.write(visual.data)

        visual = viz.visualize_text(lbl_change_vis_data_records)
        visual_file_path = os.path.join(explain_dir_path, "lbl_change_visual.html")
        with open(visual_file_path, "w") as file:
            file.write(visual.data)

    if save_attr:
        proba_diff_attr_file_path = os.path.join(explain_dir_path, "proba_diff_attributions.pt")
        torch.save(proba_diff_attributions_list, proba_diff_attr_file_path)

        lbl_change_attr_file_path = os.path.join(explain_dir_path, "lbl_change_attributions.pt")
        torch.save(lbl_change_attributions_list, lbl_change_attr_file_path)

    df = pd.DataFrame(data)
    # Dropping lines relative to tokens that do not appear in the test file
    df = df.loc[df[cst.N_SENTENCES] != 0]
    # Computing other values
    df[cst.F_PROBA_REL_DIFF] = df[cst.F_PROBA_DIFF_PLUS] - df[cst.M_PROBA_DIFF_PLUS]
    df[cst.M2F_LBL_REL_CHANGE] = df[cst.M2F_LBL_CHANGE] - df[cst.F2M_LBL_CHANGE]
    df[cst.F_PROBA_REL_SCORE] = df[cst.F_PROBA_REL_DIFF] / df[cst.N_SENTENCES]
    df[cst.M2F_LBL_REL_SCORE] = df[cst.M2F_LBL_REL_CHANGE] / df[cst.N_SENTENCES]

    if save_vocab:
        vocab_file_path = os.path.join(explain_dir_path, "vocab.tsv")
        df.to_csv(vocab_file_path, sep="\t", index=True, decimal=decimal, columns=cst.EXTRACT_VOC_COLUMNS)

    if save_hd_tl:
        # For convenience
        for n in 1, 3, 9:
            # Dropping lines relative to tokens that appear less than n times in the test file
            df = df.loc[df[cst.N_SENTENCES] >= n]

            df = df.sort_values([cst.F_PROBA_REL_SCORE], ascending=True)
            proba_diff_df = pd.concat([df.head(25), df.tail(25)])
            proba_diff_vocab_file_path = os.path.join(explain_dir_path, "n%d_proba_diff_vocab.tsv" % n)
            columns = [cst.TOKEN, cst.N_SENTENCES, cst.F_PROBA_REL_SCORE]
            proba_diff_df.to_csv(proba_diff_vocab_file_path, sep="\t", index=True, decimal=decimal, columns=columns)

            df = df.sort_values([cst.M2F_LBL_REL_SCORE], ascending=True)
            lbl_change_df = pd.concat([df.head(25), df.tail(25)])
            lbl_change_vocab_file_path = os.path.join(explain_dir_path, "n%d_lbl_change_vocab.tsv" % n)
            columns = [cst.TOKEN, cst.N_SENTENCES, cst.M2F_LBL_REL_SCORE]
            lbl_change_df.to_csv(lbl_change_vocab_file_path, sep="\t", index=True, decimal=decimal, columns=columns)


def lay_int_grad(model_class, tokenizer_class, model_dir_path, test_file_path, explain_dir_path, decimal=".",
                 max_len=brt.MAX_SEQ_LEN, criteria_str=None, n_steps=N_LIG_STEPS, batch_size=LIG_BATCH_SIZE,
                 save_exples=True, save_visual=True, save_attr=True, save_vocab=True, save_hd_tl=True,
                 attr_file_path=None, sort_exples=True):
    model, tokenizer = brt.load_model(model_class, tokenizer_class, model_dir_path)
    # Put model in evaluation mode
    model.eval()

    # Inspired by https://colab.research.google.com/drive/1pgAbzUF2SzF0BdFtGpJbZPWUOhFxT2NZ (mostly)
    # and by https://captum.ai/tutorials/IMDB_TorchText_Interpret
    # and by https://coderzcolumn.com/tutorials/artificial-intelligence/captum-for-pytorch-text-classification-networks
    # and by https://www.kaggle.com/code/rhtsingh/interpreting-text-models-with-bert-on-tpu
    def custom_forward(b_input_ids):
        outputs = model(b_input_ids, token_type_ids=None)
        b_soft_preds = outputs[0].softmax(dim=1)  # (B=1, K=2)
        return b_soft_preds[:, ATTR_CLASS]

    layer = target_layer(model_class, model)
    lig = LayerIntegratedGradients(custom_forward, layer)
    hook = layer.register_forward_hook(save_act)
    hook.remove()

    # Load dataset
    df = pd.read_csv(test_file_path, sep="\t")

    # Load attributions if they have already been computed
    if attr_file_path is None:
        attributions_list = df.shape[0] * [None]
    else:
        attributions_list = torch.load(attr_file_path)
        assert len(attributions_list) == df.shape[0]

    # Filter dataset according to criteria string
    if criteria_str is not None:
        df[cst.ATTRIBUTIONS] = attributions_list

        n_examples_init = df.shape[0]
        print("Initial number of examples:")
        print(n_examples_init)
        df = utl.eval_criteria_string(criteria_str, df)
        n_examples_excluded = n_examples_init - df.shape[0]
        print("Number of selected examples:")
        print(df.shape[0], "(%.2f%%)" % (100 * df.shape[0] / n_examples_init))
        print("Number of excluded examples:")
        print(n_examples_excluded, "(%.2f%%)" % (100 * n_examples_excluded / n_examples_init))

        attributions_list = df[cst.ATTRIBUTIONS].tolist()
        df.drop(cst.ATTRIBUTIONS, axis=1)

    try:
        assert df.shape[0] != 0
    except AssertionError:
        print("Vocabulary cannot be extracted, for no examples were selected.")
        return

    if explain_dir_path:
        # os.makedirs triggers an error for an empty string argument
        os.makedirs(explain_dir_path, exist_ok=True)

    # Sort examples by pred_label and pred_label_proba
    if sort_exples:
        df = df.sort_values(by=[cst.SPEAKER_GENDER_PRED, cst.SPK_GENDER_PRED_PROBA], ascending=False)

    if save_exples:
        examples_file_path = os.path.join(explain_dir_path, "selected_examples.tsv")
        df.to_csv(examples_file_path, sep="\t", index=True)

    pred_lbl_probas = df[cst.SPK_GENDER_PRED_PROBA].tolist()
    pred_labels = df[cst.SPEAKER_GENDER_PRED].tolist()
    gold_labels = df[cst.SPEAKER_GENDER].tolist()

    input_ids_list, _, _ = brt.load_data(df, tokenizer, "test", max_seq_len=max_len, padding="do_not_pad")
    baseline_ids_list = make_baselines(input_ids_list, tokenizer)

    # Number of batches
    n_batches = len(input_ids_list)
    assert n_batches == len(pred_labels)

    # Tracking variables
    vocab = tokenizer.get_vocab()
    data = {
        cst.TOKEN: list(sorted(vocab, key=vocab.get)),
        cst.N_OCCURRENCES: np.zeros(tokenizer.vocab_size, dtype="int"),
        cst.M_ATTR_PLUS: np.zeros(tokenizer.vocab_size),
        cst.F_ATTR_PLUS: np.zeros(tokenizer.vocab_size)
    }
    assert len(data[cst.TOKEN]) == tokenizer.vocab_size

    # Extraction on test set
    print("Extracting labels for test sentences...")
    vis_data_records = list()
    # Measure how long the Extraction takes.
    t0 = time.time()

    # Predict
    for step, batch in enumerate(zip(input_ids_list, baseline_ids_list, gold_labels, pred_labels, pred_lbl_probas,
                                     attributions_list)):
        utl.report_progress(step, n_batches, t0)
        b_input_ids, b_baseline_ids, gold_label, pred_label, pred_lbl_proba, attributions = batch

        # It is possible to run the code for the purpose of generating a new visualization file,
        # without having to spend time computing attribution values again.
        if attributions is None:
            # Add batch to GPU
            b_input_ids = b_input_ids.to(brt.DEVICE)  # (B=1, S)
            b_baseline_ids = b_baseline_ids.to(brt.DEVICE)  # (B=1, S)

            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                attributions, delta = lig.attribute(inputs=b_input_ids, baselines=b_baseline_ids, n_steps=n_steps,
                                                    internal_batch_size=batch_size, return_convergence_delta=True)
                attributions = summarize_attributions(attributions)

            # Move to CPU
            b_input_ids = b_input_ids.detach().cpu()
            attributions = attributions.detach().cpu()
            attributions_list[step] = attributions

        indices = b_input_ids[0].numpy()  # (S,)
        attributions = attributions.numpy()  # (S,)

        # Storing some samples in an array for visualization purposes
        if save_visual:  # TODO: settle for a filter ?
            attr_score = attributions.sum()
            tokens = tokenizer.convert_ids_to_tokens(indices)
            vis_data_record = viz.VisualizationDataRecord(attributions,
                                                          pred_lbl_proba,
                                                          cst.GENDERS[pred_label],
                                                          cst.GENDERS[gold_label],
                                                          ATTR_LABEL,
                                                          attr_score,
                                                          tokens,
                                                          None)
            vis_data_records.append(vis_data_record)

        for indice, attribution in zip(indices, attributions):
            data[cst.N_OCCURRENCES][indice] += 1
            # The sign of attribution is linked to the chosen ATTR_CLASS
            if attribution > 0:
                data[cst.M_ATTR_PLUS][indice] += attribution
            else:
                data[cst.F_ATTR_PLUS][indice] -= attribution

    print("  DONE.")

    if save_visual:
        visual = viz.visualize_text(vis_data_records)
        visual_file_path = os.path.join(explain_dir_path, "visual.html")
        with open(visual_file_path, "w") as file:
            file.write(visual.data)

    if save_attr:
        attr_file_path = os.path.join(explain_dir_path, "attributions.pt")
        torch.save(attributions_list, attr_file_path)

    df = pd.DataFrame(data)
    # Dropping lines relative to tokens that do not appear in the test file
    df = df.loc[df[cst.N_OCCURRENCES] != 0]
    # Computing other values
    df[cst.ATTR_REL] = df[cst.M_ATTR_PLUS] - df[cst.F_ATTR_PLUS]
    df[cst.ATTR_REL_SCORE] = df[cst.ATTR_REL] / df[cst.N_OCCURRENCES]

    if save_vocab:
        vocab_file_path = os.path.join(explain_dir_path, "vocab.tsv")
        df.to_csv(vocab_file_path, sep="\t", index=True, decimal=decimal, columns=cst.EXTRACT_VOC2_COLUMNS)

    if save_hd_tl:
        # For convenience
        for n in 1, 3, 9:
            # Dropping lines relative to tokens that appear less than n times in the test file
            df = df.loc[df[cst.N_OCCURRENCES] >= n]

            df = df.sort_values([cst.ATTR_REL_SCORE], ascending=True)
            attr_df = pd.concat([df.head(25), df.tail(25)])
            attr_vocab_file_path = os.path.join(explain_dir_path, "n%d_attr_vocab.tsv" % n)
            columns = [cst.TOKEN, cst.N_OCCURRENCES, cst.ATTR_REL_SCORE]
            attr_df.to_csv(attr_vocab_file_path, sep="\t", index=True, decimal=decimal, columns=columns)


def lime(model_class, tokenizer_class, model_dir_path, test_file_path, explain_dir_path, decimal=".",
         max_len=brt.MAX_SEQ_LEN, criteria_str=None, n_samples=N_LIME_SAMPLES, alpha=LIME_ALPHA, save_exples=True,
         save_visual=True, save_attr=True, save_vocab=True, save_hd_tl=True, attr_file_path=None, sort_exples=True):
    model, tokenizer = brt.load_model(model_class, tokenizer_class, model_dir_path)
    # Put model in evaluation mode
    model.eval()

    # Inspired by https://captum.ai/tutorials/Image_and_Text_Classification_LIME
    def custom_forward(b_input_ids):
        outputs = model(b_input_ids, token_type_ids=None)
        b_soft_preds = outputs[0].softmax(dim=1)  # (B=1, K=2)
        return b_soft_preds

    # encode text indices into latent representations & calculate cosine similarity
    def cls_cosine_distance(original_inp, perturbed_inp, _, **kwargs):
        # https://discuss.huggingface.co/t/how-to-get-cls-embeddings-from-bertfortokenclassification-model/9276
        outputs = model(original_inp, token_type_ids=None, output_hidden_states=True)
        last_hidden_states = outputs[1][-1]  # (B=1, S, H)
        original_cls = last_hidden_states[0, 0, :]

        outputs = model(perturbed_inp, token_type_ids=None, output_hidden_states=True)
        last_hidden_states = outputs[1][-1]
        perturbed_cls = last_hidden_states[0, 0, :]

        distance = 1 - F.cosine_similarity(original_cls, perturbed_cls, dim=0)
        return torch.exp(-1 * (distance ** 2) / 2)

    # binary vector where each word is selected independently and uniformly at random
    def bernoulli_perturb(original_inp, **kwargs):
        probs = torch.ones_like(original_inp) * 0.5  # (B, S)
        probs[:, 0] = 1  # sparing CLS
        probs[:, -1] = 1  # sparing SEP
        return torch.bernoulli(probs).long()

    lasso_lime = Lime(
        custom_forward,
        interpretable_model=SkLearnLasso(alpha=alpha),
        similarity_func=cls_cosine_distance,
        perturb_func=bernoulli_perturb
    )

    # Load dataset
    df = pd.read_csv(test_file_path, sep="\t")

    # Load attributions if they have already been computed
    if attr_file_path is None:
        attributions_list = df.shape[0] * [None]
    else:
        attributions_list = torch.load(attr_file_path)
        assert len(attributions_list) == df.shape[0]

    # Filter dataset according to criteria string
    if criteria_str is not None:
        df[cst.ATTRIBUTIONS] = attributions_list

        n_examples_init = df.shape[0]
        print("Initial number of examples:")
        print(n_examples_init)
        df = utl.eval_criteria_string(criteria_str, df)
        n_examples_excluded = n_examples_init - df.shape[0]
        print("Number of selected examples:")
        print(df.shape[0], "(%.2f%%)" % (100 * df.shape[0] / n_examples_init))
        print("Number of excluded examples:")
        print(n_examples_excluded, "(%.2f%%)" % (100 * n_examples_excluded / n_examples_init))

        attributions_list = df[cst.ATTRIBUTIONS].tolist()
        df.drop(cst.ATTRIBUTIONS, axis=1)

    try:
        assert df.shape[0] != 0
    except AssertionError:
        print("Vocabulary cannot be extracted, for no examples were selected.")
        return

    if explain_dir_path:
        # os.makedirs triggers an error for an empty string argument
        os.makedirs(explain_dir_path, exist_ok=True)

    # Sort examples by pred_label and pred_label_proba
    if sort_exples:
        df = df.sort_values(by=[cst.SPEAKER_GENDER_PRED, cst.SPK_GENDER_PRED_PROBA], ascending=False)

    if save_exples:
        examples_file_path = os.path.join(explain_dir_path, "selected_examples.tsv")
        df.to_csv(examples_file_path, sep="\t", index=True)

    pred_lbl_probas = df[cst.SPK_GENDER_PRED_PROBA].tolist()
    pred_labels = df[cst.SPEAKER_GENDER_PRED].tolist()
    gold_labels = df[cst.SPEAKER_GENDER].tolist()

    input_ids_list, _, _ = brt.load_data(df, tokenizer, "test", max_seq_len=max_len, padding="do_not_pad")

    # Number of batches
    n_batches = len(input_ids_list)
    assert n_batches == len(pred_labels)

    # Tracking variables
    vocab = tokenizer.get_vocab()
    data = {
        cst.TOKEN: list(sorted(vocab, key=vocab.get)),
        cst.N_OCCURRENCES: np.zeros(tokenizer.vocab_size, dtype="int"),
        cst.M_ATTR_PLUS: np.zeros(tokenizer.vocab_size),
        cst.F_ATTR_PLUS: np.zeros(tokenizer.vocab_size)
    }
    assert len(data[cst.TOKEN]) == tokenizer.vocab_size

    # Extraction on test set
    print("Extracting labels for test sentences...")
    vis_data_records = list()
    # Measure how long the Extraction takes.
    t0 = time.time()

    # Predict
    for step, batch in enumerate(zip(input_ids_list, gold_labels, pred_labels, pred_lbl_probas, attributions_list)):
        utl.report_progress(step, n_batches, t0)
        b_input_ids, gold_label, pred_label, pred_lbl_proba, attributions = batch

        # It is possible to run the code for the purpose of generating a new visualization file,
        # without having to spend time computing attribution values again.
        if attributions is None:
            # Add batch to GPU
            b_input_ids = b_input_ids.to(brt.DEVICE)  # (B=1, S)

            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                attributions = lasso_lime.attribute(
                    b_input_ids,
                    target=ATTR_CLASS,
                    n_samples=n_samples,
                    baselines=tokenizer.mask_token_id,
                    show_progress=True
                ).squeeze(0)

            # Move to CPU
            b_input_ids = b_input_ids.detach().cpu()
            attributions = attributions.detach().cpu()
            attributions_list[step] = attributions

        indices = b_input_ids[0].numpy()  # (S,)
        attributions = attributions.numpy()  # (S,)

        # Storing some samples in an array for visualization purposes
        if save_visual:  # TODO: settle for a filter ?
            attr_score = attributions.sum()
            tokens = tokenizer.convert_ids_to_tokens(indices)
            vis_data_record = viz.VisualizationDataRecord(attributions * 10,
                                                          pred_lbl_proba,
                                                          cst.GENDERS[pred_label],
                                                          cst.GENDERS[gold_label],
                                                          ATTR_LABEL,
                                                          attr_score,
                                                          tokens,
                                                          None)
            vis_data_records.append(vis_data_record)

        for indice, attribution in zip(indices, attributions):
            data[cst.N_OCCURRENCES][indice] += 1
            # The sign of attribution is linked to the chosen ATTR_CLASS
            if attribution > 0:
                data[cst.M_ATTR_PLUS][indice] += attribution
            else:
                data[cst.F_ATTR_PLUS][indice] -= attribution

    print("  DONE.")

    if save_visual:
        visual = viz.visualize_text(vis_data_records)
        visual_file_path = os.path.join(explain_dir_path, "visual.html")
        with open(visual_file_path, "w") as file:
            file.write(visual.data)

    if save_attr:
        attr_file_path = os.path.join(explain_dir_path, "attributions.pt")
        torch.save(attributions_list, attr_file_path)

    df = pd.DataFrame(data)
    # Dropping lines relative to tokens that do not appear in the test file
    df = df.loc[df[cst.N_OCCURRENCES] != 0]
    # Computing other values
    df[cst.ATTR_REL] = df[cst.M_ATTR_PLUS] - df[cst.F_ATTR_PLUS]
    df[cst.ATTR_REL_SCORE] = df[cst.ATTR_REL] / df[cst.N_OCCURRENCES]

    if save_vocab:
        vocab_file_path = os.path.join(explain_dir_path, "vocab.tsv")
        df.to_csv(vocab_file_path, sep="\t", index=True, decimal=decimal, columns=cst.EXTRACT_VOC2_COLUMNS)

    if save_hd_tl:
        # For convenience
        for n in 1, 3, 9:
            # Dropping lines relative to tokens that appear less than n times in the test file
            df = df.loc[df[cst.N_OCCURRENCES] >= n]

            df = df.sort_values([cst.ATTR_REL_SCORE], ascending=True)
            attr_df = pd.concat([df.head(25), df.tail(25)])
            attr_vocab_file_path = os.path.join(explain_dir_path, "n%d_attr_vocab.tsv" % n)
            columns = [cst.TOKEN, cst.N_OCCURRENCES, cst.ATTR_REL_SCORE]
            attr_df.to_csv(attr_vocab_file_path, sep="\t", index=True, decimal=decimal, columns=columns)


def attr_coherence(model_name, tokenizer_class, test_file_path, attr_file_path, stats_file_path=None,
                   decimal=".", hyperparams="", max_len=brt.MAX_SEQ_LEN, criteria_str=None, ci=False,
                   n_bootstraps=cst.N_BOOTSTRAPS, alpha=cst.CI_ALPHA, cond_column=None):
    hp = ", %s" % hyperparams if hyperparams else ""
    model_dir_path = attr_file_path

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=brt.DO_LOWER_CASE)

    # Load dataset
    print("Loading data...")
    df = pd.read_csv(test_file_path, sep="\t")

    # Load attributions
    attributions_list = torch.load(attr_file_path)
    assert len(attributions_list) == df.shape[0]

    # Filter dataset according to criteria string
    if criteria_str is not None:
        df[cst.ATTRIBUTIONS] = attributions_list

        n_examples_init = df.shape[0]
        print("Initial number of examples:")
        print(n_examples_init)
        df = utl.eval_criteria_string(criteria_str, df)
        n_examples_excluded = n_examples_init - df.shape[0]
        print("Number of selected examples:")
        print(df.shape[0], "(%.2f%%)" % (100 * df.shape[0] / n_examples_init))
        print("Number of excluded examples:")
        print(n_examples_excluded, "(%.2f%%)" % (100 * n_examples_excluded / n_examples_init))

        attributions_list = df[cst.ATTRIBUTIONS].tolist()
        df.drop(cst.ATTRIBUTIONS, axis=1)

    input_ids_list, _, _ = brt.load_data(df, tokenizer, "test", max_seq_len=max_len, padding="do_not_pad")

    attr_scores = list()
    for attributions in attributions_list:
        attributions = attributions.numpy()  # (S,)
        attr_score = attributions.sum()
        attr_scores.append(attr_score)

    # Code shared with profile_coherence
    cut.coherence(model_name, model_dir_path, test_file_path, df, attr_scores, hp=hp,
                  stats_file_path=stats_file_path, decimal=decimal, ci=ci, n_bootstraps=n_bootstraps, alpha=alpha,
                  cond_column=cond_column)


# MAIN  ################################################################################################################

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", type=str, choices=["occlusion", "lay_int_grad", "lime", "attr_coherence"],
                        help="explainability method to use")

    parser.add_argument("--model_type", "-modt", choices=list(brt.MODEL_TYPES.keys()),
                        help="type that defines model and tokenizer classes")
    parser.add_argument("--model_name", "-modn",
                        help="path to pre-trained model or shortcut name selected in the list: " +
                             ", ".join(brt.MODEL_NAMES))
    parser.add_argument("--model_dir", "-modd",
                        help="directory where the fine-tuned model is stored")
    parser.add_argument("--test_file", "-tstf",
                        help="tsv file that contains the examples used for evaluation")
    parser.add_argument("--statistics_file", "-staf",
                        help="csv file of gathered statistics")
    parser.add_argument("--explain_dir", "-expd",
                        help="directory where data related to explainability is stored")
    parser.add_argument("--decimal", "-d", default=".",
                        help="character to recognize as decimal point")
    parser.add_argument("--hyperparameters", "-hyp",
                        help="string describing the hyperparameters used when training the model (for tracing purpose)")
    parser.add_argument("--attribution_file", "-attf",
                        help="pytorch file that contains PRE-COMPUTED attributions"
                             "corresponding to the evaluated examples")

    parser.add_argument("--dont_save_exples", "-noex", action="store_true",
                        help="do not save selected examples in explain_dir")
    parser.add_argument("--dont_save_visual", "-novis", action="store_true",
                        help="do not save visualization in explain_dir")
    parser.add_argument("--dont_save_attr", "-noatt", action="store_true",
                        help="do not save attributions in explain_dir")
    parser.add_argument("--dont_save_vocab", "-novoc", action="store_true",
                        help="do not save full vocabulary in explain_dir")
    parser.add_argument("--dont_save_hd_tl", "-nohdtl", action="store_true",
                        help="do not save head-tail vocabulary in explain_dir")
    parser.add_argument("--sort_exples", "-srtex", action="store_true",
                        help="sort selected examples")

    parser.add_argument("--max_seq_length", "-msl", type=int, default=brt.MAX_SEQ_LEN,
                        help="maximum total input sequence length after tokenization"
                             "(sequences longer than this will be truncated, sequences shorter will be padded)")
    parser.add_argument("--n_lig_steps", "-nls", type=int, default=N_LIG_STEPS,
                        help="number of steps used by the lig approximation method (lay_int_grad specifics)")
    parser.add_argument("--lig_batch_size", "-lbs", type=int, default=LIG_BATCH_SIZE,
                        help="internal batch size per GPU/CPU for the lig method (lay_int_grad specifics)")
    parser.add_argument("--n_lime_samples", "-nlimes", type=int, default=N_LIME_SAMPLES,
                        help="number of samples of the original model used by lime"
                             "to train the surrogate interpretable model (lime specifics)")
    parser.add_argument("--lime_alpha", "-limea", type=float, default=LIME_ALPHA,
                        help="fit coefficient for linear lasso interpretable model (lime specifics)")

    parser.add_argument("--criteria_string", "-cs",
                        help="string describing a filter to apply to the examples before processing")
    parser.add_argument("--weight_tokens", "-wt", action="store_true",
                        help="weight examples according to the number of unique tokens (occlusion specifics)")

    parser.add_argument("--confidence_intervals", "-ci", action="store_true",
                        help="compute the confidence intervals for evaluation metrics, using bootstrap resampling.")
    parser.add_argument("--n_bootstraps", "-nb", type=int, default=cst.N_BOOTSTRAPS,
                        help="number of bootstrap sets to be created")
    parser.add_argument("--ci_alpha", "-cia", type=float, default=cst.CI_ALPHA,
                        help="level of the interval"
                             "(the confidence interval will be computed between alpha/2 and 100-alpha/2 percentiles)")
    parser.add_argument("--cond_column", "-cc", default=None,
                        help="name of the column that contains the conditions of the samples")

    args = parser.parse_args()
    return args


def main(args):
    mode = args.mode

    model_type = args.model_type
    model_class, tokenizer_class = brt.MODEL_TYPES[model_type]
    model_name = args.model_name
    model_dir_path = args.model_dir
    test_file_path = args.test_file
    stats_file_path = args.statistics_file
    explain_dir_path = args.explain_dir
    decimal = args.decimal
    hyperparams = args.hyperparameters
    attr_file_path = args.attribution_file

    save_exples = not args.dont_save_exples
    save_visual = not args.dont_save_visual
    save_attr = not args.dont_save_attr
    save_vocab = not args.dont_save_vocab
    save_hd_tl = not args.dont_save_hd_tl
    sort_exples = args.sort_exples

    max_len = args.max_seq_length
    n_lig_steps = args.n_lig_steps
    lig_batch_size = args.lig_batch_size
    n_lime_samples = args.n_lime_samples
    lime_alpha = args.lime_alpha

    criteria_str = args.criteria_string
    weight_tokens = args.weight_tokens

    ci = args.confidence_intervals
    n_bootstraps = args.n_bootstraps
    ci_alpha = args.ci_alpha
    cond_column = args.cond_column

    if mode == "occlusion":
        occlusion(model_class, tokenizer_class, model_dir_path, test_file_path, explain_dir_path, decimal=decimal,
                  max_len=max_len, criteria_str=criteria_str, weight_tokens=weight_tokens, save_exples=save_exples,
                  save_visual=save_visual, save_attr=save_attr, save_vocab=save_vocab, save_hd_tl=save_hd_tl,
                  attr_file_path=attr_file_path, sort_exples=sort_exples)

    elif mode == "lay_int_grad":
        lay_int_grad(model_class, tokenizer_class, model_dir_path, test_file_path, explain_dir_path, decimal=decimal,
                     max_len=max_len, criteria_str=criteria_str, n_steps=n_lig_steps, batch_size=lig_batch_size,
                     save_exples=save_exples, save_visual=save_visual, save_attr=save_attr, save_vocab=save_vocab,
                     save_hd_tl=save_hd_tl, attr_file_path=attr_file_path, sort_exples=sort_exples)

    elif mode == "lime":
        lime(model_class, tokenizer_class, model_dir_path, test_file_path, explain_dir_path, decimal=decimal,
             max_len=max_len, criteria_str=criteria_str, n_samples=n_lime_samples, alpha=lime_alpha,
             save_exples=save_exples, save_visual=save_visual, save_attr=save_attr, save_vocab=save_vocab,
             save_hd_tl=save_hd_tl, attr_file_path=attr_file_path, sort_exples=sort_exples)

    elif mode == "attr_coherence":
        attr_coherence(model_name, tokenizer_class, test_file_path, attr_file_path, stats_file_path=stats_file_path,
                       decimal=decimal, hyperparams=hyperparams, max_len=max_len, criteria_str=criteria_str, ci=ci,
                       n_bootstraps=n_bootstraps, alpha=ci_alpha, cond_column=cond_column)


if __name__ == "__main__":
    main(parse_args())
