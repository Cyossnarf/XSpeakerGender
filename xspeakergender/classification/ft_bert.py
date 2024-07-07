# coding: utf-8

import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import \
    BertForSequenceClassification, BertTokenizer, \
    CamembertForSequenceClassification, CamembertTokenizer, \
    FlaubertForSequenceClassification, FlaubertTokenizer, \
    BertJapaneseTokenizer, \
    get_linear_schedule_with_warmup

# We include the path of the toplevel package in the system path,
# so we can always use absolute imports within the package.
toplevel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if toplevel_path not in sys.path:
    sys.path.insert(1, toplevel_path)

import classification.metrics as mtr
import utils.constants as cst
import utils.util as utl


DO_LOWER_CASE = True
MODEL_TYPES = {
    "bert": (BertForSequenceClassification, BertTokenizer),
    "camembert": (CamembertForSequenceClassification, CamembertTokenizer),
    "flaubert": (FlaubertForSequenceClassification, FlaubertTokenizer),
    "bert-japanese": (BertForSequenceClassification, BertJapaneseTokenizer)
}
MODEL_NAMES = [
    "bert-base-multilingual-uncased",
    "camembert-base",
    "camembert/camembert-base-ccnet",
    "flaubert/flaubert_base_uncased",
    "nherve/flaubert-oral-asr",
    "nherve/flaubert-oral-asr_nb",
    "nherve/flaubert-oral-mixed",
    "nherve/flaubert-oral-ft",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-large-japanese",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-large-japanese-char-v2"
]

# Default learning rate is 2e-5, as in Chris McCormick's notebook
LEARNING_RATE = 2e-5
WEIGHT_DECAY_RATE = 0.0
ADAM_EPSILON = 1e-8
# Default warmup_steps value in run_glue.py
N_WARMUP_STEPS = 0
# The BERT authors recommend between 2 and 4 training epochs.
N_EPOCHS = 4
# For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32.
BATCH_SIZE = 32
MAX_SEQ_LEN = 64
PATIENCE = 3
MIN_DELTA = 0.0

# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    DEVICE = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    DEVICE = torch.device("cpu")


def flat_accuracy(preds, labels):
    """
    Function to calculate the accuracy of our predictions vs labels
    :param preds: predictions
    :param labels: labels
    :return: accuracy
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def load_data(df_or_file_path, tokenizer, name, max_seq_len=MAX_SEQ_LEN, gold_column=cst.SPEAKER_GENDER,
              gold_pos_value=cst.FEMALE, padding="max_length"):
    if isinstance(df_or_file_path, str):
        print("Loading", name, "data...")
        df = pd.read_csv(df_or_file_path, sep="\t")
    else:
        df = df_or_file_path
    print("Number of", name, "examples:")
    print(df.shape[0])
    sentences = df[cst.SENTENCE].tolist()
    labels = (df[gold_column] == gold_pos_value).astype(int).tolist()

    sentence = random.choice(sentences)
    print(" Original: ", sentence)
    print("Tokenized: ", tokenizer.tokenize(sentence))
    print("Token IDs: ", tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence)))

    # Tokenize all the sentences and map the tokens to their word IDs.
    input_ids = list()
    attention_masks = list()
    for sentence in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sentence,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'.
            truncation=True,  # Truncate to a maximum length specified with the argument max_length.
            max_length=max_seq_len,  # Controls the maximum length to use by one of the truncation/padding parameters.
            padding=padding,  # Pad to a maximum length specified with the argument max_length.
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors="pt",  # Return pytorch tensors.
        )
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict["input_ids"])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict["attention_mask"])

    return input_ids, attention_masks, labels


def make_dataset(input_ids, attention_masks, labels):
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)  # attention_masks: 0 - padding, 1 - no padding
    labels = torch.tensor(labels)
    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    return dataset


def load_dataset(df_or_file_path, tokenizer, name, max_seq_len=MAX_SEQ_LEN, gold_column=cst.SPEAKER_GENDER,
                 gold_pos_value=cst.FEMALE):
    input_ids, attention_masks, labels = load_data(df_or_file_path, tokenizer, name, max_seq_len=max_seq_len,
                                                   gold_column=gold_column, gold_pos_value=gold_pos_value,
                                                   padding="max_length")
    dataset = make_dataset(input_ids, attention_masks, labels)

    return dataset


def save_model(model, tokenizer, model_dir_path):
    print("Saving model to %s" % model_dir_path)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(model_dir_path)
    tokenizer.save_pretrained(model_dir_path)


def load_model(model_class, tokenizer_class, model_dir_path):
    # Load a trained model and vocabulary that you have fine-tuned
    print("Loading model...")
    model = model_class.from_pretrained(model_dir_path)
    print("Loading tokenizer...")
    tokenizer = tokenizer_class.from_pretrained(model_dir_path)
    # Copy the model to the GPU.
    model.to(DEVICE)
    return model, tokenizer


def format_and_save_output(test_file_path, pred_labels, gold_lbl_probas, pred_lbl_probas, output_file_path=None,
                           pred_column=cst.SPEAKER_GENDER_PRED, gold_proba_column=cst.SPK_GENDER_PROBA,
                           pred_proba_column=cst.SPK_GENDER_PRED_PROBA, columns=cst.OUTPUT_COLUMNS):
    df = pd.read_csv(test_file_path, sep="\t")
    df[pred_column] = pred_labels
    df[gold_proba_column] = gold_lbl_probas
    df[pred_proba_column] = pred_lbl_probas
    df = df[columns]

    if output_file_path is not None:
        dir_path = os.path.dirname(output_file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        df.to_csv(output_file_path, sep="\t", index=False)

    return df


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# FUNCTIONS ############################################################################################################

def seq_lengths(model_name, tokenizer_class, train_file_path):
    print("Loading tokenizer...")
    tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=DO_LOWER_CASE)

    print("Loading train dataset...")
    df = pd.read_csv(train_file_path, sep="\t")
    print("Number of train examples:")
    print(df.shape[0])
    sentences = df[cst.SENTENCE].tolist()
    lengths = list()

    sentence = random.choice(sentences)
    print(" Original: ", sentence)
    print("Tokenized: ", tokenizer.tokenize(sentence))
    print("Token IDs: ", tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence)))

    for sentence in sentences:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sentence, add_special_tokens=True)
        lengths.append(len(input_ids))

    lengths = np.array(lengths)
    print()
    print("Seq. length max.:", np.max(lengths))
    print("Seq. length mean:", np.mean(lengths))
    print("Seq. length std.:", np.std(lengths))


def finetune(model_name, model_class, tokenizer_class, model_dir_path, train_file_path, val_file_path,
             stats_file_path=None, learning_rate=LEARNING_RATE, weight_decay_rate=WEIGHT_DECAY_RATE,
             adam_epsilon=ADAM_EPSILON, n_warmup_steps=N_WARMUP_STEPS, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
             max_seq_len=MAX_SEQ_LEN, seed=cst.SEED, patience=PATIENCE, min_delta=MIN_DELTA):
    # Set the seed value all over the place to make this reproducible.
    set_seed(seed)
    # Create output directory (if it does not already exist).
    os.makedirs(model_dir_path, exist_ok=True)

    # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top.
    print("Loading model...")
    model = model_class.from_pretrained(
        model_name,
        num_labels=2,  # The number of output labels--2 for binary classification.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
        return_dict=False  # todo: use dict, https://huggingface.co/docs/transformers/migration paragraph 4
    )
    # Copy the model to the GPU.
    model.to(DEVICE)
    print("Loading tokenizer...")
    tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=DO_LOWER_CASE)

    # Load datasets.
    train_dataset = load_dataset(train_file_path, tokenizer, "train", max_seq_len=max_seq_len)
    val_dataset = load_dataset(val_file_path, tokenizer, "val", max_seq_len=max_seq_len)

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

    # This code is taken from:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L102

    # Don't apply weight decay to any parameters whose names include these tokens.
    # (Here, the BERT doesn't have `gamma` or `beta` parameters, only `bias` terms)
    no_decay = ['bias', 'LayerNorm.weight']
    # Separate the `weight` parameters from the `bias` parameters.
    # - For the `weight` parameters, this specifies a 'weight_decay_rate'.
    # - For the `bias` parameters, the 'weight_decay_rate' is 0.0.
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': weight_decay_rate},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    # Note: `optimizer_grouped_parameters` only includes the parameter values, not
    # the names.
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix'
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    # Number of batches (also number of steps per epoch)
    n_batches = len(train_dataloader)
    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    n_training_steps = n_batches * n_epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=n_warmup_steps,
                                                num_training_steps=n_training_steps)

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []
    # Measure the total training time for the whole run.
    total_t0 = time.time()
    # Use patience to stop the training.
    wait = 0
    best_val_loss = -1

    # For each epoch...
    for epoch_i in range(n_epochs):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.

        print()
        print("======== Epoch {:} / {:} ========".format(epoch_i + 1, n_epochs))
        print("Training...")

        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_train_loss = 0
        # Put the model into training mode. Don't be mislead--the call to `train` just changes the *mode*,
        # it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training vs. test
        # (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            utl.report_progress(step, n_batches, t0)

            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(DEVICE)
            b_input_mask = batch[1].to(DEVICE)
            b_labels = batch[2].to(DEVICE)

            # Always clear any previously calculated gradients before performing a backward pass.
            # PyTorch doesn't do this automatically because accumulating the gradients is
            # "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()
            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments are given and what flags are set.
            # For our usage here, it returns the loss (because we provided labels) and the "logits"--
            # the model outputs prior to activation.
            loss, b_logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            # Accumulate the training loss over all the batches so that we can calculate the average loss at the end.
            # `loss` is a Tensor containing a single value;
            # the `.item()` function just returns the Python value from the tensor.
            total_train_loss += loss.item()
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are modified based on their gradients,
            # the learning rate, etc.
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)
        # Measure how long this epoch took.
        training_time = utl.format_time(time.time() - t0)
        print()
        print("  Average training loss: %.4f" % avg_train_loss)
        print("  Training epoch took: %s" % training_time)

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on our validation set.

        print()
        print("Running Validation...")

        t0 = time.time()
        # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
        model.eval()
        # Tracking variables
        total_val_accuracy = 0
        total_val_loss = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(DEVICE)
            b_input_mask = batch[1].to(DEVICE)
            b_labels = batch[2].to(DEVICE)

            # Tell pytorch not to bother with constructing the computation graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids",
                # which differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                loss, b_logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            # Accumulate the validation loss.
            total_val_loss += loss.item()
            # Move logits and labels to CPU
            b_logits = b_logits.detach().cpu().numpy()
            b_labels = b_labels.to("cpu").numpy()
            # Calculate the accuracy for this batch of val sentences, and accumulate it over all batches.
            total_val_accuracy += flat_accuracy(b_logits, b_labels)

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_val_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        # Calculate the average loss over all the batches.
        avg_val_loss = total_val_loss / len(validation_dataloader)
        # Measure how long the validation run took.
        validation_time = utl.format_time(time.time() - t0)
        print("  Validation Loss: %.4f" % avg_val_loss)
        print("  Validation took: %s" % validation_time)

        # Record all statistics from this epoch.
        training_stats.append({
            "epoch": epoch_i + 1,
            "Training Loss": avg_train_loss,
            "Valid. Loss": avg_val_loss,
            "Valid. Accur.": avg_val_accuracy,
            "Training Time": training_time,
            "Validation Time": validation_time
        })

        if avg_val_loss < best_val_loss - min_delta or best_val_loss == -1:
            wait = 0
            best_val_loss = avg_val_loss
            # Save best model.
            save_model(model, tokenizer, model_dir_path)
            print()
        else:
            wait += 1
            print("  Best Validation Loss: %.4f" % best_val_loss)
            print("  No Validation Loss improvement for %d epoch(s)" % wait)
            if wait >= patience:
                print("  Early stopping at epoch %d" % (epoch_i + 1))
                break

    print()
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(utl.format_time(time.time() - total_t0)))

    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)
    if stats_file_path is not None:
        df_stats.to_csv(stats_file_path, index=False)
    # Display the table.
    df_stats = df_stats.set_index("epoch")
    pd.options.display.precision = 2
    print(df_stats)


def evaluate(model_class, tokenizer_class, model_dir_path, test_file_path, model_name=None, output_file_path=None,
             stats_file_path=None, decimal=".", batch_size=BATCH_SIZE, max_len=MAX_SEQ_LEN,
             gold_column=cst.SPEAKER_GENDER, gold_pos_value=cst.FEMALE, hyperparams="", ci=False,
             n_bootstraps=cst.N_BOOTSTRAPS, alpha=cst.CI_ALPHA, cond_column=None):
    model, tokenizer = load_model(model_class, tokenizer_class, model_dir_path)
    # Put model in evaluation mode
    model.eval()

    # Load dataset.
    test_dataset = load_dataset(test_file_path, tokenizer, "test", max_seq_len=max_len, gold_column=gold_column,
                                gold_pos_value=gold_pos_value)

    # Create the DataLoader for our test set.
    # For test the order doesn't matter, so we'll just read them sequentially.
    prediction_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)
    # Number of batches
    n_batches = len(prediction_dataloader)

    # Prediction on test set
    print("Predicting labels for test sentences...")
    # Measure how long the prediction takes.
    t0 = time.time()
    # Tracking variables
    soft_preds, gold_labels = list(), list()

    # Predict
    for step, batch in enumerate(prediction_dataloader):
        utl.report_progress(step, n_batches, t0)

        # Add batch to GPU
        batch = tuple(t.to(DEVICE) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch  # (B, S=max_len), (B, S=max_len), (B,)

        # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        b_logits = outputs[0]
        b_soft_preds = b_logits.softmax(dim=1)
        # Move soft predictions and labels to CPU
        b_soft_preds = b_soft_preds.detach().cpu().numpy()
        b_labels = b_labels.to('cpu').numpy()
        # Store predictions and true labels
        soft_preds.append(b_soft_preds)
        gold_labels.append(b_labels)

    print("  DONE.")

    # Combine the results across all batches.
    soft_preds = np.concatenate(soft_preds, axis=0)  # (test_size, K=2)
    # For each sample, pick the label (0 or 1) with the higher score.
    pred_labels = np.argmax(soft_preds, axis=1)  # (test_size,)
    # Combine the correct labels for each batch into a single list.
    gold_labels = np.concatenate(gold_labels, axis=0)  # (test_size,)
    # Extract probas associated with reference and prediction labels
    gold_lbl_probas = np.take_along_axis(soft_preds, np.expand_dims(gold_labels, axis=1), axis=1).flatten()
    pred_lbl_probas = np.take_along_axis(soft_preds, np.expand_dims(pred_labels, axis=1), axis=1).flatten()

    if gold_column == cst.SPEAKER_GENDER:
        pred_column = cst.SPEAKER_GENDER_PRED
        gold_proba_column = cst.SPK_GENDER_PROBA
        pred_proba_column = cst.SPK_GENDER_PRED_PROBA
        columns = cst.OUTPUT_COLUMNS
        # Classification label values - Male: 0, Female: 1 -> Dataset label values - Male: 1, Female: 2
        pred_labels += 1
    else:
        pred_column = cst.get_pred_column(gold_column, gold_pos_value)
        gold_proba_column = cst.get_gold_proba_column(gold_column, gold_pos_value)
        pred_proba_column = cst.get_pred_proba_column(gold_column, gold_pos_value)
        columns = [cst.SENTENCE_SRC, gold_column, pred_column, gold_proba_column, pred_proba_column, cst.SENTENCE]

    df = format_and_save_output(test_file_path, pred_labels, gold_lbl_probas, pred_lbl_probas,
                                output_file_path=output_file_path, pred_column=pred_column,
                                gold_proba_column=gold_proba_column, pred_proba_column=pred_proba_column,
                                columns=columns)

    score(df, model_name=model_name, model_dir_path=model_dir_path, test_file_path=test_file_path,
          stats_file_path=stats_file_path, decimal=decimal, gold_column=gold_column, gold_pos_value=gold_pos_value,
          hyperparams=hyperparams, ci=ci, n_bootstraps=n_bootstraps, alpha=alpha, cond_column=cond_column)


def score(output_df_or_file_path, model_name=None, model_dir_path=None, test_file_path=None, stats_file_path=None,
          decimal=".", gold_column=cst.SPEAKER_GENDER, gold_pos_value=cst.FEMALE, pred_column=cst.SPEAKER_GENDER_PRED,
          hyperparams="", ci=False, n_bootstraps=cst.N_BOOTSTRAPS, alpha=cst.CI_ALPHA, cond_column=None):
    if isinstance(output_df_or_file_path, str):
        print("Loading outputs...")
        df = pd.read_csv(output_df_or_file_path, sep="\t")
        print("Number of examples:")
        print(df.shape[0])
    else:
        df = output_df_or_file_path

    if gold_column == cst.SPEAKER_GENDER:
        # Dataset label values - Male: 1, Female: 2 -> Classification label values - Male: 0, Female: 1
        df[pred_column] -= 1
        str_pos = cst.FEM
        str_neg = cst.MAL
    else:
        pred_column = cst.get_pred_column(gold_column, gold_pos_value)
        str_pos = cst.POS
        str_neg = cst.NEG

    gold_labels = (df[gold_column] == gold_pos_value).to_numpy()
    pred_labels = df[pred_column].to_numpy()
    conditions = None if cond_column is None else df[cond_column].to_numpy()

    mtr.binary_classification_scores(gold_labels, pred_labels, stats_file_path=stats_file_path, model_name=model_name,
                                     model_dir_path=model_dir_path, test_file_path=test_file_path, decimal=decimal,
                                     str_pos=str_pos, str_neg=str_neg, hyperparams=hyperparams, ci=ci,
                                     n_bootstraps=n_bootstraps, alpha=alpha, conditions=conditions)


# MAIN  ################################################################################################################

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", type=str, choices=["seq_lengths", "finetune", "eval", "score"])

    parser.add_argument("--model_type", "-modt", choices=list(MODEL_TYPES.keys()),
                        help="type that defines model and tokenizer classes")
    parser.add_argument("--model_name", "-modn",
                        help="path to pre-trained model or shortcut name selected in the list: " +
                             ", ".join(MODEL_NAMES))
    parser.add_argument("--model_dir", "-modd",
                        help="directory where the fine-tuned model is stored")
    parser.add_argument("--output_file", "-outf",
                        help="tsv file where the model predictions are written")
    parser.add_argument("--train_file", "-trnf",
                        help="tsv file that contains the training examples")
    parser.add_argument("--val_file", "-valf",
                        help="tsv file that contains the examples used to compute validation loss")
    parser.add_argument("--test_file", "-tstf",
                        help="tsv file that contains the examples used for evaluation")
    parser.add_argument("--statistics_file", "-staf",
                        help="csv file of gathered statistics")
    parser.add_argument("--decimal", "-d", default=".",
                        help="character to recognize as decimal point")
    parser.add_argument("--hyperparameters", "-hyp",
                        help="string describing the hyperparameters used when training the model (for tracing purpose)")

    parser.add_argument("--learning_rate", "-lr", type=float, default=LEARNING_RATE,
                        help="initial learning rate for Adam")
    parser.add_argument("--weight_decay_rate", "-wdr", type=float, default=WEIGHT_DECAY_RATE,
                        help="weight decay rate if we apply some")
    parser.add_argument("--adam_epsilon", "-ae", type=float, default=ADAM_EPSILON,
                        help="epsilon for Adam optimizer")
    parser.add_argument("--n_warmup_steps", "-nws", type=int, default=N_WARMUP_STEPS,
                        help="linear warmup over n_warmup_steps")
    parser.add_argument("--n_epochs", "-ne", type=int, default=N_EPOCHS,
                        help="maximum number of training epochs to perform")
    parser.add_argument("--batch_size", "-bs", type=int, default=BATCH_SIZE,
                        help="batch size per GPU/CPU")
    parser.add_argument("--max_seq_length", "-msl", type=int, default=MAX_SEQ_LEN,
                        help="maximum total input sequence length after tokenization"
                             "(sequences longer than this will be truncated, sequences shorter will be padded)")
    parser.add_argument("--seed", "-s", type=int, default=cst.SEED,
                        help="random seed for initialization")
    parser.add_argument("--patience", "-p", type=int, default=PATIENCE,
                        help="maximum number of epochs without validation loss improvement")
    parser.add_argument("--min_delta", "-md", type=float, default=MIN_DELTA,
                        help="minimum delta in validation loss to be considered an improvement")

    parser.add_argument("--gold_column", "-gc", default=cst.SPEAKER_GENDER,
                        help="name of the column that contains the gold values")
    parser.add_argument("--gold_pos_value", "-gv", type=int, default=cst.FEMALE,
                        help="value in gold_column to be considered as 'positive' for classification")  # todo: convert type according to -gc
    parser.add_argument("--pred_column", "-pc", default=cst.SPEAKER_GENDER_PRED,
                        help="name of the column that contains the predicted values")
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
    if model_type is not None:
        model_class, tokenizer_class = MODEL_TYPES[model_type]
    else:
        model_class, tokenizer_class = None, None
    model_name = args.model_name
    model_dir_path = args.model_dir
    output_file_path = args.output_file
    train_file_path = args.train_file
    val_file_path = args.val_file
    test_file_path = args.test_file
    stats_file_path = args.statistics_file
    decimal = args.decimal
    hyperparams = args.hyperparameters

    learning_rate = args.learning_rate
    weight_decay_rate = args.weight_decay_rate
    adam_epsilon = args.adam_epsilon
    n_warmup_steps = args.n_warmup_steps
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    max_seq_len = args.max_seq_length
    seed = args.seed
    patience = args.patience
    min_delta = args.min_delta

    gold_column = args.gold_column
    gold_pos_value = args.gold_pos_value
    pred_column = args.pred_column
    ci = args.confidence_intervals
    n_bootstraps = args.n_bootstraps
    ci_alpha = args.ci_alpha
    cond_column = args.cond_column

    if mode == "seq_lengths":
        seq_lengths(model_name, tokenizer_class, train_file_path)

    elif mode == "finetune":
        finetune(model_name, model_class, tokenizer_class, model_dir_path, train_file_path, val_file_path,
                 stats_file_path=stats_file_path, learning_rate=learning_rate, weight_decay_rate=weight_decay_rate,
                 adam_epsilon=adam_epsilon, n_warmup_steps=n_warmup_steps, n_epochs=n_epochs, batch_size=batch_size,
                 max_seq_len=max_seq_len, seed=seed, patience=patience, min_delta=min_delta)

    elif mode == "eval":
        evaluate(model_class, tokenizer_class, model_dir_path, test_file_path, model_name=model_name,
                 output_file_path=output_file_path, stats_file_path=stats_file_path,
                 decimal=decimal, batch_size=batch_size, max_len=max_seq_len, gold_column=gold_column,
                 gold_pos_value=gold_pos_value, hyperparams=hyperparams, ci=ci, n_bootstraps=n_bootstraps,
                 alpha=ci_alpha, cond_column=cond_column)

    elif mode == "score":
        score(output_file_path, model_name=model_name, model_dir_path=model_dir_path, test_file_path=test_file_path,
              stats_file_path=stats_file_path, decimal=decimal, gold_column=gold_column, gold_pos_value=gold_pos_value,
              pred_column=pred_column, hyperparams=hyperparams, ci=ci, n_bootstraps=n_bootstraps, alpha=ci_alpha,
              cond_column=cond_column)


if __name__ == "__main__":
    main(parse_args())
