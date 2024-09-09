# XSpeakerGender

## Requirements (tested version)

* numpy (1.24.4)
* pandas (1.5.2)
* pytorch (2.0.0)
* transformers (4.27.3)
* scikit-learn (1.4.1.post1)
* sentencepiece (0.2.0) (if using CamemBERT)
* fugashi (1.3.2) (if using bert-japanese)
* ipadic (1.0.0) (if using bert-japanese)
* captum (0.7.0)
* ipython (8.15.0)
* confidence-intervals (0.0.3)

## Gender classification and attributions

### Fine-tune a BERT classifier

**Command:**

```
python xspeakergender/classification/ft_bert.py finetune -modt <model_type> -modn <model_name> -modd <model_dir> -trnf <train_file> -valf <val_file> [OPTIONS]
```

**Parameters:**

* `--model_type`, `-modt`: Type that defines model and tokenizer classes (e.g. "bert", "camembert", "flaubert").
* `--model_name`, `-modn`: Path to pre-trained model or shortcut name (e.g. "bert-base-multilingual-uncased", "camembert-base", "nherve/flaubert-oral-asr_nb").
* `--model_dir`, `-modd`: Directory where the fine-tuned model is stored.
* `--train_file`, `-trnf`: Tsv file that contains the training examples.
* `--val_file`, `-valf`: Tsv file that contains the examples used to compute validation loss.

**Options:**

* `--statistics_file`, `-staf`: Csv file where are written statistics on the finetuning process (default: None).
* `--learning_rate`, `-lr`: Initial learning rate for Adam (default: 2e-5).
* `--weight_decay_rate`, `-wdr`: Weight decay rate (default: 0.0).
* `--adam_epsilon`, `-ae`: Epsilon for Adam optimizer (default: 1e-8).
* `--n_warmup_steps`, `-nws`: Number of steps for linear warmup (default: 0).
* `--n_epochs`, `-ne`: Maximum number of training epochs to perform (default: 4).
* `--batch_size`, `-bs`: Batch size per GPU/CPU (default: 32).
* `--max_seq_length`, `-msl`: Maximum total input sequence length after tokenization (sequences longer than this will be truncated, sequences shorter will be padded; default: 64).
* `--seed`, `-s`: Random seed for initialization (default: 42).
* `--patience`, `-p`: Maximum number of epochs without validation loss improvement (default: 3).
* `--min_delta`, `-md`: Minimum delta in validation loss to be considered an improvement (default: 0.0).

**Example:**

```
python xspeakergender/classification/ft_bert.py finetune -modt flaubert -modn nherve/flaubert-oral-asr_nb -modd flaubert-o_0 -trnf data/train.tsv -valf data/val.tsv -staf flaubert-o_0/training.csv -ne 10 -msl 128 -lr 5e-5
```

### Evaluate a BERT classifier

**Command:**

```
python xspeakergender/classification/ft_bert.py eval -modt <model_type> -modd <model_dir> -tstf <test_file> [OPTIONS]
```

**Parameters:**

* `--model_type`, `-modt`: Type that defines model and tokenizer classes (e.g. "bert", "camembert", "flaubert").
* `--model_dir`, `-modd`: Directory where the fine-tuned model is stored.
* `--test_file`, `-tstf`: Tsv file that contains the examples used for evaluation.

**Options:**

* `--model_name`, `-modn`: Path to pre-trained model or shortcut name (for tracing purpose; default: None).
* `--output_file`, `-outf`: Tsv file where the model predictions are written (default: None).
* `--statistics_file`, `-staf`: Csv file where are written the evaluation scores (default: None).
* `--decimal`, `-d`: Character to recognize as decimal point in `statistics_file` (default: ".").
* `--batch_size`, `-bs`: Batch size per GPU/CPU (default: 32).
* `--max_seq_length`, `-msl`: Maximum total input sequence length after tokenization (sequences longer than this will be truncated, sequences shorter will be padded; default: 64).
* `--gold_column`, `-gc`: Name of the column in `test_file` that contains the gold values (default: "speaker_gender").
* `--gold_pos_value`, `-gv`: Value in `gold_column` to be considered as 'positive' for classification (default: 2).
* `--hyperparameters`, `-hyp`: String describing the hyperparameters used when training the model (for tracing purpose; default: None).

**Example:**

```
python xspeakergender/classification/ft_bert.py eval -modt flaubert -modn nherve/flaubert-oral-asr_nb -modd flaubert-o_0 -tstf data/test.tsv -outf flaubert-o_0/out.tsv -staf data/results.csv -d "," -msl 128
```

### Compute attributions with XAI

**Command:**

```
python xspeakergender/classification/explainability.py {occlusion,lay_int_grad} -modt <model_type> -modd <model_dir> -tstf <test_file> -expd <explain_dir> [OPTIONS]
```

**Parameters:**

* `{occlusion,lay_int_grad,lime}`: Explainability method to use.
* `--model_type`, `-modt`: Type that defines model and tokenizer classes (e.g. "bert", "camembert", "flaubert").
* `--model_dir`, `-modd`: Directory where the fine-tuned model is stored.
* `--test_file`, `-tstf`: Tsv file that contains the examples to explain AND the model predictions (the model must have been evaluated on these examples beforehand).
* `--explain_dir`, `-expd`: Directory where data related to explainability is stored.

**Options:**

* `--decimal`, `-d`: Character to recognize as decimal point in the output files (default: ".").
* `--dont_save_exples`, `-noex`: Do not save selected examples in `explain_dir`.
* `--dont_save_visual`, `-novis`: Do not save visualization in `explain_dir`.
* `--dont_save_attr`, `-noatt`: Do not save attributions in `explain_dir`.
* `--dont_save_vocab`, `-novoc`: Do not save full vocabulary in `explain_dir`.
* `--dont_save_hd_tl`, `-nohdtl`: Do not save head-tail vocabulary in `explain_dir`.
* `--sort_exples`, `-srtex`: Sort selected examples.
* `--attribution_file`, `-attf`: Pytorch file that contains PRE-COMPUTED attributions corresponding to the evaluated examples (does not work with `occlusion` for now).
* `--max_seq_length`, `-msl`: Maximum total input sequence length after tokenization (sequences longer than this will be truncated, sequences shorter will be padded; default: 64).
* `--criteria_string`, `-cs`: String describing a filter to apply to the examples before processing (e.g. drop examples for which the model confidence is lower than a threshold; default: None).
* `--weight_tokens`, `-wt`: Weight examples according to the number of unique tokens (`occlusion` specifics).
* `--n_lig_steps`, `-nls`: Number of steps used by the lig approximation method (`lay_int_grad` specifics; default: 600).
* `--lig_batch_size`, `-lbs`: Internal batch size per GPU/CPU for the lig method (`lay_int_grad` specifics; default: 5).
* `--n_lime_samples`, `-nlimes`: Number of samples of the original model used by lime to train the surrogate interpretable model (`lime` specifics; default: 200).
* `--lime_alpha`, `-limea`: Fit coefficient for linear lasso interpretable model (`lime` specifics; default: 0.001).

**Examples:**

Occlusion method:

```
python xspeakergender/classification/explainability.py occlusion -modt flaubert -modd flaubert-o_0 -tstf flaubert-o_0/out.tsv -expd flaubert-o_0/explain/occl -d "," -msl 128 -wt -cs "[spk_gender_pred_proba] > 0.6" -srtex
```

Layer Integrated Gradients method:

```
python xspeakergender/classification/explainability.py lay_int_grad -modt flaubert -modd flaubert-o_0 -tstf flaubert-o_0/out.tsv -expd flaubert-o_0/explain/lig -d "," -msl 128 -nls 100 -lbs 5 -cs "[spk_gender_pred_proba] > 0.6" -srtex
```

Layer Integrated Gradients method, re-using the attributions previously computed:

```
python xspeakergender/classification/explainability.py lay_int_grad -modt flaubert -modd flaubert-o_0 -tstf flaubert-o_0/explain/lig/selected_examples.tsv -attf flaubert-o_0/explain/lig/attributions.pt -expd flaubert-o_0/explain/lig_bis  -d "," -msl 128 -nls 100 -lbs 5 -cs "([spk_gender_pred_proba] > 0.75) & ([speaker_gender_pred] == [speaker_gender])" -srtex -noatt -novoc -nohdtl
```

Lime method:

```
python xspeakergender/classification/explainability.py lime -modt flaubert -modd flaubert-o_0 -tstf flaubert-o_0/out.tsv -expd flaubert-o_0/explain/lime -d "," -msl 128 -nlimes 500 -limea 0.001 -cs "[spk_gender_pred_proba] > 0.6" -srtex
```

