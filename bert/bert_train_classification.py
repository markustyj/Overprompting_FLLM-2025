### functions that are used to support few-shot prompting

import json
import logging
import warnings
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments
from transformers import EvalPrediction
from transformers.trainer_utils import IntervalStrategy


from sklearn.exceptions import UndefinedMetricWarning

# import os
# os.environ['WANDB_DISABLED'] = 'true'

# !pip install datasets
import datasets
from datasets import load_dataset
from datasets import DatasetDict

# !pip install evaluate
from evaluate import load





# pylint: disable=too-many-arguments, too-many-locals

logger = logging.getLogger(__name__)


def transformer_finetune_textcat_downstream(
    train_file: str,
    val_file: str,
    test_file: str,
    base_model: str,
    output_model: str,
    training_data_fraction: float,
    logging_dir: str,
    tokenizer_model: str,
    evaluation_strategy: Union[str, IntervalStrategy] = "epoch",
    save_strategy: Union[str, IntervalStrategy] = "epoch",
    num_train_epochs: int = 2,
    save_total_limit: int = 2,
    train_batch_size: int = 4,
    eval_batch_size: int = 4,
    evaluation_metrics_output: str = None,
    load_best_model_at_end: bool = False,
    metric_for_best_model: str = "loss",
    greater_is_better: bool = False,
    evaluate_before_training: bool = False,
) -> Tuple[Dict, Dict]:
    """
    Fine tune a model for a downstream task.

    In this script, we fine-tune a pre-trained language model. We use the
    [🤗 Datasets](https://github.com/huggingface/datasets/) library to
    load and preprocess the datasets.

    Parameters
    ----------
    train_file: Full training data file in JSON Lines format.
    val_file: Full validation data file in JSON Lines format.
    test_file: Full test data file in JSON Lines format.
    base_model: Base Transformer model that would be fine-tuned. It can be a
      path to a model or name of a out-of-the-box
    HuggingFace model.
    output_model: Path where the output model will be saved.
    training_data_fraction: What fraction of the full training data that
      should be used for fine-tuning.
    logging_dir: Path for the logging directory.
    tokenizer_model: Tokenizer model. Must be a HuggingFace Tokenizer model.
    evaluation_strategy:
            The evaluation strategy to adopt during training. Possible values
            are:

                * `"no"`: No evaluation is done during training.
                * `"steps"`: Evaluation is done (and logged) every `eval_steps`.
                * `"epoch"`: Evaluation is done at the end of each epoch.
    save_strategy:
            The checkpoint save strategy to adopt during training. Possible
            values are:

                * `"no"`: No save is done during training.
                * `"epoch"`: Save is done at the end of each epoch.
                * `"steps"`: Save is done every `save_steps`.
    num_train_epochs:
            Total number of training epochs to perform (if not an integer, will
            perform the decimal part percents of the last epoch before stopping
            training).
    save_total_limit:
            If a value is passed, will limit the total amount of checkpoints.
            Deletes the older checkpoints in `output_dir`.
    train_batch_size:
        Per device training batch size.
    eval_batch_size:
        Per device evaluation batch size.
    evaluation_metrics_output:
        File name where the evaluation metrics would be written.
    load_best_model_at_end:
         Whether or not to load the best model found during training at
         the end of training. Defaults to False.
    metric_for_best_model:
        Use in conjunction with load_best_model_at_end to specify the metric
        to use to compare two different models. Must be the name of a metric
        returned by the evaluation with or without the prefix "eval_".
        Will default to "loss" if unspecified
        and load_best_model_at_end=True (to use the evaluation loss).

        If you set this value, greater_is_better will default to True.
        Don’t forget to set it to False if your metric is better when lower.
    greater_is_better:
        Use in conjunction with load_best_model_at_end and
        metric_for_best_model to specify if better models
        should have a greater metric or not. Will default to False.
    evaluate_before_training:
        Evaluating the untrained model.
    """
    # First, we use the `load_dataset` function to download, load, and cache
    # the dataset:
    raw_datasets = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "validation": val_file,
            "test": test_file,
        },
    )

    # only the text and label fields are required for text classification
    raw_datasets = drop_features_except(raw_datasets, ["text", "label"])

    # The `raw_datasets` object is a dictionary with three keys: `"train"`,
    # `"test"` and `"unsupervised"` (which correspond to the three splits of
    # that dataset).

    # Compute the number of labels automatically from the training data.
    labels_set = set()
    raw_datasets["train"].map(lambda x: labels_set.add(x["label"]))
    num_labels = len(labels_set)

    # To preprocess our data, we use the tokenizer passed as the parameter,
    # for example the tokenizer of the
    # [`distilroberta-base`](https://huggingface.co/distilroberta-base)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=True)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
    )

    # Next we will generate a small subset of the training and validation set,
    # to enable faster training:

    train_data_size = int(
        tokenized_datasets.shape["train"][0] * training_data_fraction
    )
    logger.info("Training data size: %s", train_data_size)
    logger.info(
        "Training data original size: %s", tokenized_datasets.shape["train"][0]
    )

    full_train_dataset = (
        tokenized_datasets["train"]
        .shuffle(seed=42)
        .select(range(train_data_size))
    )
    full_test_dataset = tokenized_datasets["test"]
    full_validation_dataset = tokenized_datasets["validation"]

    # Fine-tuning in PyTorch with the Trainer API
    # --------------------------------------------
    # PyTorch does not provide a training loop. The 🤗 Transformers
    # library provides a `Trainer` API that is optimized for 🤗 Transformers
    # models

    # First, we define our model:
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=num_labels
    )

    # Then, we define our `Trainer`, we need to instantiate a
    # `TrainingArguments`. This class contains all the hyper-parameters
    # we can tune for the `Trainer` or the flags to activate the different
    # training options it supports.

    model_name = output_model.split("/")[-1]
    training_args = TrainingArguments(
        model_name,
        evaluation_strategy=evaluation_strategy,
        num_train_epochs=num_train_epochs,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        logging_dir=logging_dir,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
    )

    # To fine-tune our model, we need to call `trainer.train()` which will
    # start the training. To have the `Trainer` compute and report metrics,
    # we need to give it a `compute_metrics` function that takes predictions
    # and labels (grouped in a namedtuple called `EvalPrediction`) and return
    # a dictionary with string items (the metric names) and float values (the
    # metric values).

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_train_dataset,
        eval_dataset=full_validation_dataset,
        compute_metrics=compute_classification_metrics,
    )
    eval_before = {}
    if evaluate_before_training:
        logger.info("Evaluating the untrained model...")
        untrained_evaluator = Trainer(
            model=model,
            args=training_args,
            eval_dataset=full_test_dataset,
            compute_metrics=compute_classification_metrics,
        )
        eval_before = untrained_evaluator.evaluate()

    trainer.train()
    trainer.save_model(output_model)

    trained_model = AutoModelForSequenceClassification.from_pretrained(
        output_model, num_labels=num_labels
    )

    logger.info("Evaluating the fine-tuned model...")
    final_evaluator = Trainer(
        model=trained_model,
        args=training_args,
        eval_dataset=full_test_dataset,
        compute_metrics=compute_classification_metrics,
    )

    eval_after = final_evaluator.evaluate()

    if evaluation_metrics_output is not None:
        with open(evaluation_metrics_output, "w", encoding="utf-8") as fout:
            json.dump(eval_after, fout, indent=4)

    return eval_before, eval_after




def drop_features_except(
    dataset: DatasetDict, retain_fields: List[str]
) -> DatasetDict:
    """
    Drop all features from the given Dataset except those specified.

    Typically this function is used to retain the "text" and "label" features.

    Parameters
    ----------
    dataset: DatasetDict
        A dataset, typically containing train and test sets
    retain_fields: List[str]
        A list of features which will be retained in each dataset

    Returns
    -------
    DatasetDict:
        The new dataset containing only the features which were retained

    Raises
    ------
    ValueError:
        If the fields to retain are not in the data set
    """
    if not retain_fields:
        msg = (
            "There should be at least one feature to retain "
            "(an empty list was passed)."
        )
        logger.warning(msg)
        raise ValueError(msg)

    result = DatasetDict()
    for name, single_dataset in dataset.items():
        missing_fields = set(retain_fields).difference(
            set(single_dataset.column_names)
        )
        if missing_fields:
            msg = (
                f"Some features are not part of this dataset and could not"
                f" be retained: {missing_fields}. Features in the "
                f"dataset: {single_dataset.column_names}"
            )
            logger.warning(msg)
            raise ValueError(msg)

        remove_these_columns = set(single_dataset.column_names).difference(
            set(retain_fields)
        )
        result[name] = single_dataset.remove_columns(remove_these_columns)

    return result






def compute_classification_metrics(
    eval_pred: EvalPrediction,
) -> Dict[str, float]:
    """
    Compute classification metrics for the given predictions.

    The compute function needs to receive a tuple (with logits and labels)
    and has to return a dictionary with string keys
    (the name of the metric) and float values. It will be called at
    the end of each evaluation phase on the whole arrays of
    predictions/labels.

    This function uses prediction evaluation code from HuggingFace.

    Parameters
    ----------
    eval_pred: (logits, labels) tuple.

    Returns
    -------
    dict
    A dictionary with the following keys:
    - accuracy
    - f1
    - precision
    - recall
    """
    # Huggingface load_metrics for f1 and precision
    # haven't implemented the zero_division param. So muting the warnings
    # until they do so.
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    metric_acc = load("accuracy")
    metric_f1 = load('f1')
    metric_prec = load("precision")
    metric_rec = load("recall")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy_val = metric_acc.compute(
        predictions=predictions, references=labels
    )["accuracy"]

    f1_val = metric_f1.compute(
        predictions=predictions, references=labels, average="macro"
    )["f1"]

    prec_val = metric_prec.compute(
        predictions=predictions, references=labels, average="macro"
    )["precision"]

    recall_val = metric_rec.compute(
        predictions=predictions, references=labels, average="macro"
    )["recall"]

    return {
        "accuracy": accuracy_val,
        "f1": f1_val,
        "precision": prec_val,
        "recall": recall_val,
    }




eval_before, eval_after = transformer_finetune_textcat_downstream(
        train_file='drive/MyDrive/re_genai/bert_training_data/bert_processed_promise_train_bi.jsonl',
        val_file='drive/MyDrive/re_genai/bert_training_data/bert_processed_promise_val_bi.jsonl',
        test_file='drive/MyDrive/re_genai/bert_training_data/bert_processed_promise_test_bi.jsonl',
        #train_file='drive/MyDrive/re_genai/bert_training_data/bert_processed_promise_train_mul.jsonl',
        #val_file='drive/MyDrive/re_genai/bert_training_data/bert_processed_promise_val_mul.jsonl',
        #test_file='drive/MyDrive/re_genai/bert_training_data/bert_processed_promise_test_mul.jsonl',        
        #train_file='drive/MyDrive/re_genai/train_salmon_vilocify_small.jsonl',
        #val_file='drive/MyDrive/re_genai/test_salmon_vilocify_small.jsonl',
        #test_file='drive/MyDrive/re_genai/test_salmon_vilocify_small.jsonl',
        base_model= 'bert-base-uncased', # "bert-base-uncased",
        output_model='finetuned_test_bert_model',
        logging_dir='logs_cli_test_pythonapi',
        tokenizer_model='bert-base-uncased', # distilroberta-base
        save_strategy='epoch',
        num_train_epochs=5,
        training_data_fraction=1.0,
        load_best_model_at_end = True,
        evaluate_before_training=True,
    )

print(eval_before)
print(eval_after)