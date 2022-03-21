import numpy as np
import datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import nltk
import wandb

import argparse
from typing import Text

parser = argparse.ArgumentParser(description='Arguments to train seq2seq model')
parser.add_argument('--model_name', type=Text, default="facebook/bart-base",
                    help='name of the model to train')

parser.add_argument('--model_dir', type=Text, default=None,
                    help='place to save checkpoints and models')

parser.add_argument('--output_dir', type=Text, default=None,
                    help='place to save output files')

parser.add_argument('--encoder_max_len', type=int, default=512,
                    help='max input length')

parser.add_argument('--generation_max_len', type=int, default=90,
                    help='max output length')

parser.add_argument('--num_epochs', type=int, default=3,
                    help='max number of epochs to run')

parser.add_argument('--batch_size', type=int, default=16,
                    help='the batch size')

parser.add_argument('--seed', type=int, default=1,
                    help='seed for nondeterminism')

parser.add_argument('--patience', type=int, default=3,
                    help='patience for early stopping')

parser.add_argument('--weight_decay', type=float, default=0.1,
                    help='weight decay hyperparameter')

parser.add_argument('--learning_rate', type=float, default=5e-5,
                    help='learning rate')

parser.add_argument('--wandb', type=bool, default=True,
                    help='whether to run wandb logging')

parser.add_argument('--do_train', action='store_true', default=False,
                    help='whether to run training')

parser.add_argument('--do_eval', action='store_true', default=False,
                    help='whether to run eval')

parser.add_argument('--save_model', action='store_true', default=False,
                    help='whether to save model')


parser.add_argument('--project', type=Text, default="perspective-shift",
                    help='the wandb project to log into')

parser.add_argument('--group', type=Text, default="",
                    help='the wandb project group to log into')

parser.add_argument('--name', type=Text, default="",
                    help='a name for the wandb run')

parser.add_argument('--start', type=int, default=0, help='start index of val')
parser.add_argument('--end', type=int, default=-1, help='end index of val')

parser.add_argument('--start_train', type=int, default=0, help='start index of train')
parser.add_argument('--end_train', type=int, default=-1, help='end index of train')

from load_data import load_prompts

args = parser.parse_args()
model_dir = args.model_dir if args.model_dir is not None else "models/" + args.group + "/" + args.name
output_dir = args.output_dir if args.output_dir is not None else "outputs/" + args.group + "/" + args.name


if args.wandb:
    wandb.init(name=args.name, group=args.group, entity="gormleylab", project=args.project)


def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(args.model_name, return_dict=False)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

train_data, test_data = load_prompts()
#print(train_data["original"][65:70])
#print(train_data["shifted"][65:70])

train_data["prompt"] = train_data["prompt"][args.start_train:args.end_train]
train_data["story"] = train_data["story"][args.start_train:args.end_train]

test_data["prompt"] = test_data["prompt"][args.start:args.end]
test_data["story"] = test_data["story"][args.start:args.end]
nltk.download("punkt", quiet=True)
metric = datasets.load_metric("rouge")

train_data_txt = datasets.Dataset.from_dict(train_data)
validation_data_txt = datasets.Dataset.from_dict(test_data)


NUM_EPOCHS = args.num_epochs


def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length, train=False):
    source, target = batch['prompt'], batch['story']

    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )

    batch = {k: v for k, v in source_tokenized.items()}
    # Ignore padding in the loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch


train_data = train_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, args.encoder_max_len, args.generation_max_len, train=True
    ),
    batched=True,
    remove_columns=train_data_txt.column_names,

)

validation_data = validation_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, args.encoder_max_len, args.generation_max_len,
    ),
    batched=True,
    remove_columns=validation_data_txt.column_names,
)



def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    result["eval_rouge"] = result["rouge1"]
    print(result.keys())
    return result



training_args = Seq2SeqTrainingArguments(
    output_dir=model_dir,
    seed=args.seed,
    fp16=True,
    num_train_epochs=NUM_EPOCHS,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=1,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    predict_with_generate=True,
    load_best_model_at_end=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="rouge",
)

data_collator = DataCollatorForSeq2Seq(tokenizer)

trainer = Seq2SeqTrainer(
        model=model_init(),
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=validation_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience = args.patience)],
)

"""Train the model"""

if args.do_train:
    trainer.train()

"""Evaluate after fine-tuning"""

if args.do_eval:
    trainer.evaluate(max_length=args.generation_max_len)

if args.do_train and args.save_model:
    trainer.save_model(model_dir + "/final")

def generate_output(test_samples, trainer):
    inputs = tokenizer(
        test_samples["prompt"],
        padding="max_length",
        truncation=True,
        max_length=args.encoder_max_len,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(trainer.device)
    attention_mask = inputs.attention_mask.to(trainer.device)
    outputs = trainer.predict(input_ids, attention_mask=attention_mask, max_length=args.generation_max_len)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str

test_samples = validation_data_txt
p=0


with open(output_dir + ".txt", "a") as f:
    for i in range(args.batch_size + 1, len(test_samples) + 1, args.batch_size):
        stories_after_tuning = generate_output(test_samples.select(range(p, i)), trainer)[1]
        p = i
        for story in stories_after_tuning:
            print(story)
            f.write(story + "\n")
        print(p)
    if p < len(test_samples):
        stories_after_tuning = generate_output(test_samples.select(range(p, len(test_samples))), trainer)[1]
        for story in stories_after_tuning:
            print(story)
            f.write(story + "\n")
