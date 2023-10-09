import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_evaluator(args):

    data_files = {
        "train": args.dataset_path + "train.jsonl", 
        "test": args.dataset_path + "dev.jsonl"
        }
    dataset = load_dataset('json', data_files=data_files)

    # Load the DistilBERT tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # Tokenize the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Specify training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="steps",
        eval_steps=500,
        num_train_epochs=3,
        learning_rate=2e-5,
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)



def test_evaluator(args):

    # Load the trained model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained(args.output_dir)
    tokenizer = DistilBertTokenizer.from_pretrained(args.output_dir)

    # Load the test dataset
    if Path(args.dataset_path).is_dir():
        dataset_filename = args.dataset_path + "dev.jsonl"
    else:
        dataset_filename = args.dataset_path
    dataset = load_dataset('json', data_files=dataset_filename, split='train')

    # Tokenize the test dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Perform inference on the test dataset
    predictions = []
    i = 0
    for example in tqdm(tokenized_dataset):
        with torch.no_grad():
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0)  # Convert to tensor and add a batch dimension
            outputs = model(input_ids)  # Batch size of 1
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            predictions.append(predicted_class)


    # Get the true labels from the test dataset
    true_labels = tokenized_dataset["label"]

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## General args
    parser.add_argument("--dataset-path", type=str, help="the dataset path where the json is stored", default="data/evaluation/")
    parser.add_argument("--output-dir", type=str, help="the output directory to save the model", default="output/evaluator/")
    parser.add_argument("--train", action="store_true", help="whether to train the model", default=True)
    parser.add_argument("--evaluate", action="store_true", help="whether to evaluate the model", default=True)

    args = parser.parse_args()
    if args.train:
        train_evaluator(args)
    if args.evaluate:
        test_evaluator(args)
