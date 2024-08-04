import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import logging
import gc

# Setup logging
logging.basicConfig(level=logging.INFO)

def load_data(file_path, max_samples=100):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Loaded {len(data)} samples from {file_path}")
        return data[:max_samples]  # Only take the first max_samples
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def create_dataset(data):
    try:
        hf_dataset = Dataset.from_dict({
            'tokens': [item['tokens'] for item in data],
            'labels': [item['labels'] for item in data]
        })
        logging.info(f"Created dataset with {len(hf_dataset)} samples")
        return hf_dataset
    except Exception as e:
        logging.error(f"Error creating dataset: {e}")
        return None

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding='max_length',
        max_length=512
    )

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx] if word_idx < len(label) else -100)
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def main():
    logging.info("Starting script...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        # Load a small subset of data
        data = load_data('data/tagged_data.json', max_samples=100)
        if data is None:
            return

        # Create dataset
        hf_dataset = create_dataset(data)
        if hf_dataset is None:
            return

        # Split dataset
        train_test = hf_dataset.train_test_split(test_size=0.2)
        logging.info(f"Split dataset into {len(train_test['train'])} train and {len(train_test['test'])} test samples")

        # Load tokenizer and model
        model_name = "distilbert-base-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Determine the number of labels dynamically
        unique_labels = set(label for item in data for label in item["labels"])
        num_labels = len(unique_labels)
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
        logging.info("Loaded tokenizer and model")

        # Tokenize datasets
        tokenized_datasets = train_test.map(
            lambda examples: tokenize_and_align_labels(examples, tokenizer),
            batched=True,
            remove_columns=train_test["train"].column_names
        )
        logging.info("Tokenized datasets")
        logging.info(f"Train dataset features: {tokenized_datasets['train'].features}")

        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
        )

        # Define compute_metrics function
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            mask = labels != -100
            labels = labels[mask]
            preds = preds[mask]
            correct = (labels == preds).sum().item()
            total = len(labels)
            accuracy = correct / total if total > 0 else 0
            return {"accuracy": accuracy}

        # Initialize Trainer
        data_collator = DataCollatorForTokenClassification(tokenizer)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        logging.info("Initialized Trainer")

        # Train the model
        logging.info("Starting training...")
        trainer.train()
        logging.info("Training completed")

        # Save the model
        trainer.save_model("models/mtg_card_name_model")
        logging.info("Model saved")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.error(traceback.format_exc())

    finally:
        # Clean up
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
