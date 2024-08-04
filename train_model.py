import json
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import logging
import gc
import traceback
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from seqeval.scheme import IOB2

# Setup logging
logging.basicConfig(level=logging.INFO)

def load_data(file_path, max_samples=1000):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Define NER labels
        ner_labels = ["O", "B-CARD", "I-CARD"]
        label_map = {label: i for i, label in enumerate(ner_labels)}
        
        logging.info(f"Loaded and converted {len(data)} samples to NER format from {file_path}")
        return data[:max_samples], label_map
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None, None

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
    
def tokenize_and_align_labels(examples, tokenizer, label_to_id):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]] if word_idx < len(label) else -100)
            else:
                if label[word_idx].startswith("B-"):
                    label_ids.append(label_to_id["I-" + label[word_idx][2:]])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p, id_to_label):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_predictions = [pred for pred in true_predictions if pred]
    true_labels = [label for label in true_labels if label]

    if not true_labels or not true_predictions:
        logging.info("No samples for evaluation")
        return {
            "precision": 0,
            "recall": 0,
            "f1": 0,
        }

    results = {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }
    
    if true_labels and true_predictions:
        try:
            report = classification_report(true_labels, true_predictions, zero_division=0)
            logging.info("\n" + report)
        except ValueError as e:
            logging.error(f"Error in classification report: {e}")
    else:
        logging.info("No samples for classification report")

    return results

def main():
    logging.info("Starting script...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        data, label_to_id = load_data('sanity_check_tagged_data.json')
        if data is None:
            return

        hf_dataset = create_dataset(data)
        if hf_dataset is None:
            return
        logging.info(f"Dataset size: {len(hf_dataset)}")

        train_test = hf_dataset.train_test_split(test_size=0.2)
        logging.info(f"Train set size: {len(train_test['train'])}")
        logging.info(f"Test set size: {len(train_test['test'])}")

        model_name = "dslim/bert-base-NER"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        id_to_label = {i: label for label, i in label_to_id.items()}
        num_labels = len(label_to_id)

        logging.info(f"Label map: {label_to_id}")

        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
        logging.info("Loaded tokenizer and model")

        tokenized_datasets = train_test.map(
            lambda examples: tokenize_and_align_labels(examples, tokenizer, label_to_id),
            batched=True,
            remove_columns=train_test["train"].column_names
        )
        logging.info(f"Tokenized train set size: {len(tokenized_datasets['train'])}")
        logging.info(f"Tokenized test set size: {len(tokenized_datasets['test'])}")
        logging.info(f"Train dataset features: {tokenized_datasets['train'].features}")

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
            metric_for_best_model="f1",
            push_to_hub=False,
            greater_is_better=True,
            remove_unused_columns=False,
        )

        data_collator = DataCollatorForTokenClassification(tokenizer)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda p: compute_metrics(p, id_to_label)
        )
        logging.info("Initialized Trainer")

        logging.info("Starting training...")
        trainer.train()
        logging.info("Training completed")

        trainer.save_model("models/mtg_card_name_model")
        logging.info("Model saved")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.error(traceback.format_exc())

    finally:
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()