from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import torch
import numpy as np
import evaluate
from sklearn.metrics import classification_report, confusion_matrix, f1_score

train_df = pd.read_csv("train_balanced.csv")
val_df = pd.read_csv("val_balanced.csv")
test_df = pd.read_csv("test_balanced.csv")

label_map = {'Negative': 0, 'Notr': 1, 'Positive': 2}
train_df['label'] = train_df['label'].map(label_map)
val_df['label'] = val_df['label'].map(label_map)
test_df['label'] = test_df['label'].map(label_map)

model_name = "dbmdz/electra-base-turkish-cased-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(list(train_df['text']), truncation=True, padding=True)
val_encodings = tokenizer(list(val_df['text']), truncation=True, padding=True)
test_encodings = tokenizer(list(test_df['text']), truncation=True, padding=True)

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        } | {'labels': torch.tensor(self.labels[idx])}

train_dataset = EmotionDataset(train_encodings, list(train_df['label']))
val_dataset = EmotionDataset(val_encodings, list(val_df['label']))
test_dataset = EmotionDataset(test_encodings, list(test_df['label']))

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir="./electra_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate(eval_dataset=test_dataset)
print("\n--- Test Sonuçları ---")
print(results)

preds_output = trainer.predict(test_dataset)
preds = np.argmax(preds_output.predictions, axis=-1)
labels = preds_output.label_ids

print("\n--- Detaylı Test Raporu ---")
print(classification_report(labels, preds, target_names=['Negative', 'Notr', 'Positive']))

cm = confusion_matrix(labels, preds)
print("\nConfusion Matrix:")
print(cm)

print("\nMacro F1 Score:", f1_score(labels, preds, average='macro'))

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Notr', 'Positive'])
cm_display.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("electra_confusion_matrix.png", dpi=300)
plt.show()

report = classification_report(labels, preds, target_names=['Negative', 'Notr', 'Positive'], output_dict=True)

metrics_df = pd.DataFrame(report).T.iloc[:3][['precision', 'recall', 'f1-score']]

metrics_df.plot(kind='bar', figsize=(8, 6), colormap='viridis')
plt.title('Classification Metrics by Class')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("electra_classification_report_metrics.png", dpi=300)
plt.show()