"""
train_bert_simple.py
Minimal BERT training - 2 epochs only
"""

import sys
import os
sys.path.insert(0, '/content/PoliteMeter')
os.chdir('/content/PoliteMeter')

# Force clear cache
if 'src.models_bert' in sys.modules:
    del sys.modules['src.models_bert']

from src.data_loader import load_processed_data
from sklearn.model_selection import train_test_split

# Import after path setup
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

print("="*80)
print("🚀 BERT 2-EPOCH TRAINING (SIMPLIFIED)")
print("="*80)

# Load data
print("\n📊 Loading data...")
data = load_processed_data()
X = data['Cleaned_Text'].values
y = data['Politeness'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Load tokenizer and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}")

print("\n📦 Loading BERT...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.to(device)

# Tokenize
print("🔤 Tokenizing...")
label_map = {'Impolite': 0, 'Neutral': 1, 'Polite': 2}

def tokenize_data(texts, labels):
    encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=128)
    labels_numeric = [label_map[l] for l in labels]
    
    dataset = []
    for i in range(len(texts)):
        dataset.append({
            'input_ids': torch.tensor(encodings['input_ids'][i]),
            'attention_mask': torch.tensor(encodings['attention_mask'][i]),
            'labels': torch.tensor(labels_numeric[i])
        })
    return dataset

train_dataset = tokenize_data(X_train, y_train)
val_dataset = tokenize_data(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 2
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

print("\n🔄 Training for 2 epochs...")

for epoch in range(2):
    # Train
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/2")
    for batch in pbar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    
    # Validate
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)
    
    val_acc = 100 * val_correct / val_total
    print(f"Epoch {epoch+1}/2 - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")

# Test
print("\n📊 Testing...")
test_dataset = tokenize_data(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=16)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert back to labels
reverse_map = {0: 'Impolite', 1: 'Neutral', 2: 'Polite'}
pred_labels = [reverse_map[p] for p in all_preds]
true_labels = [reverse_map[l] for l in all_labels]

# Results
accuracy = accuracy_score(true_labels, pred_labels)
report = classification_report(true_labels, pred_labels)
cm = confusion_matrix(true_labels, pred_labels, labels=['Impolite', 'Neutral', 'Polite'])

print("\n" + "="*80)
print("📊 RESULTS")
print("="*80)
print(f"\n🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\n📋 Classification Report:")
print(report)
print(f"\n🔢 Confusion Matrix:")
print("           Imp  Neu  Pol")
print(f"Imp {cm[0]}")
print(f"Neu {cm[1]}")
print(f"Pol {cm[2]}")

print("\n" + "="*80)
print("📊 COMPARISON")
print("="*80)
print(f"SVM:            68.43%")
print(f"BERT (2 ep):    {accuracy*100:.2f}%")
print(f"BERT (4 ep):    64.05%")
print(f"LSTM:           47.99%")

if accuracy > 0.6843:
    print(f"\n🎉 BERT wins by {(accuracy-0.6843)*100:.2f} pp!")
elif accuracy >= 0.68:
    print(f"\n✅ BERT matches SVM!")
else:
    print(f"\n📊 SVM still ahead by {(0.6843-accuracy)*100:.2f} pp")

# Save
model.save_pretrained('models/bert_2ep_simple')
tokenizer.save_pretrained('models/bert_2ep_simple')
print("\n💾 Model saved!")