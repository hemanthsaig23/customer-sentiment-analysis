# Customer Sentiment Analysis Using BERT and PyTorch
# Technologies: BERT, PyTorch, Transformers, Plotly, NLP, Transfer Learning

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load pretrained BERT
MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

class ReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_len=128):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        encoding = self.tokenizer.encode_plus(
            review,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }

def generate_sample_data(n=10000):
    """Generate sample customer review data."""
    np.random.seed(42)
    positive = [
        "Great product, highly recommend!",
        "Excellent quality and fast shipping.",
        "Absolutely love it, worth every penny.",
        "Amazing customer service, will buy again."
    ]
    negative = [
        "Terrible quality, waste of money.",
        "Product broke after 2 days.",
        "Customer service was rude and unhelpful.",
        "Not as described, very disappointed."
    ]
    neutral = [
        "It's okay, nothing special.",
        "Average product for the price.",
        "Delivered on time but product is mediocre."
    ]
    
    reviews, labels = [], []
    for _ in range(n):
        choice = np.random.choice([0, 1, 2], p=[0.25, 0.5, 0.25])  # negative, positive, neutral
        if choice == 0:
            reviews.append(np.random.choice(negative))
        elif choice == 1:
            reviews.append(np.random.choice(positive))
        else:
            reviews.append(np.random.choice(neutral))
        labels.append(choice)
    
    df = pd.DataFrame({'review': reviews, 'sentiment': labels})
    return df

def train_model(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc='Training'):
        optimizer.zero_grad()
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy, predictions, true_labels

def visualize_results(y_true, y_pred):
    """Create Plotly visualizations."""
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Negative', 'Positive', 'Neutral'],
                    y=['Negative', 'Positive', 'Neutral'],
                    title='Confusion Matrix - Sentiment Analysis')
    fig.write_html('confusion_matrix.html')
    print("Confusion matrix saved: confusion_matrix.html")
    
    # Sentiment distribution
    df_results = pd.DataFrame({'True': y_true, 'Predicted': y_pred})
    fig2 = px.histogram(df_results, x='Predicted', title='Predicted Sentiment Distribution',
                       labels={'Predicted': 'Sentiment'})
    fig2.write_html('sentiment_distribution.html')
    print("Sentiment distribution saved: sentiment_distribution.html")

def main():
    print("Customer Sentiment Analysis with BERT")
    print("="*60)
    
    # Generate/load data
    df = generate_sample_data(n=10000)
    print(f"Dataset size: {len(df)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'].values, df['sentiment'].values,
        test_size=0.2, random_state=42, stratify=df['sentiment']
    )
    
    # Create datasets
    train_dataset = ReviewDataset(X_train, y_train, tokenizer)
    test_dataset  = ReviewDataset(X_test,  y_test,  tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=32)
    
    # Load BERT model for classification
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Training loop (simplified: 2 epochs for demo)
    EPOCHS = 2
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_model(model, train_loader, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")
    
    # Evaluation
    accuracy, y_pred, y_true = evaluate_model(model, test_loader, device)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive', 'Neutral']))
    
    # Visualizations
    visualize_results(y_true, y_pred)
    
    print("\nModel training complete!")
    return model

if __name__ == '__main__':
    model = main()
