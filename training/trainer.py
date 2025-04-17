import torch
from torch import optim, nn

import os
from datetime import datetime

from data import dataset
from torch.utils.data import DataLoader

from models.model import SentimentClassificationModel
from tqdm import tqdm
from utils.utils import make_checkpoint_name
from visualization.attention_viz import visualize_multiple_examples
from utils.tokenizer import tokenizer
from config import config


def train(vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx, device, learning_rate, num_epochs, visualize_per_epoch=False):
    train_dataloader = DataLoader(
        dataset.train_subset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(
        dataset.val_subset, batch_size=16, shuffle=True)
    model = SentimentClassificationModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        pad_idx=pad_idx,
        bidirectional=True
    )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0

    try:
        with open("./checkpoints/best_val_acc.txt", "r") as f:
            best_val_acc = float(f.read())
            print(f"Best validation accuracy from file: {best_val_acc:.4f}")
    except FileNotFoundError:
        best_val_acc = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_epoch(
            model,
            train_dataloader,
            optimizer,
            criterion,
            device
        )

        val_loss, val_acc = evaluate(
            model,
            val_dataloader, criterion,
            device)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            torch.save(model.state_dict(), f"./checkpoints/{make_checkpoint_name()}__best.pth")

            with open("./checkpoints/best_val_acc.txt", "w") as f:
                f.write(str(best_val_acc))                
            print(f"Model saved to {make_checkpoint_name()}.pth")
        else:
            torch.save(model.state_dict(), f"./checkpoints/{make_checkpoint_name()}__last.pth")
        
        # Generate visualizations if enabled
        if visualize_per_epoch:
            os.makedirs("./visualizations", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_output_file = f"./visualizations/epoch_{epoch+1}_{timestamp}.png"
            print(f"Generating attention visualizations for epoch {epoch+1}...")
            visualize_multiple_examples(
                model,
                tokenizer,
                config.visualizer_example_sentences,
                device,
                figsize=(15, 4*len(config.visualizer_example_sentences)),
                output_file=viz_output_file
            )
            print(f"Visualizations saved to {viz_output_file}")

        print()



def train_epoch(model, dataloader, optimizer, criterion, device):

    model.train()
    epoch_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        predictions = torch.argmax(outputs, dim=1)
        correct_predictions = (predictions == labels).sum().item()

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        total_correct += correct_predictions
        total_samples += labels.size(0)

    return epoch_loss / len(dataloader), total_correct / total_samples


def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    total_correct = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids)

            loss = criterion(outputs, labels)

            predictions = torch.argmax(outputs, dim=1)
            correct_predictions = (predictions == labels).sum().item()

            total_correct += correct_predictions
            total_predictions += labels.size(0)

            epoch_loss += loss.item()
        accuracy = total_correct / total_predictions
        return epoch_loss / len(dataloader), accuracy
