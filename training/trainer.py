import torch
from torch import optim, nn


from data import dataset
from torch.utils.data import DataLoader
from config import config
from models.model import SentimentClassificationModel
from tqdm import tqdm
from utils.utils import make_checkpoint_name


def train():
    train_dataloader = DataLoader(
        dataset.train_subset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(
        dataset.val_subset, batch_size=16, shuffle=True)
    model = SentimentClassificationModel(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        pad_idx=config.pad_idx,
        bidirectional=True
    )
    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_acc = 0

    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch+1}/{config.num_epochs}")

        train_loss, train_acc = train_epoch(
            model,
            train_dataloader,
            optimizer,
            criterion,
            config.device
        )

        val_loss, val_acc = evaluate(
            model,
            val_dataloader, criterion,
            config.device)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{make_checkpoint_name()}.pth")
            print(f"Model saved to {make_checkpoint_name()}.pth")

        print()


def train_epoch(model, dataloader, optimizer, criterion, device):

    model.train()
    epoch_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

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
            correct_predictions += (predictions == labels).sum().item()

            total_correct += correct_predictions
            total_predictions += labels.size(0)

            epoch_loss += loss.item()
        accuracy = total_correct / total_predictions
        return epoch_loss / len(dataloader), accuracy
