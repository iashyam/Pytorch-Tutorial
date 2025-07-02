import torch
from tqdm import tqdm_notebook as tqdm

def training_loop(model, epochs: int, train_loader, optimizer, criterion, device):
    """
    Training loop for the model.
    
    Args:
        model: The model to be trained.
        epochs (int): Number of epochs to train the model.
        train_loader: DataLoader for the training data.
        optimizer: Optimizer for updating model weights.
        criterion: Loss function to compute the loss.
        device: Device to run the training on (CPU or GPU).
    """
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        total_loss = 0.0
        N = len(train_loader)
        with tqdm(total=N) as pbar:
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()  # Zero the gradients
                output = model(data)  # Forward pass
                
                loss = criterion(output, target)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights
                
                total_loss += loss.item()
                pbar.update(1)  # Update progress bar 
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')


def validation_loop(model, val_loader, criterion, device):
    """
    Validation loop for the model.
    
    Args:
        model: The model to be validated.
        val_loader: DataLoader for the validation data.
        criterion: Loss function to compute the loss.
        device: Device to run the validation on (CPU or GPU).
    
    Returns:
        float: Average validation loss.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():  # Disable gradient computation
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # Forward pass
            
            loss = criterion(output, target)  # Compute loss
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    print(f'Validation Loss: {avg_loss:.4f}')
    return avg_loss
