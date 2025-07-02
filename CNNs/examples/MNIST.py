import torch 
from torch.utils.data import DataLoader
from torchvision import transforms
import core
import create_dataset as create_dataset
import training_loop, models, dataset
import os

print(os.getcwd())
#GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# MNIST dataset loading and preprocessing
total_dataset = create_dataset.MNIST()
train_dataset = total_dataset.training_data
test_dataset = total_dataset.test_data

# make a custom dataset
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x/255),  ## noramalize the images
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10)
])

train_dataset = dataset.CustomDataset(
    data=train_dataset[0],
    targets=train_dataset[1],
    transform=transforms
)

test_dataset = dataset.CustomDataset(
    data=test_dataset[0],
    targets=test_dataset[1],
    transform=transforms
)

# DataLoader for batching  
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# Model initialization
shape = train_dataset[0][0].shape
model = models.VGG(input_shape=shape, output_dim=10).to(device)

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Model initialized and ready for training.")
# Training the model
epochs = 10
training_loop.training_loop(
    model=model,
    epochs=epochs,
    train_loader=train_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device
)   

# Validation of the model
validation_loss = training_loop.validation_loop(
    model=model,
    val_loader=test_loader,
    criterion=criterion,
    device=device
)
