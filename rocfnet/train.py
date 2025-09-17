# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr

# Import the model from model.py
from model import ROCFNet

def augment_transform(image, score):

    augment = transforms.Compose([
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05), 
        transforms.Lambda(lambda x: x * (0.8 + 0.4 * torch.rand(1))), 
        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=5)
    ])
    
    augmented_image = augment(image)
    if random.choice([True, False]):
        augmented_image = transforms.RandomHorizontalFlip(p=1.0)(augmented_image)
    else:
        augmented_image = transforms.RandomVerticalFlip(p=1.0)(augmented_image)
    
    new_score = score - 1 if score >= 28 else score - 6
    
    return augmented_image, new_score

class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, augment=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.annotations.iloc[idx, 1], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        if self.augment:
            augmented_image, new_label = self.augment(image, label)
            return image, augmented_image, label, new_label
        else:
            return image, label

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Datasets and DataLoaders
    train_dataset = ImageDataset(
        csv_file="dataset1.csv", 
        root_dir="dataset1", 
        transform=transform, 
        augment=augment_transform
    )
    
    val_dataset = ImageDataset(
        csv_file="dataset2.csv", 
        root_dir="dataset1", 
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Model, criterion, and optimizer setup
    model = CustomResNet().to(device)
    criterion = nn.L1Loss() # MAE Loss
    optimizer = optim.Adam(model.parameters(), lr=5e-7)
    
    # Training loop
    num_epochs = 100
    output_file = "training.txt"

    with open(output_file, "w") as f:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                # Handle both augmented and non-augmented batches
                if len(batch) == 4: # Augmented batch
                    orig, aug, labels, new_labels = batch
                    inputs = torch.cat([orig, aug]).to(device)
                    targets = torch.cat([labels, new_labels]).to(device)
                else: # Regular batch
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            all_preds, all_truths = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs).squeeze().cpu()
                    
                    val_loss += criterion(outputs, labels).item()
                    all_preds.extend(outputs.numpy())
                    all_truths.extend(labels.numpy())
            
            # Calculate and log metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            mae = mean_absolute_error(all_truths, all_preds)
            r2 = r2_score(all_truths, all_preds)
            pcc, _ = pearsonr(all_truths, all_preds)
            
            log = (f"Epoch {epoch+1}/{num_epochs}: "
                   f"Train Loss: {avg_train_loss:.4f}, "
                   f"Val Loss: {avg_val_loss:.4f}, "
                   f"MAE: {mae:.4f}, "
                   f"R²: {r2:.4f}, "
                   f"PCC: {pcc:.4f}\n")
            print(log, end='')
            f.write(log)
    
    print(f"\nTraining completed. Information saved to {output_file}")

if __name__ == "__main__":
    main()