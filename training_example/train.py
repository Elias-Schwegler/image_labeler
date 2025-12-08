import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import argparse

# --- Dataset Definition ---
class LabelerDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.params_file = os.path.join(data_dir, "labels.json")
        
        if not os.path.exists(self.params_file):
            raise FileNotFoundError(f"labels.json not found in {data_dir}")

        with open(self.params_file, "r") as f:
            self.data = json.load(f)

        # Filter out invalid images
        self.valid_data = []
        for item in self.data:
            img_name = item.get("filename")
            # In the organized dataset, images are in the root of data_dir
            if img_name and os.path.exists(os.path.join(data_dir, img_name)):
                self.valid_data.append(item)
        
        self.labels = [item["label"] for item in self.valid_data]

    def set_class_map(self, class_to_idx):
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        item = self.valid_data[idx]
        img_path = os.path.join(self.data_dir, item["filename"])
        image = Image.open(img_path).convert("RGB")
        label = item["label"]
        
        if self.transform:
            image = self.transform(image)
            
        label_idx = self.class_to_idx[label]
        return image, label_idx

# --- Training Function ---
def train_model(data_dir, num_epochs=5, batch_size=4, learning_rate=0.001):
    print(f"Training on data in: {data_dir}")
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load Datasets
    image_datasets = {}
    
    # Train Set
    train_dir = os.path.join(data_dir, "train")
    if not os.path.exists(train_dir):
        print(f"Error: Train directory {train_dir} does not exist.")
        return

    image_datasets['train'] = LabelerDataset(train_dir, transform=data_transforms['train'])
    
    # Test Set
    test_dir = os.path.join(data_dir, "test")
    if os.path.exists(test_dir):
         image_datasets['test'] = LabelerDataset(test_dir, transform=data_transforms['test'])
    else:
        print("Warning: Test directory not found. Evaluation will be skipped.")

    # Determine Classes
    # We derive classes from the training set
    all_labels = sorted(list(set(image_datasets['train'].labels)))
    class_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    print(f"Found {len(all_labels)} classes: {all_labels}")
    
    # Set class map for both datasets
    image_datasets['train'].set_class_map(class_to_idx)
    if 'test' in image_datasets:
        image_datasets['test'].set_class_map(class_to_idx)

    # Dataloaders
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)
                   for x in image_datasets}

    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}

    # Model Setup (ResNet18)
    # We use weights=None for simplicity/offline use, or typically weights='IMAGENET1K_V1'
    # Here we assume we want pretrained if possible, but let's default to weights=None and warn
    try:
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1
    except ImportError:
        weights = None # For older torchvision versions
        
    model = models.resnet18(weights=weights)
    
    # Replace last layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(all_labels))
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training Loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase not in image_datasets:
                continue

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # Save Model
    save_path = os.path.join(data_dir, "model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # Save Class Map
    class_map_path = os.path.join(data_dir, "class_map.json")
    with open(class_map_path, "w") as f:
        json.dump(class_to_idx, f, indent=4)
    print(f"Class map saved to {class_map_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a ResNet model on labeled data.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the split dataset (containing train/ and test/ folders)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    
    args = parser.parse_args()
    
    train_model(args.data_dir, num_epochs=args.epochs, batch_size=args.batch)
