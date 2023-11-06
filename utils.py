import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np

def output_label(label):
    output_mapping = {0: "T-shirt/Top",
                    1: "Trouser",
                    2: "Pullover",
                    3: "Dress",
                    4: "Coat", 
                    5: "Sandal", 
                    6: "Shirt",
                    7: "Sneaker",
                    8: "Bag",
                    9: "Ankle Boot"}
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]

def train_loop(train_loader, model, loss_fn, optimizer, device):
    size = len(train_loader.dataset)
    num_batches = len(train_loader)

    train_loss, train_correct = 0, 0

    for images, labels in train_loader:
        # Transfering images and labels to GPU if available
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass 
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, dim=1)

        train_correct += (predicted == labels).type(torch.float).sum().item()

    train_loss /= num_batches
    train_correct /=size
    
    return train_loss, train_correct


def test_loop(test_loader, model, loss_fn, device):

    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    test_loss, test_correct = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:

            test, labels = images.to(device), labels.to(device)

            outputs = model(test)

            test_loss += loss_fn(outputs, labels).item()

            _, predicted = torch.max(outputs, dim=1)
            test_correct += (predicted == labels).type(torch.float).sum().item()

    test_loss /= num_batches
    test_correct /= size
    
    return test_loss, test_correct

class FashionMnist_Dataset(Dataset):
    def __init__(self, data, normalise = True) -> None:
        super().__init__()

        self.data = data
        self.normalise = normalise
        self.image_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

    
    def __len__(self):
        #2 times since we only have positive examples and every alternate example is negative
        return len(self.data)

    def __getitem__(self, index):

        tensor, label = self.data[index]
        if self.normalise:
            tensor = self.image_transform(tensor)

        return tensor, label
    
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False