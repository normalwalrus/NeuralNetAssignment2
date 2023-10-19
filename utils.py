import torch
from torch.autograd import Variable

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
