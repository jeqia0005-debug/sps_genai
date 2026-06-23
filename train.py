import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from app.cnn_model import CNN


def main():
    # Resize CIFAR10 images to 64x64 and convert to tensors (to match the network input)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2
        # If you hit a multiprocessing error, set num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = CNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 3
    model.train()
    for epoch in range(EPOCHS):
        running_loss, correct, total = 0.0, 0, 0
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            if (i + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}, batch {i+1}, loss={running_loss/100:.3f}")
                running_loss = 0.0
        print(f">>> Epoch {epoch+1} done, train accuracy = {100*correct/total:.1f}%")

    torch.save(model.state_dict(), "cifar10_cnn.pth")
    print("\nModel saved to cifar10_cnn.pth")


if __name__ == "__main__":
    main()
