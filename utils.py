def calculateNormalizeParameters(data_dir: str, image_size: int, batch_size: int):
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )
    dataset = ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize parameters
    mean, std = 0.0, 0.0
    total_images = 0
    for images, _ in loader:
        samples = images.size(0)
        images = images.view(samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += samples

    mean /= total_images
    std /= total_images
    return mean, std


def plot_metrics(
    train_accuracies, test_accuracies, train_losses, test_losses, choose_epoch
):
    from matplotlib import pyplot as plt

    epochs = range(1, len(train_accuracies) + 1)
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, test_accuracies, label="Test Accuracy")
    plt.plot(choose_epoch, train_accuracies[choose_epoch - 1], "ro")
    plt.plot(choose_epoch, test_accuracies[choose_epoch - 1], "ro")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Testing Accuracy")
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.plot(choose_epoch, train_losses[choose_epoch - 1], "ro")
    plt.plot(choose_epoch, test_losses[choose_epoch - 1], "ro")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Testing Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()
