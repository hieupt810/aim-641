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


def plot_data_distribution(data_dir: str, batch_size: int, image_size: int):
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Plotting the data distribution
    class_counts = [0] * len(dataset.classes)
    for _, labels in loader:
        for label in labels:
            class_counts[label] += 1

    plt.figure(figsize=(10, 5))
    plt.bar(dataset.classes, class_counts)
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.show()


def plot_class_samples(
    data_dir: str, class_name: str, image_size: int, num_samples: int = 5
):
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageFolder(root=data_dir, transform=transform)

    class_index = dataset.class_to_idx[class_name]
    class_samples = [img for img, label in dataset if label == class_index][
        :num_samples
    ]
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(class_samples):
        img = img.permute(1, 2, 0).numpy()
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{class_name.replace('_', ' ').capitalize()} - Sample {i + 1}")

    plt.show()


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


def load_model(model_class, model_path: str, device):
    import torch

    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def plot_confusion_matrix(model, test_loader, class_names, device):
    "Usage: plot_confusion_matrix(model, test_loader, dataset.classes, Config.DEVICE)"
    import seaborn as sns
    import torch
    from matplotlib import pyplot as plt
    from sklearn.metrics import confusion_matrix

    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.show()


def lime_interpretation(model, test_dataset, image_size, mean, std):
    "Usage: lime_interpretation(model, test_dataset, Config.IMAGE_SIZE, mean, std)"
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from lime import lime_image
    from PIL import Image
    from skimage.segmentation import mark_boundaries
    from torchvision import transforms

    def batch_predict(images):
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        model.eval()
        batch = torch.stack(tuple(transform(i) for i in images), dim=0)
        batch = batch.to(next(model.parameters()).device)

        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    pil_transform = transforms.Compose([transforms.Resize((image_size, image_size))])
    explainer = lime_image.LimeImageExplainer()

    num_samples = len(test_dataset)
    indices = np.random.choice(num_samples, 5, replace=False)
    _, axes = plt.subplots(1, 5, figsize=(20, 4))

    for i, idx in enumerate(indices):
        img_path, _ = test_dataset.dataset.samples[test_dataset.indices[idx]]
        original_image = Image.open(img_path).convert("RGB")
        pil_image = pil_transform(original_image)
        explanation = explainer.explain_instance(
            np.array(pil_image),
            batch_predict,
            top_labels=5,
            hide_color=0,
            num_samples=1000,
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=False,
            num_features=5,
            hide_rest=False,
        )
        img_boundry = mark_boundaries(temp / 255.0, mask)
        axes[i].imshow(img_boundry)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
