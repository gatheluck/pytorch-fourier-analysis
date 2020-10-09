import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pytorch_fourier_analysis import augmentations


if __name__ == "__main__":
    import torchvision
    import functools
    model = torchvision.models.resnet50(num_classes=10, pretrained=False)
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root="../../data", train=False, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=32)
    criterion = torch.nn.CrossEntropyLoss()

    mixaugment = augmentations.CutMix(criterion=criterion, alpha=1.0, prob=1.0)

    for x, t in loader:
        loss, retdict = mixaugment(model, x, t)
        print(loss)
        print(retdict["output"].shape)
        print(retdict["x"].shape)
        torchvision.utils.save_image(retdict["x"], "../../logs/cutmix_x.png")
        raise NotImplementedError