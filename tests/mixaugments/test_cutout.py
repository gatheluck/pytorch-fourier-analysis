import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pytorch_fourier_analysis import mixaugments


if __name__ == "__main__":
    import torchvision

    model = torchvision.models.resnet50(num_classes=10, pretrained=False)
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        root="../../data/cifar10", train=False, transform=transform, download=True
    )
    loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=32)
    criterion = torch.nn.CrossEntropyLoss()

    mixaugment = mixaugments.Cutout(prob=1.0, cutout_size=16)

    for x, t in loader:
        loss, retdict = mixaugment(model, criterion, x, t)
        print(loss)
        print(retdict["output"].shape)
        print(retdict["x"].shape)
        torchvision.utils.save_image(retdict["x"], "../../logs/cutout_x.png")
        break