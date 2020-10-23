import os
import sys
import math

import torch
import torchvision

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pytorch_fourier_analysis.attacks import PgdAttack
from pytorch_fourier_analysis import shared


if __name__ == "__main__":
    # test Gaussian
    input_size = 32
    mean = [0.49139968, 0.48215841, 0.44653091]
    std = [0.24703223, 0.24348513, 0.26158784]
    num_iteration = 7
    eps_max = 8.0
    step_size = eps_max / math.sqrt(num_iteration)
    norm = "linf"
    rand_init = True
    scale_each = True
    scale_eps = False
    avoid_target = True
    criterion = torch.nn.CrossEntropyLoss()

    transform = shared.get_transform(
        input_size, mean=mean, std=std, train=False, normalize=False,
    )
    dataset = torchvision.datasets.CIFAR10(
        root="../../data/cifar10", train=False, download=False, transform=transform
    )
    loader = torch.utils.data.DataLoader(dataset, 32, shuffle=False, num_workers=8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weightpath = "../testdata/weight_cifar10_wideresnet40_100ep.pth"
    model = shared.get_model(name="wideresnet40", num_classes=10)
    shared.load_model(model, weightpath)
    model = model.to(device)
    model.eval()

    for x, t in loader:
        x, t = x.to(device), t.to(device)
        output = model(x)
        err1 = shared.calc_error(output.detach(), t, topk=(1,))

        attack = PgdAttack(
            input_size,
            mean,
            std,
            num_iteration,
            eps_max,
            step_size,
            norm,
            rand_init,
            scale_eps,
            scale_each,
            avoid_target,
            criterion,
            device,
        )
        x_adv = attack(model, x, t)
        output_adv = model(x_adv)
        err1_adv = shared.calc_error(output_adv.detach(), t, topk=(1,))

        print(
            "clean err: {clean}, adv err: {adv}".format(
                clean=err1[0].item(), adv=err1_adv[0].item()
            )
        )
        torchvision.utils.save_image(x, "../../logs/clean.png")
        torchvision.utils.save_image(x_adv, "../../logs/adv.png")
        break
