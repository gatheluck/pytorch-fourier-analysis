import os
import sys
import torch
import torchvision
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pytorch_fourier_analysis import fourier




if __name__ == "__main__":
    fourier.create_soft_basis_grid(16, 32, "logs", "cuda")

    # device = "cuda"
    # # mu = torch.tensor([[0.5, 0.0], [-1.0, 1.0], [0.0, 0.0]], dtype=torch.float, device=device, requires_grad=True)
    # index_h = (2.0 * torch.rand(3, dtype=torch.float, device=device, requires_grad=True)) - 1.0
    # index_w = torch.rand(3, dtype=torch.float, device=device, requires_grad=True)
    # mu = torch.stack([index_h, index_w], dim=-1)
    # # sigma = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float, device=device, requires_grad=True)
    # sigma = torch.ones(3, dtype=torch.float, device=device, requires_grad=True) * 0.01
    # print(mu.shape)
    # print(sigma.shape)
    # s = fourier.soft_spectrum_matrix(32, mu, sigma=sigma, device=device)
    # print(s)
    # print(s.shape)
    # torchvision.utils.save_image(s[0], "../../logs/soft_spectrum_0.png")
    # torchvision.utils.save_image(s[1], "../../logs/soft_spectrum_1.png")
    # torchvision.utils.save_image(s[2], "../../logs/soft_spectrum_2.png")

    # soft_basis, spectrum = fourier.soft_basis(32, mu, sigma, device)
    # for i in range(3):
    #     print("spectrum")
    #     print("shape: {}".format(spectrum.shape))
    #     print("max: {}".format(spectrum[0].max()))
    #     print("min: {}".format(spectrum[0].min()))
    #     print("mean: {}".format(spectrum[0].mean()))
    #     # torchvision.utils.save_image(spectrum[i], "../../logs/spectrum_{i}.png".format(i=i))

    #     basis_i = soft_basis[i].repeat(3, 1, 1) + 0.5
    #     print("basis")
    #     print("shape: {}".format(basis_i.shape))
    #     print("max: {}".format(basis_i[0].max()))
    #     print("min: {}".format(basis_i[0].min()))
    #     # torchvision.utils.save_image(soft_basis[i].repeat(3, 1, 1) + 0.5, "../../logs/soft_basis_{i}.png".format(i=i))
        


    # batch_size = 8
    # device = "cuda"
    # index_h = (2.0 * torch.rand(batch_size, dtype=torch.float, device=device, requires_grad=True)) - 1.0
    # index_w = torch.rand(batch_size, dtype=torch.float, device=device, requires_grad=True)
    # print(index_h)
    # print(index_w)
    # index = torch.stack([index_h, index_w], dim=-1)
    # sigma = torch.ones(batch_size, dtype=torch.float, device=device, requires_grad=True) * 0.1

    # print(index.shape)
    # print(index)
    # print(sigma.shape)
    # print(sigma)

    # spectrum, soft_basis = fourier.soft_basis(32, index, sigma, device)
    # for i in range(batch_size):
    #     print(soft_basis[i])
    #     torchvision.utils.save_image(spectrum[i], "../../logs/spectrum_{i}.png".format(i=i))
    #     torchvision.utils.save_image(soft_basis[i], "../../logs/soft_basis_{i}.png".format(i=i))
