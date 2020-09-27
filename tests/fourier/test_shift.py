import os
import sys
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pytorch_fourier_analysis import fourier


class TestFFTShift:
    def test_compre_to_numpy(self):
        # single batch
        x = (
            torch.tensor([i for i in range(4)], dtype=torch.float)
            .view(1, 2, 2, 1)
            .repeat(1, 1, 1, 2)
        )
        assert fourier.fft_shift(x).equal(torch.from_numpy(np.fft.fftshift(x.numpy())))

        x = (
            torch.tensor([i for i in range(9)], dtype=torch.float)
            .view(1, 3, 3, 1)
            .repeat(1, 1, 1, 2)
        )
        assert fourier.fft_shift(x).equal(torch.from_numpy(np.fft.fftshift(x.numpy())))

        x = (
            torch.tensor([i for i in range(6)], dtype=torch.float)
            .view(1, 2, 3, 1)
            .repeat(1, 1, 1, 2)
        )
        assert fourier.fft_shift(x).equal(torch.from_numpy(np.fft.fftshift(x.numpy())))

        x = (
            torch.tensor([i for i in range(6)], dtype=torch.float)
            .view(1, 3, 2, 1)
            .repeat(1, 1, 1, 2)
        )
        assert fourier.fft_shift(x).equal(torch.from_numpy(np.fft.fftshift(x.numpy())))

        # multiple batch
        x = (
            torch.tensor([i for i in range(8)], dtype=torch.float)
            .view(2, 2, 2, 1)
            .repeat(1, 1, 1, 2)
        )
        assert fourier.fft_shift(x).equal(
            torch.from_numpy(np.fft.fftshift(x.numpy(), axes=(1, 2, 3)))
        )

        x = (
            torch.tensor([i for i in range(18)], dtype=torch.float)
            .view(2, 3, 3, 1)
            .repeat(1, 1, 1, 2)
        )
        assert fourier.fft_shift(x).equal(
            torch.from_numpy(np.fft.fftshift(x.numpy(), axes=(1, 2, 3)))
        )

        x = (
            torch.tensor([i for i in range(12)], dtype=torch.float)
            .view(2, 2, 3, 1)
            .repeat(1, 1, 1, 2)
        )
        assert fourier.fft_shift(x).equal(
            torch.from_numpy(np.fft.fftshift(x.numpy(), axes=(1, 2, 3)))
        )

        x = (
            torch.tensor([i for i in range(12)], dtype=torch.float)
            .view(2, 3, 2, 1)
            .repeat(1, 1, 1, 2)
        )
        assert fourier.fft_shift(x).equal(
            torch.from_numpy(np.fft.fftshift(x.numpy(), axes=(1, 2, 3)))
        )


class TestIFFTShift:
    def test_compre_to_numpy(self):
        # single batch
        x = (
            torch.tensor([i for i in range(4)], dtype=torch.float)
            .view(1, 2, 2, 1)
            .repeat(1, 1, 1, 2)
        )
        assert fourier.ifft_shift(x).equal(
            torch.from_numpy(np.fft.ifftshift(x.numpy()))
        )

        x = (
            torch.tensor([i for i in range(9)], dtype=torch.float)
            .view(1, 3, 3, 1)
            .repeat(1, 1, 1, 2)
        )
        assert fourier.ifft_shift(x).equal(
            torch.from_numpy(np.fft.ifftshift(x.numpy()))
        )

        x = (
            torch.tensor([i for i in range(6)], dtype=torch.float)
            .view(1, 2, 3, 1)
            .repeat(1, 1, 1, 2)
        )
        assert fourier.ifft_shift(x).equal(
            torch.from_numpy(np.fft.ifftshift(x.numpy()))
        )

        x = (
            torch.tensor([i for i in range(6)], dtype=torch.float)
            .view(1, 3, 2, 1)
            .repeat(1, 1, 1, 2)
        )
        assert fourier.ifft_shift(x).equal(
            torch.from_numpy(np.fft.ifftshift(x.numpy()))
        )

        # multiple batch
        x = (
            torch.tensor([i for i in range(8)], dtype=torch.float)
            .view(2, 2, 2, 1)
            .repeat(1, 1, 1, 2)
        )
        assert fourier.ifft_shift(x).equal(
            torch.from_numpy(np.fft.ifftshift(x.numpy(), axes=(1, 2, 3)))
        )

        x = (
            torch.tensor([i for i in range(18)], dtype=torch.float)
            .view(2, 3, 3, 1)
            .repeat(1, 1, 1, 2)
        )
        assert fourier.ifft_shift(x).equal(
            torch.from_numpy(np.fft.ifftshift(x.numpy(), axes=(1, 2, 3)))
        )

        x = (
            torch.tensor([i for i in range(12)], dtype=torch.float)
            .view(2, 2, 3, 1)
            .repeat(1, 1, 1, 2)
        )
        assert fourier.ifft_shift(x).equal(
            torch.from_numpy(np.fft.ifftshift(x.numpy(), axes=(1, 2, 3)))
        )

        x = (
            torch.tensor([i for i in range(12)], dtype=torch.float)
            .view(2, 3, 2, 1)
            .repeat(1, 1, 1, 2)
        )
        assert fourier.ifft_shift(x).equal(
            torch.from_numpy(np.fft.ifftshift(x.numpy(), axes=(1, 2, 3)))
        )
