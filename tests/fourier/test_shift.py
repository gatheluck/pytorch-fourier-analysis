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

    def test_complex_tensor(self):
        # single batch
        x = (
            torch.tensor([0, 1, 2, 3], dtype=torch.cfloat)
            .view(1, 2, 2)
        )
        x_ans = (
            torch.tensor([3, 2, 1, 0], dtype=torch.cfloat)
            .view(1, 2, 2)
        )
        assert fourier.fft_shift(x).equal(x_ans)

        x = (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=torch.cfloat)
            .view(1, 4, 4)
        )
        x_ans = (
            torch.tensor([10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5], dtype=torch.cfloat)
            .view(1, 4, 4)
        )
        assert fourier.fft_shift(x).equal(x_ans)

        # batched
        x = (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.cfloat)
            .view(2, 2, 2)
        )
        x_ans = (
            torch.tensor([3, 2, 1, 0, 7, 6, 5, 4], dtype=torch.cfloat)
            .view(2, 2, 2)
        )
        assert fourier.fft_shift(x).equal(x_ans)


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


if __name__ == "__main__":
    # single batch
    x = (
        torch.tensor([0, 1, 2, 3], dtype=torch.cfloat)
        .view(1, 2, 2)
    )

    x_ans = (
        torch.tensor([3, 2, 1, 0], dtype=torch.cfloat)
        .view(1, 2, 2)
    )
    assert fourier.fft_shift(x).equal(x_ans)