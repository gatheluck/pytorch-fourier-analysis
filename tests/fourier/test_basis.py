import torchvision

import pytorch_fourier_analysis


class TestCreateFourierBasisGrid:
    def test_generate_sample_result(self):
        grid_size = 16
        image_size = 32
        list_basis = pytorch_fourier_analysis.fourier.basis.basis_grid(
            grid_size, image_size
        )
        torchvision.utils.save_image(list_basis, "logs/grid.png", nrow=grid_size)
