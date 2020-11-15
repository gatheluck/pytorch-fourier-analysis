from .base import NoiseAugmentationBase
from .gaussian import (
    Gaussian,
    PatchGaussian,
    get_gaussian_noise,
    BandpassGaussian,
    BandpassPatchGaussian,
)
from .fourier import Fourier, PatchFourier
