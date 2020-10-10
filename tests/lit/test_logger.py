import os
import sys
from torch import Tensor

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pytorch_fourier_analysis import lit


class TestGetEpochEndLog:
    def test_valid_input(self):
        # test unnested output
        outputs = [
            dict(err=Tensor([0.0])),
            dict(err=Tensor([1.0])),
            dict(err=Tensor([2.0])),
        ]
        assert lit.get_epoch_end_log(outputs) == dict(err_avg=Tensor([1.0]))

        # test nested output
        outputs = [
            [dict(err=Tensor([0.0])), dict(err=Tensor([1.0])), dict(err=Tensor([2.0]))],
            [dict(err=Tensor([3.0])), dict(err=Tensor([4.0])), dict(err=Tensor([5.0]))],
            [dict(err=Tensor([6.0])), dict(err=Tensor([7.0])), dict(err=Tensor([8.0]))],
        ]
        assert lit.get_epoch_end_log(outputs) == dict(err_avg=Tensor([4.0]))
