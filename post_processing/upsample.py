import torch

from typing import Dict, Any, Union


class TorchInterpolation():
    '''
    Performs interpolation on given tensor, will attempt to use GPU
    Fixed on trilinear interpolation for now
    '''
    def __init__(self, factor: int, order: int):
        self.factor = factor
        self.order = order

    def __call__(self, output: torch.Tensor, metadata: Dict[str, Any]) -> Union[torch.Tensor, Dict[str, Any]]:
        inpt_device = output.device

        nunsqueezes = 0
        while len(output.shape) < 5:
            output = output.unsqueeze(0)
            nunsqueezes += 1
        
        with torch.no_grad():
            if torch.cuda.is_available:
                output = output.cuda()

            output = torch.nn.functional.interpolate(output, scale_factor=(1, self.factor, self.factor), mode="trilinear", align_corners=True)
        
        output = output.to(inpt_device)

        for _ in range(nunsqueezes):
            output = output.squeeze(0)

        return output, metadata
