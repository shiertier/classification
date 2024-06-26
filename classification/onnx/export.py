import io

import numpy as np
import torch
from ditk import logging
from torch import nn

from .onnx import onnx_quick_export
from ..models import load_model_from_ckpt
from ..train import torch_model_profile


class ModelWithSoftMax(nn.Module):
    def __init__(self, model):
        nn.Module.__init__(self)
        self.model = model

    def forward(self, x):
        x = self.model(x)
        x = torch.softmax(x, dim=-1)
        return x


def export_model_to_onnx(model, onnx_filename, opset_version: int = 14, verbose: bool = True,
                         imgsize: int = 384, dynamic: bool = True, no_optimize: bool = False,
                         use_softmax: bool = True):
    logging.info('Preparing input and model ...')
    example_input = torch.randn(1, 3, imgsize, imgsize).float()
    model = model.cpu()
    if use_softmax:
        model = ModelWithSoftMax(model)
    model = model.float()
    model.eval()
    torch_model_profile(model, example_input)

    logging.info(f'Start exporting to {onnx_filename!r}')
    onnx_quick_export(
        model, example_input, onnx_filename,
        opset_version, verbose, no_optimize,
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        } if dynamic else {
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
        no_gpu=True,
    )


def export_onnx_from_ckpt(model_filename, onnx_filename, opset_version: int = 14, verbose: bool = True,
                          imgsize: int = 384, dynamic: bool = True, no_optimize: bool = False,
                          use_softmax: bool = True):
    model = load_model_from_ckpt(model_filename)
    _arguments = model.__arguments__
    model_name = _arguments['name']
    labels = _arguments['labels']

    _info = model.__info__.copy()
    step = _info.pop('step')
    metrics = {
        key: value for key, value in _info.items()
        if isinstance(value, (int, float)) or (isinstance(value, (torch.Tensor, np.ndarray)) and not value.shape)
    }

    logging.info(f'Load from best ckeckpoint {model_filename!r}, with step {step!r}, metrics: {metrics!r}.')
    logging.info(f'Model name: {model_name!r}, labels: {labels!r}')
    try:
        export_model_to_onnx(model, onnx_filename, opset_version, verbose, imgsize, dynamic, no_optimize, use_softmax)
    except:
        logging.error('Export failed!')
        raise
    else:
        logging.info('[green]Export Success![/green]')


def validate_onnx_model(onnx_filename):
    logging.info(f'Validating the exported onnx file {onnx_filename!r} ...')
    import onnxruntime as ort
    model = ort.InferenceSession(onnx_filename, providers=["CPUExecutionProvider"])

    with io.StringIO() as sf:
        print('Input items:', file=sf)
        for item in model.get_inputs():
            print(f'    {item.name}: {item.type}{item.shape}', file=sf)
        logging.info(sf.getvalue().rstrip())

    with io.StringIO() as sf:
        print('Output items:', file=sf)
        for item in model.get_outputs():
            print(f'    {item.name}: {item.type}{item.shape}', file=sf)
        logging.info(sf.getvalue().rstrip())
