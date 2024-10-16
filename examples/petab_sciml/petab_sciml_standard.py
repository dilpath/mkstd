from __future__ import annotations

import inspect

import torch.nn as nn
from pydantic import BaseModel, Field

from mkstd import YamlStandard


class Input(BaseModel):
    """Specify (transformations of) the input layer."""

    input_id: str
    transform: dict | None = Field(
        default=None
    )  # TODO class of supported transforms


class Layer(BaseModel):
    """Specify layers."""

    layer_id: str
    layer_type: str
    args: dict | None = Field(
        default=None
    )  # TODO class of layer-specific supported args


class MLModel(BaseModel):
    """An easy-to-use format to specify simple deep ML models.

    There is a function to export this to a PyTorch module, or to YAML.
    """

    mlmodel_id: str

    inputs: list[Input]

    layers: list[Layer]
    """The components of the model (e.g., layers of a neural network)."""

    @staticmethod
    def from_pytorch_module(
        module: nn.Module, mlmodel_id: str, inputs: list[Input]
    ) -> MLModel:
        """Create a PEtab SciML ML model from a pytorch module."""
        layers = []
        for layer_id, layer_module in module.named_modules():
            if not layer_id:
                # first entry is all modules combined
                continue
            supported_args = [
                arg
                for arg in layer_module.__constants__
                if arg
                in set(inspect.signature(layer_module.__init__).parameters)
            ]
            layer = Layer(
                layer_id=layer_id,
                layer_type=type(layer_module).__name__,
                args={
                    arg: getattr(layer_module, arg) for arg in supported_args
                },
            )
            layers.append(layer)

        mlmodel = MLModel(mlmodel_id=mlmodel_id, inputs=inputs, layers=layers)
        return mlmodel

    def to_pytorch_module(self) -> nn.Module:
        """Create a pytorch module from a PEtab SciML ML model."""
        self2 = self

        class _PytorchModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                for layer in self2.layers:
                    setattr(
                        self,
                        layer.layer_id,
                        getattr(nn, layer.layer_type)(**layer.args),
                    )

        return _PytorchModule()


class MLModels(BaseModel):
    """Specify all ML models of your hybrid model."""

    models: list[MLModel]


PetabScimlStandard = YamlStandard(model=MLModels)

if __name__ == "__main__":
    PetabScimlStandard.save_schema("output/schema.json")
