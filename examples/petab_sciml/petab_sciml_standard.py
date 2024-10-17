from __future__ import annotations

import inspect

import torch.fx
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
    # FIXME currently handled as kwargs
    args: dict | None = Field(
        default=None
    )  # TODO class of layer-specific supported args


class Node(BaseModel):
    """A node of the computational graph.

    e.g. a node in the forward call of a PyTorch model.
    Ref: https://pytorch.org/docs/stable/fx.html#torch.fx.Node
    """

    name: str
    op: str
    target: str
    args: list | None = Field(default=None)
    kwargs: dict | None = Field(default=None)


class MLModel(BaseModel):
    """An easy-to-use format to specify simple deep ML models.

    There is a function to export this to a PyTorch module, or to YAML.
    """

    mlmodel_id: str

    inputs: list[Input]

    layers: list[Layer]
    """The components of the model (e.g., layers of a neural network)."""

    forward: list[Node]

    @staticmethod
    def from_pytorch_module(
        module: nn.Module, mlmodel_id: str, inputs: list[Input]
    ) -> MLModel:
        """Create a PEtab SciML ML model from a pytorch module."""
        layers = []
        layer_ids = []
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
            layer_ids.append(layer_id)

        nodes = []
        node_names = []
        pytorch_nodes = list(torch.fx.symbolic_trace(module).graph.nodes)
        for pytorch_node in pytorch_nodes:
            op = pytorch_node.op
            target = pytorch_node.target
            if op == "call_function":
                target = pytorch_node.target.__name__
            node = Node(
                name=pytorch_node.name,
                op=pytorch_node.op,
                target=target,
                args=[
                    (arg if arg not in pytorch_nodes else str(arg))
                    for arg in pytorch_node.args
                ],
                kwargs=pytorch_node.kwargs,
            )
            nodes.append(node)
            node_names.append(node.name)

        mlmodel = MLModel(
            mlmodel_id=mlmodel_id, inputs=inputs, layers=layers, forward=nodes
        )
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

        graph = torch.fx.Graph()
        state = {}
        for node in self.forward:
            args = []
            if node.args:
                for node_arg in node.args:
                    arg = node_arg
                    try:
                        if arg in state:
                            arg = state[arg]
                    except TypeError:
                        pass
                    args.append(arg)
            args = tuple(args)
            kwargs = {}
            if node.kwargs:
                kwargs = {k: state.get(v, v) for k, v in node.kwargs.items()}
            match node.op:
                case "placeholder":
                    state[node.name] = graph.placeholder(node.target)
                case "call_function":
                    if node.target in ["flatten"]:
                        function = getattr(torch, node.target)
                    else:
                        function = getattr(nn.functional, node.target)
                    state[node.name] = graph.call_function(
                        function, args, kwargs
                    )
                case "call_module":
                    state[node.name] = graph.call_module(
                        node.target, args, kwargs
                    )
                case "output":
                    graph.output(args[0])

        return torch.fx.GraphModule(_PytorchModule(), graph)


class MLModels(BaseModel):
    """Specify all ML models of your hybrid model."""

    models: list[MLModel]


PetabScimlStandard = YamlStandard(model=MLModels)

if __name__ == "__main__":
    PetabScimlStandard.save_schema("standard/schema.yaml")
