import torch.nn as nn
from petab_sciml_standard import Input, MLModel, PetabScimlStandard


class Net(nn.Module):
    """Example pytorch module."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flat2 = nn.Flatten(start_dim=1)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()

        self.fc3 = nn.Linear(84, 10)


# Create a pytorch module, convert it to PEtab SciML, then save it to disk.
net0 = Net()
mlmodel0 = MLModel.from_pytorch_module(
    module=net0, mlmodel_id="model1", inputs=[Input(input_id="input1")]
)
petab_sciml_models0 = PetabScimlStandard.model(models=[mlmodel0])
PetabScimlStandard.save_data(
    data=petab_sciml_models0, filename="output/models0.yaml"
)

# Read the stored model from disk, reconstruct the pytorch module
loaded_petab_sciml_models = PetabScimlStandard.load_data("output/models0.yaml")
net1 = loaded_petab_sciml_models.models[0].to_pytorch_module()

# Store the pytorch module to disk again and verify that the round-trip was successful
mlmodel1 = MLModel.from_pytorch_module(
    module=net1, mlmodel_id="model1", inputs=[Input(input_id="input1")]
)
petab_sciml_models1 = PetabScimlStandard.model(models=[mlmodel1])
PetabScimlStandard.save_data(
    data=petab_sciml_models1, filename="output/models1.yaml"
)

with open("output/models0.yaml") as f:
    data0 = f.read()
with open("output/models1.yaml") as f:
    data1 = f.read()


if not data0 == data1:
    raise ValueError(
        "The round-trip of saving the pytorch modules to disk failed."
    )
