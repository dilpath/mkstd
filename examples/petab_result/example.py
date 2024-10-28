from petab_result_standard import (
    Result,
    Problem,
    Tool,
    MultiStartOptimizeOutput,
    SingleStartOptimizeOutput,
    Component,
    PetabResultYamlStandard,
    PetabResultJsonStandard,
    PetabResultHdf5Standard,
    Author,
    Task,
)

from mkstd.standards.hdf5 import hdfdict


data = {
    "version": "1",
    "license": "MIT",
    "problem": Problem(location="path/to/problem.yaml"),
    "tools": {
        "pyPESTO__0-5": Tool(
            name="pyPESTO",
            author="The pyPESTO developers.",
            version="0.5",
            dependencies=[
                Component(id="Python", version="3.13"),
                Component(id="SciPy", version="2.0.0"),
            ],
            software=[Component(id="OS", version="Ubuntu 24.04")],
            hardware=[Component(id="CPU", brand="Intel", model="Core Ultra 9 185H")],
        )
    },
    "authors": [
        Author(name="Paul Jonas Jost", affiliations=["University of Bonn"]),
        Author(name="Domagoj Doresic", contributions=["optimize1"]),
        Author(name="Dilan Pathirana", other_contributions=["mkstd"]),
    ],
    "tasks": {
        "optimize1": Task(
            date="2024-10-10",
            tool_id="pyPESTO__0-5",
            output=MultiStartOptimizeOutput(
                optimizer=Component(id="pypesto_scipy_lbfgsb", tool="pyPESTO__0-5", settings="SciPy L-BFGS-B optimizer", other="Gradients computed by forward sensitivity analysis with AMICI."),
                starts=[
                    SingleStartOptimizeOutput(id="start1", endpoint=[0.5, 0.2, 0.1], fval=100.0),
                    SingleStartOptimizeOutput(id="start2", endpoint=[0.55, 0.21, 0.1], fval=105.0),
                ],
            )
        )
    }
}


# Create a result object, save it to disk.
result = Result.parse_obj(data)
PetabResultYamlStandard.save_data(data=result, filename="data/result0.yaml")
PetabResultHdf5Standard.save_data(data=result, filename="data/result0.hdf5")
PetabResultJsonStandard.save_data(data=result, filename="data/result0.json")

# Read the stored result from disk, reconstruct the result
loaded_result_yaml = PetabResultYamlStandard.load_data("data/result0.yaml")
loaded_result_hdf5 = PetabResultHdf5Standard.load_data("data/result0.hdf5")

# Write the result to disk again and verify that the round-trip was successful (disk and memory)
PetabResultYamlStandard.save_data(data=loaded_result_yaml, filename="data/result1.yaml")
PetabResultHdf5Standard.save_data(data=loaded_result_hdf5, filename="data/result1.hdf5")

with open("data/result0.yaml") as f:
    data_yaml0 = f.read()
with open("data/result1.yaml") as f:
    data_yaml1 = f.read()
data_hdf50 = hdfdict.load("data/result0.hdf5", lazy=False)
data_hdf51 = hdfdict.load("data/result1.hdf5", lazy=False)

if not data_yaml0 == data_yaml1:
    raise ValueError("The round-trip of saving the PEtab Result to YAML failed.")
if not data_hdf50 == data_hdf51:
    raise ValueError("The round-trip of saving the PEtab Result to HDF5 failed.")

if not result == loaded_result_yaml:
    raise ValueError("The round-trip of reconstructing the PEtab Result from YAML failed.")
if not result == loaded_result_hdf5:
    raise ValueError("The round-trip of reconstructing the PEtab Result from HDF5 failed.")
