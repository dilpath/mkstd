from petab_result_standard import Result, Problem, Tool, MultiStartOptimizeOutput, SingleStartOptimizeOutput, Component, PetabResultYamlStandard, Author, Task


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

# Read the stored result from disk, reconstruct the result
loaded_result = PetabResultYamlStandard.load_data("data/result0.yaml")

# Write the result to disk again and verify that the round-trip was successful (disk and memory)
PetabResultYamlStandard.save_data(data=loaded_result, filename="data/result1.yaml")

with open("data/result0.yaml") as f:
    data0 = f.read()
with open("data/result1.yaml") as f:
    data1 = f.read()

if not data0 == data1:
    raise ValueError("The round-trip of saving the PEtab Result to disk failed.")

if not result == loaded_result:
    raise ValueError("The round-trip of reconstructing the PEtab Result failed.")
