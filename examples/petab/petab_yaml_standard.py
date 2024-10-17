from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Extra, Field, StringConstraints

from mkstd import YamlStandard

FileList = list[str]
PetabId = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z_]\\w*$")]
Version = Annotated[
    str,
    StringConstraints(
        pattern=r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$"
    ),
]


class ModelFile(BaseModel):
    """Define the file location and file format of a model."""

    filename: str
    language: Literal["sbml"] | Literal["cellml"] | Literal["bngl"] | Literal[
        "pysb"
    ]


class ProblemYaml(BaseModel):
    """Define the files for a single parameter estimation problem."""

    model_files: dict[PetabId, ModelFile]
    measurement_files: FileList
    condition_files: FileList
    observable_files: FileList
    visualization_files: FileList = Field(default=None)
    mapping_file: str = Field(default=None)


class ExtensionYaml(BaseModel, extra=Extra.allow):
    """Define a PEtab extension."""

    version: Version


class PetabYaml(BaseModel):
    """PEtab parameter estimation problem config file schema."""

    format_version: Version
    parameter_file: str | FileList
    problems: list[ProblemYaml]
    extensions: dict[PetabId, ExtensionYaml] = Field(default=None)


PetabYamlStandard = YamlStandard(model=PetabYaml)

if __name__ == "__main__":
    PetabYamlStandard.save_schema("standard/schema.yaml")
