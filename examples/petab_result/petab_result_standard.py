from __future__ import annotations

from pydantic import BaseModel, Field

from mkstd import YamlStandard, Hdf5Standard, JsonStandard, XmlStandard

from _outputs import ProfileOutput, McmcSampleOutput, OptimizeOutput, GenericOutput, Output, MultiStartOptimizeOutput, SingleStartOptimizeOutput


class Component(BaseModel, extra="allow"):
    """The state of a single component.

    e.g. a system package (id/version), a tool (id/version), or a method (name, hyperparameters)
    """

    id: str
    version: str | None = Field(default=None)
    value: str | None = Field(default=None)


class Tool(BaseModel, extra="allow"):
    """A tool used to create a result."""
    
    name: str
    """The full tool name (not necessarily a valid PEtab ID."""
    author: str
    """The author/owner of the tool."""
    version: str
    dependencies: list[Component]
    """Significant dependencies required to reproduce a result with a tool."""
    software: list[Component]
    """Other information on software used to create a result, less critical for reproducibility.
    e.g. operating system
    """
    hardware: list[Component]


class Problem(BaseModel):
    """The PEtab problem."""
    location: str
    """FIXME currently just the location of the problem. Eventually embed the full PEtab problem (or just the location, for very large problems)"""


class Author(BaseModel):
    """The author of a result."""
    name: str
    affiliations: list[str] | None = Field(default=None)
    contributions: list[str] | None = Field(default=None)
    """The IDs of the tasks this author contributed to."""
    other_contributions: list[str] | None = Field(default=None)
    funding_sources: list[str] | None = Field(default=None)


class Task(BaseModel):
    """A task result."""
    date: str
    tool_id: str
    """Corresponds to a `Result.tools` entry."""
    output: ProfileOutput | McmcSampleOutput | SingleStartOptimizeOutput | MultiStartOptimizeOutput | None = Field(default=None)
    """The output from a specific task, in a format defined in PEtab Result."""
    other_output: GenericOutput | None = Field(default=None)
    """Other output, not (yet) defined in PEtab Result."""


class Result(BaseModel):
    """Specify the result structure."""
    version: str
    """The PEtab Result version."""
    license: str
    problem: dict[str, Problem] | Problem
    tools: dict[str, Tool]
    """Keys are user-defined IDs for the tool (must be a valid PEtab ID)"""
    authors: list[Author] | Author
    tasks: dict[str, Task]
    """Keys are user-defined IDs for the tasks (must be a valid PEtab ID)"""

PetabResultHdf5Standard = Hdf5Standard(model=Result)
PetabResultJsonStandard = JsonStandard(model=Result)
#PetabResultXmlStandard = XmlStandard(model=Result)
PetabResultYamlStandard = YamlStandard(model=Result)

if __name__ == "__main__":
    PetabResultHdf5Standard.save_schema("standard/schema.hdf5")
    PetabResultJsonStandard.save_schema("standard/schema.json")
    #PetabResultXmlStandard.save_schema("standard/schema.xml")
    PetabResultYamlStandard.save_schema("standard/schema.yaml")
