from __future__ import annotations


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from petab_result_standard import Component

from pydantic import BaseModel, Field


class Output(BaseModel):
    """The base class for an output of a computation."""

class GenericOutput(Output, extra="allow"):
    """The output of a computation."""


class OptimizeOutput(Output):
    """The result of a single optimization."""

    id: str = Field(
        description="Unique identifier for the optimization result. "
        "Multi-start local optimizations should be separated for "
        "each startpoint."
    )
    optimizer: Component = Field(
        description="Metadata on the optimizer used."
    )
    startpoint: list[float] | list[list[float]] = Field(
        description="Starting point(s) for the optimization. May be multiple in"
        "case of swarm based optimizers. Dimension: (n_parameters) | (n_starts, n_parameters)"
    )
    endpoint: list[float] = Field(
        description="End point of the "
        "optimization. Dimension: (n_parameters)"
    )
    fval: float = Field(description="Final value of the objective function.")
    fval0: float | None = Field(
        None, description="Initial value of the objective function."
    )
    grad: list[float] | None = Field(
        None, description="Gradient at the endpoint. Dimension: (n_parameters)"
    )
    hess: list[list[float]] | None = Field(
        None,
        description="Hessian at the endpoint. Dimension: (n_parameters, n_parameters)",
    )
    history: str | None = Field(
        None,
        description="Link to file of History of the optimization process",
        # FIXME should actually be the file itself, embedded here, as an `OptimizeHistoryOutput` object...
    )


class IndividualProfileOutput(Output):
    """The profile result for a single parameter."""
    parameter_trace: list[list[float]] | None = Field(
        None, description="Dimension: (trace_length, n_parameters)"
    )
    objective_trace: list[float] | None = Field(
        None, description="Dimension: (trace_length)"
    )
    confidence_interval: tuple[float, float] | None = Field(
        None, description="Confidence interval for the profile"
    )


class ProfileOutput(Output):
    method: Component
    settings: list[Component]
    startpoint: list[float] | None = Field(
        None, description="Parameter vector from which the profile was started"
    )
    confidence_level: float | None = Field(
        None, description="Confidence level for the profile"
    )
    profiles: dict[str, IndividualProfileOutput]
    """The keys are the PEtab IDs of the parameters."""


class McmcSampleOutput(Output):
    """Sampling result for a single sampling execution with one or more chains.

    TODO split into separate types: `SingleChainMcmcOutput` and `MultiChainMcmcOutput`
    """

    sampler: Component
    settings: list[Component] = Field(description="Settings for the sampler")
    startpoints: list[list[float]] | list[float] | None = Field(
        None,
        description="Dimension: (n_chains, n_parameters) or (n_parameters)",
    )
    samples: list[list[list[float]]] | list[list[float]] | None = Field(
        None,
        description="Dimension: (n_chains, n_samples, n_parameters) or (n_samples, n_parameters)",
    )
    parameter_trace: list[list[list[float]]] | list[list[float]] | None = Field(
        None,
        description="Dimension (n_chains, trace_length, n_parameters) or (trace_length, n_parameters)",
    )
    log_posterior_trace: list[list[float]] | list[float] | None = Field(
        None,
        description="Dimension: (n_chains, trace_length) or (trace_length)",
    )
    number_of_chains: int | None = Field(
        None, description="Number of chains used in the sampling"
    )
    burn_in: int | None = Field(
        None, description="Number of burn-in samples"
    )
    thinning: int | None = Field(None, description="Thinning factor")
