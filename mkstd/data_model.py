from enum import Enum

from typing import Annotated

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

import numpy
numpy.string_ = numpy.bytes_

import hdfdict
#from numpydantic import NDArray
import pydantic_numpy
import pydantic_yaml
import pydantic_xmlmodel


NDArray = pydantic_numpy.typing.NpNDArray


class SSRData(BaseModel):
    ssr_level: int = Field(None, ge=0)
    ssr_version: int = Field(None, ge=0)
    #variable_names: list[str]

    simulation_times: NDArray
    sample_size: int = Field(None, ge=1)
    ecf_evals: NDArray
    ecf_tval: NDArray
    ecf_nval: int = Field(None, ge=1)

    error_metric_mean: float
    error_metric_stdev: float = Field(None, ge=0)

    sig_figs: int = Field(None, ge=1)


