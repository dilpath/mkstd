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

    def to_hdf5(self, filename: str):
        data = dict(self)
        hdfdict.dump(data, filename)

    @staticmethod
    def from_hdf5(filename: str):
        data = hdfdict.load(filename)
        return SSRData.parse_obj(data)

    def to_json(self, filename: str):
        data = self.model_dump_json(indent=2)
        with open(filename, 'w') as f:
            f.write(data)

    @staticmethod
    def from_json(filename: str):
        with open(filename, 'r') as f:
            data = f.read()
        return SSRData.model_validate_json(data)

    def to_yaml(self, filename: str):
        data = pydantic_yaml.to_yaml_str(self)
        with open(filename, 'w') as f:
            f.write(data)

    @staticmethod
    def from_yaml(filename: str):
        with open(filename, 'r') as f:
            data = f.read()
        return pydantic_yaml.parse_yaml_raw_as(SSRData, data)

    def to_xml(self, filename: str):
        data = pydantic_xmlmodel.model_dump_xml(self)
        with open(filename, 'w') as f:
            f.write(data)

    @staticmethod
    def from_xml(filename: str):
        with open(filename, 'r') as f:
            data = f.read()
        return pydantic_xmlmodel.model_validate_xml(SSRData, data)

