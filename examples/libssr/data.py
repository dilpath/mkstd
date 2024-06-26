from pydantic import BaseModel, Field

from mkstd.pydantic import NumpyArray
from mkstd.standards import XmlStandard

import numpy as np


class SSRData(BaseModel):
    ssr_level: int = Field(None, ge=0)
    ssr_version: int = Field(None, ge=0)
    #variable_names: list[str]

    simulation_times: NumpyArray
    sample_size: int = Field(None, ge=1)
    ecf_evals: NumpyArray
    ecf_tval: NumpyArray
    ecf_nval: int = Field(None, ge=1)

    error_metric_mean: float
    error_metric_stdev: float = Field(None, ge=0)

    sig_figs: int = Field(None, ge=1)

xml_standard = XmlStandard(model=SSRData)
xml_standard.save_schema("test.xsd")


data = SSRData.parse_obj({
    "ssr_level": 1,
    "ssr_version": 3,
    #"variable_names": ["v1", "v2", "v3"],
    "simulation_times": np.linspace(0, 10, 11),
    "sample_size": 100000,
    "ecf_evals": np.eye(5),
    "ecf_tval": np.eye(3),
    "ecf_nval": 1000,
    "error_metric_mean": 0.5,
    "error_metric_stdev": 0.2,
    "sig_figs": 5,
})

xml_standard.save_data(data=data, filename="test.xml")
test_data = xml_standard.load_data(filename="test.xml")
