from typing import Any
import json
from pydantic import BaseModel

# TODO quickfix for hdfdict
import numpy as np
np.string_ = np.bytes_

import hdfdict

from .standard import Standard


class Hdf5Standard(Standard):
    """The HDF5 standard.

    Use this to create a standard in the HDF5 file format.

    See :class:`Standard` for inherited methods and attributes.
    
    N.B.: Schema are generated in the JSON schema format.
    https://en.wikipedia.org/wiki/JSON#Metadata_and_schema

    Attributes:
        dump_kwargs:
            Keyword arguments that will be passed to `json.dumps`. Defaults to
            setting the indentation of generated schema files to 4 spaces.
    """
    default_dump_kwargs = {"indent": 4}

    def __init__(self, *args, dump_kwargs: dict[str, Any] = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.dump_kwargs = Hdf5Standard.default_dump_kwargs
        if dump_kwargs is not None:
            self.dump_kwargs = dump_kwargs

    def get_schema(self):
        return json.dumps(
            self.model.model_json_schema(mode="serialization"),
            **self.dump_kwargs,
        )

    def format_data(self, data: BaseModel) -> str:
        return data.model_dump()

    def save_data(self, data: BaseModel, filename: str):
        hdfdict.dump(self.format_data(data), filename)

    def load_data(self, filename: str):
        data = hdfdict.load(filename)
        return self.model.parse_obj(data)

