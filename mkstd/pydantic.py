from typing import Annotated
from pydantic import BeforeValidator, PlainSerializer
import numpy as np
from io import StringIO
import pydantic_numpy

def NumpySerializer(array: np.ndarray):
    io_array_string = StringIO()
    np.savetxt(io_array_string, array)
    return io_array_string.getvalue()


def NumpyDeserializer(string: str | np.ndarray):
    if not isinstance(string, str):
        return string

    io_array_string = StringIO(string)
    array = np.loadtxt(io_array_string)
    return array


NumpyArray = Annotated[
    pydantic_numpy.typing.NpNDArray,  # FIXME just use np.ndarray now?
    BeforeValidator(NumpyDeserializer),
    PlainSerializer(NumpySerializer, return_type=str),
]
