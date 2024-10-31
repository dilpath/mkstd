import inspect
import typing
from functools import reduce
from operator import mul
from typing import Annotated, Any, Iterable

import numpy as np
from pydantic import (
    BaseModel,
    BeforeValidator,
    PlainSerializer,
    WithJsonSchema,
)
from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation

DTYPE = "dtype"
FORMAT = "fmt"
ORDER = "C"


SHAPE_ARRAY_DELIMITER = "#"
FORMATS = {
    str: "%s",
    float: "%.18e",
}

DELIMITER = ","


def get_dtype(field_name: str, model: BaseModel) -> type:
    """Get the data type of elements in an array field of a model.

    Args:
        field_name:
            The name of the array field of the model.
        model:
            The model.

    Returns:
        The data type.
    """
    field_type = model.model_fields[field_name].annotation
    if field_type in (list[str],):
        return str
    elif field_type.__name__ == "ndarray" and issubclass(
        field_type.__args__[1].__args__[0], np.floating
    ):
        return float
    raise NotImplementedError(f"Field type: `{field_type}`.")


def array_to_string(
    field_name: str, array: Iterable[Any], model: type[BaseModel]
) -> str:
    """Convert the value of an array field from an array to a string.

    Args:
        field_name:
            The name of the field.
        array:
            The field value.
        model:
            The data model.

    Returns:
        The string.
    """
    dtype = get_dtype(field_name=field_name, model=model)
    array = np.asarray(array)

    # Convert array to "shape array" string.
    shape = [str(s) for s in array.shape]
    array = [
        *shape,
        SHAPE_ARRAY_DELIMITER,
        *[FORMATS[dtype] % value for value in array.flatten(order=ORDER)],
    ]

    return DELIMITER.join(array)


def string_to_array(
    field_name: str, array: str, model: type[BaseModel]
) -> np.ndarray:
    """Convert the value of an array field from a string to an array.

    Args:
        field_name:
            The name of the field.
        array:
            The field value, in the format output by :func:`to_string`.
        model:
            The data model.

    Returns:
        The array.
    """
    dtype = get_dtype(field_name=field_name, model=model)

    # Convert "shape array" string to array.
    array_str = array.split(DELIMITER)
    shape = []
    for start_index, value_str in enumerate(array_str):  # noqa: B007
        if value_str == SHAPE_ARRAY_DELIMITER:
            break
        shape.append(int(value_str))
    n_values = reduce(mul, shape, 1)
    array = np.array(
        array_str[start_index + 1 : start_index + 1 + n_values], dtype=dtype
    ).reshape(shape)

    return array


def array_to_list(array: np.ndarray) -> list:
    """Convert an array to a (nested) list.

    Args:
        array:
            The array,

    Returns:
        The (nested) list.
    """
    return array.tolist()


def list_to_array(list_: list) -> np.ndarray:
    """Convert a (nested) list to an array.

    Args:
        list_:
            The list,

    Returns:
        The array.
    """
    return np.array(list_, dtype=float)


# TODO array equality checks for model0 == model1 checks


def get_array_type(
    dtype: type, dimensions: int, strict_dtype: bool = False
) -> type:
    """Get a customized array type.

    Args:
        dtype:
            The type of the individual array elements.
        dimensions:
            The number of dimensions.
        strict_dtype:
            If `True`, no type conversions will be performed.

    Returns:
        The customized array type.
    """
    return Annotated[
        np.typing.NDArray[dtype],
        NpArrayPydanticAnnotation.factory(
            data_type=dtype,
            dimensions=dimensions,
            strict_data_typing=strict_dtype,
        ),
        BeforeValidator(list_to_array),
        PlainSerializer(array_to_list, return_type=list),
        WithJsonSchema({"type": "array"}, mode="serialization"),
    ]


def is_array_type(type_: type) -> bool:
    """Identify arrays.

    Args:
        type_:
            The type of the object.

    Returns:
        `True` if `type_` is consistent with the return value of
        :func:`get_array_type`, else `False`.
    """
    if (
        inspect.isclass(typing.get_origin(type_))
        and type_.__name__ == "ndarray"
    ):
        return True
    return False
