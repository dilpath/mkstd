import json
from typing import Any

from pydantic import BaseModel

from .hdfdict import hdfdict
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
        escape_keys:
            %-encode group names so that e.g. if a name contains `/` then it is
            not interpreted as a nested group.
        lazy:
            Whether to read on access (`True`) or at load time (`False`).
    """

    default_dump_kwargs = {"indent": 4}

    def __init__(
        self,
        *args,
        dump_kwargs: dict[str, Any] = None,
        escape_keys: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.dump_kwargs = Hdf5Standard.default_dump_kwargs
        if dump_kwargs is not None:
            self.dump_kwargs = dump_kwargs
        self.escape_keys = escape_keys
        self.lazy = lazy

    def get_schema(self) -> str:
        """See :class:`Standard`."""
        return json.dumps(
            self.model.model_json_schema(mode="serialization"),
            **self.dump_kwargs,
        )

    def format_data(self, data: BaseModel) -> str:
        """See :class:`Standard`."""
        return data.model_dump()

    def save_data(self, data: BaseModel, filename: str) -> None:
        """See :class:`Standard`."""
        hdfdict.dump(
            self.format_data(data), filename, escape_keys=self.escape_keys
        )

    def _load_data(self, filename: str) -> dict[str, Any]:
        """See :class:`Standard`."""
        data = hdfdict.load(
            filename, lazy=self.lazy, escape_keys=self.escape_keys
        )
        return data
