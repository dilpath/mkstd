import json
from typing import Any

from pydantic import BaseModel

from .standard import Standard


class JsonStandard(Standard):
    """The JSON standard.

    Use this to create a standard in the JSON file format.

    See :class:`Standard` for inherited methods and attributes.

    Schema are generated in the JSON schema format.
    https://en.wikipedia.org/wiki/JSON#Metadata_and_schema

    Attributes:
        dump_kwargs:
            Keyword arguments that will be passed to `json.dumps`. Defaults to
            setting the indentation of generated schema files to 4 spaces.
    """

    default_dump_kwargs = {"indent": 4}

    def __init__(
        self, *args, dump_kwargs: dict[str, Any] = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.dump_kwargs = self.default_dump_kwargs
        if dump_kwargs is not None:
            self.dump_kwargs = dump_kwargs

    def get_schema(self, to_json: bool = True) -> str | dict:
        """See :class:`Standard`.

        Args:
            to_json:
                Whether to convert the schema `dict` to a JSON string.
        """
        schema = self.model.model_json_schema(mode="serialization")
        if to_json:
            return json.dumps(schema, **self.dump_kwargs)
        return schema

    def format_data(self, data: BaseModel) -> str:
        """See :class:`Standard`."""
        return data.model_dump_json(**self.dump_kwargs)

    def _load_data(self, filename: str) -> dict[str, Any]:
        """See :class:`Standard`."""
        with open(filename) as f:
            data = json.load(f)
        return data
