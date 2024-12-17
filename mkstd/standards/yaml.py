from typing import Any

import oyaml as yaml
from pydantic import BaseModel

from .json import JsonStandard


class YamlStandard(JsonStandard):
    """The YAML standard.

    Use this to create a standard in the YAML file format.

    See :class:`Standard` for inherited methods and attributes.

    N.B.: Schema are generated in the JSON schema format.
    https://en.wikipedia.org/wiki/JSON#Metadata_and_schema

    Attributes:
        dump_kwargs:
            Keyword arguments that will be passed to `json.dumps` and
            `yaml.dump`. Defaults to setting the indentation of generated
            schema and data files to 4 spaces.
    """

    default_dump_kwargs = {"indent": 2}

    def format_data(self, data: BaseModel) -> str:
        """See :class:`Standard`."""
        # TODO handle custom `model_dump` kwargs?
        return yaml.safe_dump(data.model_dump(), **self.dump_kwargs)

    def get_schema(self) -> str:
        """See :class:`Standard`."""
        schema = super().get_schema(to_json=False)
        return yaml.safe_dump(schema, **self.dump_kwargs)

    def _load_data(self, filename: str) -> dict[str, Any]:
        """See :class:`Standard`."""
        with open(filename) as f:
            data = yaml.safe_load(f)
        return data
