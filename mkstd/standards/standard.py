from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, RootModel


class Standard(ABC):
    """Base class for file standards.

    A standard is data in a specific file format according to the specific
    schema. To create a standard, create a data model as a subclass of
    :class:`pydantic.BaseModel`, then choose one of the specific file format
    `Standard` subclasses, which can automatically generate a schema for the
    file format and facilitate reading/writing/validation of data in the file
    format.

    Attributes:
        model:
            The data model.
    """

    def __init__(self, model: type[BaseModel]) -> None:
        self.model = model

    @abstractmethod
    def get_schema(self) -> str:
        """Get the schema for the data model.

        Returns:
            The schema, as a string.
        """
        pass

    @abstractmethod
    def format_data(self, data: BaseModel) -> str:
        """Convert data from the data model to the file format.

        Args:
            data:
                The data, as an instance of the data model.

        Returns:
            The data, as the file format.
        """
        pass

    def save_schema(self, filename: str) -> None:
        """Save the schema for the data model to a file.

        Args:
            filename:
                The location where the schema will be stored.
        """
        schema = self.get_schema()
        if schema[-1] != "\n":
            schema += "\n"
        with open(filename, "w") as f:
            f.write(schema)

    def save_data(self, data: BaseModel, filename: str) -> None:
        """Save data as the file format.

        Args:
            data:
                The data, as an instance of the data model.
            filename:
                The location where the data will be stored.
        """
        content = self.format_data(data)
        if content[-1] != "\n":
            content += "\n"
        with open(filename, "w") as f:
            f.write(content)

    @abstractmethod
    def _load_data(self, filename: str) -> dict[str, Any]:
        """Load data from the file format into a dictionary."""
        pass

    def load_data(self, filename: str, **kwargs: dict[str, Any]) -> BaseModel:
        """Load data from the file format.

        Args:
            filename:
                The location where the data is stored.
            **kwargs:
                Private instance attributes.

        Returns:
            The data, as an instance of the data model.
        """
        data = self._load_data(filename=filename)
        if issubclass(self.model, RootModel):
            data = {"root": data}
        return self.model.model_validate({**data, **kwargs})
