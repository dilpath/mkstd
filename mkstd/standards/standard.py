from pathlib import Path
from abc import abstractmethod
from pydantic import BaseModel


class Standard:
    """Base class for file standards."""
    def __init__(self, model: BaseModel):
        self.model = model

    @abstractmethod
    def get_schema(self):
        pass

    @abstractmethod
    def parse_data(self):
        pass

    def save_schema(self, filename: str):
        """Save the model schema to a file."""
        schema = self.get_schema()
        with open(filename, 'w') as f:
            f.write(schema)

    def save_data(self, data: BaseModel, filename: str):
        """Save the data to a file."""
        with open(filename, 'w') as f:
            f.write(self.parse_data(data))

    @abstractmethod
    def load_data(self, filename: str):
        """Load the data from a file."""
        pass
