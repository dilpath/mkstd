from typing import Annotated
from pathlib import Path as _Path

from pydantic import PlainSerializer


__all__ = ["Path"]

Path = Annotated[_Path, PlainSerializer(str, return_type=str)]
