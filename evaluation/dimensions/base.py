from typing import Any, ClassVar
from pydantic import BaseModel, Field, PrivateAttr, computed_field


class BaseDimension(BaseModel):
    metric_name:ClassVar[str] = ""
    metric_type:ClassVar[Any|None] = None
    required_slide_type:ClassVar[str|None] = None
    is_llm_judge:ClassVar = False
    _index:int | None = PrivateAttr(default=None)

    @computed_field
    @property
    def index(self)->int|None:
        return self._index

