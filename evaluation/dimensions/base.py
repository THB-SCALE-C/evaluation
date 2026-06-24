from typing import Any, ClassVar
from pydantic import BaseModel, Field


class BaseDimension(BaseModel):
    metric_name:ClassVar[str] = ""
    metric_type:ClassVar[Any|None] = None
    required_slide_type:ClassVar[str|None] = None
    is_llm_judge:ClassVar = False
    index_:int | None = Field(default=None)

