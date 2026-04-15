from typing import ClassVar
from pydantic import BaseModel, Field


class BaseRubric(BaseModel):
    metric_name:ClassVar[str]
    metric_type:ClassVar[str]
    required_slide_type:ClassVar[str|None] = None
    is_llm_judge:ClassVar = False
    index_:int | None = Field(default=None)

