from typing import ClassVar
from pydantic import BaseModel


class BaseMetric(BaseModel):
    metric_name:ClassVar[str]
    metric_type:ClassVar[str]
    required_slide_type:ClassVar[str|None] = None
    is_llm_judge:ClassVar = False
    slide_idx:int|None = None


    