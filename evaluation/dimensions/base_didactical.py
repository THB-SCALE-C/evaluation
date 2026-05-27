from typing import ClassVar
from evaluation.dimensions.base import BaseDimension


class BaseDidacticalDimension(BaseDimension):
    metric_type:ClassVar = "didactical"
    is_llm_judge:ClassVar = True
    