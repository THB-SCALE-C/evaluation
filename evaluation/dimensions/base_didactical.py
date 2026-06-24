from typing import ClassVar
from evaluation.dimensions.base import BaseDimension


class BaseDidacticalDimension(BaseDimension):
    is_llm_judge:ClassVar = True
    