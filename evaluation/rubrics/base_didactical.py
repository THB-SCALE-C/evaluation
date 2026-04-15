from typing import ClassVar
from evaluation.rubrics.base import BaseRubric


class BaseDidacticalRubric(BaseRubric):
    metric_type:ClassVar = "didactical"
    is_llm_judge:ClassVar = True
    