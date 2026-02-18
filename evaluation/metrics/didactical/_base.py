from typing import ClassVar
from evaluation.metrics.base import BaseMetric


class BaseDidacticalMetric(BaseMetric):
    metric_type:ClassVar = "didactical"
    is_llm_judge:ClassVar = True
    