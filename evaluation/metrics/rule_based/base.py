from typing import ClassVar
from evaluation.metrics.base import BaseMetric



class BaseRuleMetric(BaseMetric):
    metric_name:ClassVar[str]


