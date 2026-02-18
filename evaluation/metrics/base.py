from typing import ClassVar
from pydantic import BaseModel


class BaseMetric(BaseModel):
    metric_name:ClassVar[str]



    