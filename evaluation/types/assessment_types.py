from typing import ClassVar, Literal, Optional
from pydantic import BaseModel, Field


class BaseMetricType(BaseModel):
    type: ClassVar[str]
    scale: ClassVar[tuple]
    max: ClassVar[int]
    min: ClassVar[int]
    criterion:Optional[str] = None
    score: int | float | Literal["yes", "no"]
    feedback: str = Field(
        description="Give detailed feedback on how you made your scoring-decision. Provide details on what justifies your decision.")


class BinaryMetricType(BaseMetricType):
    type: ClassVar[str] = "binary"
    scale: ClassVar[tuple] = (0, 1)
    max: ClassVar[int] = 1# type: ignore
    min: ClassVar[int] = 0# type: ignore
    score: Literal[0, 1]  # type: ignore
    feedback: str = Field(
        description="Give detailed feedback on how you made your scoring-decision. Provide details on what justifies your decision.")


class LikertMetricType(BaseMetricType):
    type: ClassVar[str] = "likert"
    scale: ClassVar[tuple] = (1,2,3,4,5)
    max: ClassVar[int] = 5# type: ignore
    min: ClassVar[int] = 1# type: ignore
    score: Literal[1,2,3,4,5]  # type: ignore
    feedback: str = Field(
        description="Give detailed feedback on how you made your scoring-decision. Provide details on what justifies your decision.")
