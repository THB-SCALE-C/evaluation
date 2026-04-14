from typing import ClassVar, Literal, Optional
from pydantic import BaseModel, Field


class BaseAssessment(BaseModel):
    type: ClassVar[str]
    scale: ClassVar[tuple[str, str]]
    max: ClassVar[str|int]
    min: ClassVar[str|int]
    criterion:Optional[str] = None
    score: int | float | Literal["yes", "no"]
    feedback: str = Field(
        description="Give detailed feedback on how you made your scoring-decision. Provide details on what justifies your decision.")


class BinaryAssessment(BaseAssessment):
    type: ClassVar[str] = "binary"
    scale: ClassVar[tuple[str, str]] = ("yes", "no")
    max: ClassVar[str] = "yes"# type: ignore
    min: ClassVar[str] = "no"# type: ignore
    score: Literal["yes", "no"]  # type: ignore
    feedback: str = Field(
        description="Give detailed feedback on how you made your scoring-decision. Provide details on what justifies your decision.")


class LikertAssessment(BaseAssessment):
    type: ClassVar[str] = "likert"
    scale: ClassVar[tuple[str, str]] = ("1", "5")
    max: ClassVar[int] = 5# type: ignore
    min: ClassVar[int] = 1# type: ignore
    score: Literal[1,2,3,4,5]  # type: ignore
    feedback: str = Field(
        description="Give detailed feedback on how you made your scoring-decision. Provide details on what justifies your decision.")
