from typing import ClassVar, Literal, Optional
from pydantic import BaseModel, Field


class BaseAssessment(BaseModel):
    type: ClassVar[str]
    scale: ClassVar[tuple[str, str]]
    max: ClassVar[str]
    min: ClassVar[str]
    score: int | float | Literal["yes", "no"]
    feedback: str = Field(
        description="Give detailed feedback on how you made your scoring-decision. Provide details on what justifies your decision.")


class BinaryAssessment(BaseAssessment):
    type: ClassVar[str] = "binary"
    scale: ClassVar[tuple[str, str]] = ("yes", "no")
    max: ClassVar[str] = "yes"
    min: ClassVar[str] = "no"
    score: Literal["yes", "no"]  # type: ignore
    feedback: str = Field(
        description="Give detailed feedback on how you made your scoring-decision. Provide details on what justifies your decision.")
