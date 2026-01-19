from typing import Literal
from pydantic import BaseModel, Field



class BaseAssessment(BaseModel):
    score: int | float | Literal["yes","no"]
    feedback: str = Field(
        description="Give detailed feedback on how you made your scoring-decision. Provide details on what justifies your decision.")
    

class BinaryAssessment(BaseAssessment):
    score: Literal["yes", "no"] # type: ignore
    feedback: str = Field(
        description="Give detailed feedback on how you made your scoring-decision. Provide details on what justifies your decision.")