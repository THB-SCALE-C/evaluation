from typing import Literal
from pydantic import BaseModel, Field



class BaseAssessment(BaseModel):
    score: str | int | float
    feedback: str = Field(
        description="Give detailed feedback on how you made your scoring-decision. Provide details on what justifies your decision.")
    

class BinaryAssessment(BaseModel):
    score: Literal["yes", "no"]
    feedback: str = Field(
        description="Give detailed feedback on how you made your scoring-decision. Provide details on what justifies your decision.")