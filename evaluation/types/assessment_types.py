from typing import Any, ClassVar, Literal, Optional, Tuple
from pydantic import BaseModel, Field, PrivateAttr, computed_field


class BaseMetricType(BaseModel):
    _criterion:str = PrivateAttr(default="")
    _meta: dict[str, Any] = PrivateAttr(default_factory=dict)
    type: ClassVar[str|None] = None
    scale: ClassVar[Tuple|None] = None
    max: ClassVar[int|None] = None
    min: ClassVar[int|None] = None


    @computed_field
    @property
    def criterion(self) -> str:
        return self._criterion
    @computed_field
    @property
    def is_llm_judge(self) -> bool:
        return bool(self._meta.get("is_llm_judge", False))
    @computed_field
    @property
    def description(self) -> str:
        return str(self._meta.get("description", ""))
    @computed_field
    @property
    def meta(self) -> dict:
        return self._meta

class BinaryMetricType(BaseMetricType):
    type: ClassVar[str] = "binary"
    scale: ClassVar[tuple] = (0, 1)
    max: ClassVar[int] = 1# type: ignore
    min: ClassVar[int] = 0# type: ignore
    score: Literal[0, 1]
    feedback: str = Field(
        description="Give detailed feedback on how you made your scoring-decision. Provide details on what justifies your decision.")


class LikertMetricType(BaseMetricType):
    type: ClassVar[str] = "likert"
    scale: ClassVar[tuple] = (1,2,3,4,5)
    max: ClassVar[int] = 5# type: ignore
    min: ClassVar[int] = 1# type: ignore
    score: Literal[1,2,3,4,5]
    feedback: str = Field(
        description="Give detailed feedback on how you made your scoring-decision. Provide details on what justifies your decision.")
