from typing import Literal, Optional, Union
from pydantic import BaseModel


class BaseMetricType(BaseModel):
    type:Literal["binary","likert","continuous"] = "likert"
    name:str
    scale: tuple[str,str] | tuple[float,float] | tuple[int,int] = (0,1)
    max:Optional[str|float|int] = None
    min:Optional[str|float|int] = None

class BinaryMetricType(BaseMetricType):
    name:str
    type:Literal["binary","likert","continuous"] = "binary"
    scale:tuple[str,str] | tuple[float,float] | tuple[int,int]  = ("yes","no")
    max:Optional[str|float|int] = "yes"
    min:Optional[str|float|int] = "no"