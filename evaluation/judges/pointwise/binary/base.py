from typing import Any, Literal, override
import dspy

class BaseJudge(dspy.Signature):
    slide: str = dspy.InputField(
        desc="The LLM generated content you have to assess")