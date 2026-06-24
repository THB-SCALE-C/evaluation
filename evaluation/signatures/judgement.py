from typing import Any

import dspy

class Judgement(dspy.Signature):
    """You are a highly critical llm-as-a-judge."""
    slides:list[Any] = dspy.InputField(
        desc="The LLM generated slides you have to assess")