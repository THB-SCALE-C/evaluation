import dspy

class Judgement(dspy.Signature):
    """You are a highly critical llm-as-a-judge."""
    slides: str = dspy.InputField(
        desc="The LLM generated slides you have to assess")