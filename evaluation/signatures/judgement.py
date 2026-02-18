import dspy

class Judgement(dspy.Signature):
    slides: str = dspy.InputField(
        desc="The LLM generated slides you have to assess")
    # todo: add appropriate context Inputs here