
import dspy

class ReductionistJudge(dspy.Signature):
    slide: str = dspy.InputField(
        desc="The LLM generated content you have to assess")
    