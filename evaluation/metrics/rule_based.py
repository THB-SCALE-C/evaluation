from abc import abstractmethod
from typing import Any, ClassVar, Tuple
import pydantic
from evaluation.metrics.base import BaseMetric
from evaluation.types.assessment_types import BaseAssessment, BinaryAssessment


class BaseRuleMetric[T]():
    """
    Rule based metric checker. 
    Define Metrics as instance functions that return `Tuple[bool, str (feedback)]`, use `def check` for auto suggestion.
    
    Returns a pydantic Model with the results as attributes.

    Example:
    ```class CheckSingleChoice(BaseRuleMetric[SingleChoice]):
    def check_questions_count(self, data: SingleChoice) -> Tuple[bool, str]:
        to_many = len(data.questions) > 4
        to_few = len(data.questions) < 2
        feedback = "to many question items" if to_many 
            else "to few question items" if to_few else "okay"
        return not (to_many or to_few), feedback```
    """
    metric_type: ClassVar[str] = "rule_based"
    metric_name: ClassVar[str] = ""
    required_slide_type:ClassVar = ""
    is_llm_judge = False

    def __new__(cls, data: T, index:int|None =None) -> BaseAssessment:
        return super().__new__(cls)._evaluate(data,index)

    
    def _evaluate(self,data:T,index:int|None =None) -> Any:
        evals = dict()
        for key in self.__dir__():
            if key.startswith("_") or key == "evaluate":
                continue
            func = getattr(self, key)
            if hasattr(func, "__call__"):
                res = func(data)
                if not res:
                    continue
                checked, feedback = res
                evals[key] = BinaryAssessment(
                    criterion=key,
                    score="yes" if checked else "no",
                    feedback=feedback
                )
        fields = {key: BaseAssessment for key in evals.keys()} \
            | {"metric_type": (ClassVar, self.metric_type), "metric_name": (ClassVar, self.metric_name), "required_slide_type":(ClassVar, self.required_slide_type),
               "index_":int|None}
        model = pydantic.create_model(
            self.__class__.__name__,
            __base__=BaseMetric,
              **fields)  # type:ignore
        evals["index_"] = index
        return model(**evals)

    @abstractmethod
    def check(self, data: T) -> Tuple[bool, str]:
        pass
