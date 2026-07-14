from abc import abstractmethod
import inspect
from typing import Any, Callable, ClassVar, Tuple
import pydantic
from evaluation.dimensions.base import BaseDimension
from evaluation.types.assessment_types import BaseMetricType, BinaryMetricType


class BaseRuleDimension[T]():
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
    metric_type: ClassVar = BinaryMetricType
    metric_name: ClassVar[str] = ""
    required_slide_type:ClassVar = ""
    is_llm_judge = False

    def __new__(cls, data: T, index:int|None =None, **context) -> BaseMetricType:
        return super().__new__(cls)._evaluate(data,index)

    
    def _evaluate(self,data:T,index:int|None =None, **context) -> Any:
        evals = dict()
        for key in self.__dir__():
            if key.startswith("_") or key == "evaluate":
                continue
            func = getattr(self, key)
            if inspect.ismethod(func) and not func.__name__.startswith("_"):
                res = func(data,**context)
                if not res:
                    continue
                checked, feedback = res
                if isinstance(checked,bool):
                    checked = int(checked)
                evals[key] = self.metric_type(
                    score=checked, #type:ignore
                    feedback=feedback
                )
        fields = {key: BaseMetricType for key in evals.keys()} \
            | {"metric_name": (ClassVar, self.metric_name), "required_slide_type":(ClassVar, self.required_slide_type)}
        model = pydantic.create_model(
            self.__class__.__name__,
            __base__=BaseDimension,
              **fields)  # type:ignore
        instance = model(**evals)
        instance._index = index
        return instance


    def check(self, data: T, **context) -> Tuple[Any, str]: #type:ignore
        pass
