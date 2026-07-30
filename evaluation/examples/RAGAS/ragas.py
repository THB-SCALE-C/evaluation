from evaluation.judges.judge import FunctionsJudge
from evaluation.types.assessment_types import BaseMetricType
from ragas.metrics.collections import Faithfulness

def RAGAS(slides,llm=None,**context):
    faithfulness_metric = Faithfulness(
        llm=llm,
    )
    score = faithfulness_metric.score(
        user_input=context["learning_objective"],
        response=slides,
        retrieved_contexts=context["context"],
    )
    return {
        "faithfulness":BaseMetricType(score=1, feedback="1")
    }