from pydantic import  Field
from typing import ClassVar
from evaluation.rubrics.base import BaseRubric
from evaluation.rubrics.base_didactical import BaseDidacticalRubric
from evaluation.types.assessment_types import BinaryMetricType


class AttentionHookMetric(BaseDidacticalRubric):
    metric_name:ClassVar[str] = "attention_hook"
    metric_type:ClassVar = "didactical"

    presence_of_hook:BinaryMetricType = Field(description="Is the hook presented as a realistic cyber security scenario or surprising cyber security fact that grabs attention while clearly connected to the overall topic?")
    salience:BinaryMetricType = Field(description="Is the hook surprising, emotionally salient, or personally relevant?")
    personal_salience:BinaryMetricType = Field(description="Does the hook address the reader personally or involve a personalized scenario?")
    conciseness:BinaryMetricType = Field(description="Is the hook shorter than or exactly two sentences? And if 'yes', are the sentences short and comprehensive?")
    alignment_learning_objective:BinaryMetricType = Field(description="Does the hook clearly relate to the learning objectives rather than introducing unrelated fun facts?")
    alignment_further_content:BinaryMetricType = Field(description="Does the hook clearly relate to the content?")


    