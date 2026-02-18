from pydantic import  Field
from typing import ClassVar
from evaluation.metrics.base import BaseMetric
from evaluation.types.assessment_types import BinaryAssessment


class AttentionHookMetric(BaseMetric):
    metric_name:ClassVar[str] = "attention_hook"
    presence_of_hook:BinaryAssessment = Field(description="Is the hook presented as a realistic cyber security scenario or surprising cyber security fact that grabs attention while clearly connected to the overall topic?")
    salience:BinaryAssessment = Field(description="Is the hook surprising, emotionally salient, or personally relevant?")
    personal_salience:BinaryAssessment = Field(description="Does the hook address the reader personally or involve a personalized scenario?")
    conciseness:BinaryAssessment = Field(description="Is the hook shorter than or exactly two sentences? And if 'yes', are the sentences short and comprehensive?")
    alignment_learning_objective:BinaryAssessment = Field(description="Does the hook clearly relate to the learning objectives rather than introducing unrelated fun facts?")
    alignment_further_content:BinaryAssessment = Field(description="Does the hook clearly relate to the content?")


    