from typing import ClassVar
from pydantic import Field
from evaluation.metrics.didactical._base import BaseDidacticalMetric
from evaluation.types.assessment_types import BinaryAssessment


class AssessPerformanceMetric(BaseDidacticalMetric):
    metric_name: ClassVar[str] = "assess_performance"
    metric_type:ClassVar = "didactical"
    
    summative_assessment_present: BinaryAssessment = Field(
        description="Is there at least one summative assessment item at the end of the unit?")
    mcq_stem_clarity_and_key_validity: BinaryAssessment = Field(
        description="Is the question clearly worded, and is exactly one answer clearly correct based on the content?")
    distractor_plausibility: BinaryAssessment = Field(
        description="Are incorrect answer options plausible and based on common learner errors?")
    independence_from_guidance: BinaryAssessment = Field(
        description="Does the assessment avoid hints, explanations, or guiding language?")
    application_related_assessment: BinaryAssessment = Field(
        description="Does the question require applying knowledge to a realistic workplace scenario?")
    compactness: BinaryAssessment = Field(
        description="Does the assessment consist of only a small number of focused items (e.g., <=5)?")
    alignment_learning_objective: BinaryAssessment = Field(
        description="Do the assessment items directly test the learning objective?")
    alignment_content: BinaryAssessment = Field(
        description="Are all assessment items covered by the instructional content?")
