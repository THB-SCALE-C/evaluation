from typing import ClassVar
from pydantic import Field
from evaluation.rubrics.base import BaseRubric
from evaluation.rubrics.base_didactical import BaseDidacticalRubric
from evaluation.types.assessment_types import BinaryMetricType


class EnhanceRetentionTransferMetric(BaseDidacticalRubric):
    metric_name: ClassVar[str] = "enhance_retention_transfer"
    metric_type:ClassVar = "didactical"


    presence_of_summary_takeaways: BinaryMetricType = Field(
        description="Is there a final summary or list of key takeaways?")
    quality_of_summary: BinaryMetricType = Field(
        description="Does the summary include only the most important concepts from the unit?")
    summary_conciseness: BinaryMetricType = Field(
        description="Is the summary no longer than 3-5 bullet points or short sentences?")
    support_to_act: BinaryMetricType = Field(
        description="Does the unit explicitly prompt learners to apply the knowledge in practice?")
    realistic_transfer_cues: BinaryMetricType = Field(
        description="Does the transfer cue refer to the learner's real work environment or tasks?")
    summary_aligned_with_learning_objective: BinaryMetricType = Field(
        description="Does the summary restate or reinforce the learning objective?")
    transfer_cues_aligned_with_learning_objective: BinaryMetricType = Field(
        description="Do transfer prompts ask learners to practice the behavior defined in the learning objective?")
