from typing import ClassVar
from pydantic import Field
from evaluation.metrics.base import BaseMetric
from evaluation.metrics.didactical._base import BaseDidacticalMetric
from evaluation.types.assessment_types import BinaryAssessment


class PresentContentMetric(BaseDidacticalMetric):
    metric_name: ClassVar[str] = "present_content"
    metric_type:ClassVar = "didactical"

    presence_and_completeness_of_core_content: BinaryAssessment = Field(
        description="Does the content include all core concepts explicitly required by the stated learning objective?")
    logical_structure: BinaryAssessment = Field(
        description="Is the content structured by logical sections and section titles?")
    coherence: BinaryAssessment = Field(
        description="Is the content coherent, without misleading or irrelevant topic shifts or anecdotes?")
    cognitive_progression: BinaryAssessment = Field(
        description="Does the content progress from simple concepts to more complex ones in a structured manner?")
    quality_of_examples_and_visuals: BinaryAssessment = Field(
        description="Is at least one example provided, and does it directly support key cybersecurity concepts introduced in the text?")
    micro_chunking: BinaryAssessment = Field(
        description="Is the content broken into short, clearly separated chunks (e.g., bullet points, short paragraphs)?")
    single_idea_per_chunk: BinaryAssessment = Field(
        description="Does each chunk focus on exactly one idea?")
    length_suitability: BinaryAssessment = Field(
        description="Is the overall content length less than one page of written text?")
    contextual_realism: BinaryAssessment = Field(
        description="Do examples, terminology, and scenarios reflect realistic situations, tools, and language from the learner's workplace context?")
    alignment_learning_objective: BinaryAssessment = Field(
        description="Does the content explicitly provide information or practice that enables achievement of the stated learning objective?")
    alignment_learner_context: BinaryAssessment = Field(
        description="Is the content adapted to the learner's role, typical responsibilities, and constraints (e.g., decision authority, tools available)?")
