# Evaluation

This is the main repository for providing LLM-as-a-Judge-based utilities and classes for assessing learning units.

## Architecture

The project is organized around dimensions and judges:

- `evaluation/dimensions`: base dimension models (`BaseDimension`) and rule-based dimension execution (`BaseRuleDimension`).
- `evaluation/types`: `MetricTypes` used as normalized scoring payloads (`BinaryMetricType`, `LikertMetricType`, etc.).
- `evaluation/judges`: orchestration classes (`Judge`, `Evaluation`) for computing and rendering results.
- `evaluation/lib`: helper functions for metric conversion, restoration, merging, and formatting.
- `evaluation/examples`: ready-to-use example dimensions and judge setups.

Computation flow:

1. Define dimensions (criteria grouped in a `BaseDimension` subclass).
    _optional_: Define own metric types using `BaseMetricType` 
2. Instantiate Judge(s) through `Judge`-class
3. Run rule-based dimensions and/or LLM-as-a-judge dimensions through judges.
    _optional_: Combine evaluations like `eval1 + eval2`.
4. Format, score, or visualize output using `Evaluation` helpers.

## Terminology

- **Dimension**: multiple grouped criteria defined in a dimension class based on `BaseDimension`.
- **Rule-based metrics**: dimensions that define callable checking functions (algorithms), and can also include ML-assisted logic when needed.
- **LLM-as-a-judge metrics**: dimensions computed by LLMs through the DSPy interface.
- **MetricTypes**: standardized result formats (for example Likert-style or binary scales) that judge outputs must conform to.
- **Judge**: a `dspy.Module` that provides methods to compute metrics (sync and async) and combine rule-based + LLM outputs.
- **Evaluation**: an object that contains combined rule-based and LLM-judge metric values and provides rich formatting/visualization helpers.

## Usage

### 1. Install

```bash
uv sync
```

This repository is intended to be run with `uv` and is also a Python package (`evaluation`).
If you need an editable package install, use:

```bash
uv pip install -e .
```

### 2. Configure a Judge

```python
import dspy
from evaluation.judges.judge import Judge
from evaluation.examples.dimensions.didactical.fidemm.elicit_performance import ElicitPerformanceMetric
from evaluation.examples.dimensions.rule_based.drag_text import DragTextRuleBased

lm = dspy.LM("openai/gpt-4.1-mini")

judge = Judge(
    llm=lm,
    llm_as_a_judge_metrics=[ElicitPerformanceMetric],
    rule_based_metrics=[DragTextRuleBased],
)
```

### 3. Evaluate Slides

```python
# slides must be list[creator.schemas.base.BaseComponent]
result = judge.forward(slides=slides)

print(result.get_total_score())
print(result.generate_assessment())

df = result.to_dataframe()
as_dict = result.to_dict()
```

If you only run rule-based metrics, you can omit LLM metrics and pass `llm=None`.
