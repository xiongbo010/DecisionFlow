"""Dataset schemas and policies used by the paper experiments.

The entries contain no data loaders or model code. They translate standardized
cached scores into DecisionFlow requests and constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple


QuestionRows = Tuple[Mapping[str, Any], ...]
ConstraintDocument = Mapping[str, Any]


def _noul(*names: str) -> QuestionRows:
    return tuple({"id": name, "type": "noul"} for name in names)


def _implies(left: str, right: str, value: Any = True) -> Mapping[str, Any]:
    return {
        "implies": [
            {"eq": [{"var": left}, True]},
            {"eq": [{"var": right}, value]},
        ]
    }


@dataclass(frozen=True)
class ExperimentSpec:
    """Static portion of a cached-score experiment."""

    name: str
    questions: QuestionRows
    constraints: ConstraintDocument
    description: str
    primary_metrics: Tuple[str, ...]
    record_mode: str = "fixed"

    def request(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        if self.record_mode == "grounded":
            request = dict(record["request"])
            if "probabilities" not in request:
                request["probabilities"] = record["probabilities"]
            return request
        return {
            "id": record.get("id"),
            "state": record.get("state", {}),
            "questions": [dict(question) for question in self.questions],
            "probabilities": record["probabilities"],
        }

    def constraint_document(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        # Cached records may carry a frozen, provenance-tracked policy variant
        # (for example train-estimated soft penalties). The default remains the
        # registered primary policy.
        if "constraints" in record:
            return record["constraints"]
        if self.record_mode == "grounded":
            return self.constraints
        return self.constraints


MODERATION_NAMES = ("S", "H", "V", "HR", "SH", "S3", "H2", "V2")

GOEMOTIONS_NAMES = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
)


EXPERIMENTS: Dict[str, ExperimentSpec] = {
    "openai_moderation": ExperimentSpec(
        name="openai_moderation",
        questions=_noul(*MODERATION_NAMES),
        constraints={
            "name": "openai-moderation-hierarchy",
            "version": "1",
            "hard": [
                {"name": "sexual-minors-implies-sexual", "expr": _implies("S3", "S")},
                {"name": "hate-threatening-implies-hate", "expr": _implies("H2", "H")},
                {"name": "violence-graphic-implies-violence", "expr": _implies("V2", "V")},
            ],
        },
        description="Eight Boolean moderation categories with label hierarchy.",
        primary_metrics=("micro_f1", "macro_f1", "average_precision"),
    ),
    "toxigen": ExperimentSpec(
        name="toxigen",
        questions=(
            {"id": "toxic", "type": "noul"},
            {"id": "toxicity", "type": "score", "options": [0, 1, 2, 3, 4]},
        ),
        constraints={
            "name": "toxigen-semantic-anchors",
            "version": "1",
            "hard": [
                {
                    "name": "zero-is-nontoxic",
                    "expr": {"implies": [
                        {"eq": [{"var": "toxicity"}, 0]},
                        {"eq": [{"var": "toxic"}, False]},
                    ]},
                },
                {
                    "name": "four-is-toxic",
                    "expr": {"implies": [
                        {"eq": [{"var": "toxicity"}, 4]},
                        {"eq": [{"var": "toxic"}, True]},
                    ]},
                },
            ],
        },
        description="A Boolean toxicity judgment and an ordinal toxicity score.",
        primary_metrics=("toxicity_f1", "score_mae", "score_spearman"),
    ),
    "goemotions": ExperimentSpec(
        name="goemotions",
        questions=_noul(*GOEMOTIONS_NAMES),
        constraints={
            "name": "goemotions-neutral-exclusion",
            "version": "1",
            "hard": [
                {
                    "name": "neutral-excludes-%s" % name,
                    "expr": _implies("neutral", name, False),
                }
                for name in GOEMOTIONS_NAMES[:-1]
            ],
        },
        description="Twenty-eight Boolean emotion labels with neutral exclusion.",
        primary_metrics=("micro_f1", "macro_f1", "average_precision"),
    ),
    "toxicchat": ExperimentSpec(
        name="toxicchat",
        questions=_noul("toxic", "jailbreak"),
        constraints={
            "name": "toxicchat-label-policy",
            "version": "1",
            "hard": [{"name": "jailbreak-implies-toxic", "expr": _implies("jailbreak", "toxic")}],
        },
        description="Two Boolean labels with a cross-label implication.",
        primary_metrics=("micro_f1", "macro_f1", "exact_match"),
    ),
    "helpsteer2": ExperimentSpec(
        name="helpsteer2",
        questions=tuple(
            {"id": name, "type": "score", "options": [0, 1, 2, 3, 4]}
            for name in ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
        ),
        constraints={
            "name": "helpsteer2-score-relations",
            "version": "1",
            "hard": [
                {
                    "name": "high-help-needs-correctness",
                    "expr": {"implies": [
                        {"ge": [{"var": "helpfulness"}, 3]},
                        {"ge": [{"var": "correctness"}, 1]},
                    ]},
                },
                {
                    "name": "high-help-needs-coherence",
                    "expr": {"implies": [
                        {"ge": [{"var": "helpfulness"}, 3]},
                        {"ge": [{"var": "coherence"}, 1]},
                    ]},
                },
                {
                    "name": "max-help-needs-coherence",
                    "expr": {"implies": [
                        {"ge": [{"var": "helpfulness"}, 4]},
                        {"ge": [{"var": "coherence"}, 2]},
                    ]},
                },
            ],
        },
        description="Five ordinal quality scores with cross-score relations.",
        primary_metrics=("mean_spearman", "mae", "joint_exact_match"),
    ),
    "typed_decisions": ExperimentSpec(
        name="typed_decisions",
        questions=(),
        constraints={},
        description="Per-example heterogeneous typed workflows and grounded finite relations.",
        primary_metrics=("value_accuracy", "constraint_violations", "nll"),
        record_mode="grounded",
    ),
    "sop_bench": ExperimentSpec(
        name="sop_bench",
        questions=(),
        constraints={},
        description="Per-state SOP decisions with grounded admissibility tables.",
        primary_metrics=("decision_accuracy", "procedure_success", "constraint_violations"),
        record_mode="grounded",
    ),
}

# The soft penalties were estimated from the HelpSteer2 training split with
# Laplace-smoothed violation odds and then frozen before evaluation. Keeping
# this as a distinct registry entry prevents the hard and hard+soft conditions
# from being silently mixed in one result file.
_helpsteer_hard = EXPERIMENTS["helpsteer2"]
EXPERIMENTS["helpsteer2_soft"] = ExperimentSpec(
    name="helpsteer2_soft",
    questions=_helpsteer_hard.questions,
    constraints={
        **_helpsteer_hard.constraints,
        "name": "helpsteer2-hard-and-train-estimated-soft-relations",
        "soft": [
            {
                "name": "high-help-prefers-correctness-2",
                "penalty": 6.1327680776239095,
                "source": "HelpSteer2 train split; Laplace-smoothed violation odds",
                "expr": {"implies": [
                    {"ge": [{"var": "helpfulness"}, 3]},
                    {"ge": [{"var": "correctness"}, 2]},
                ]},
            },
            {
                "name": "high-help-prefers-coherence-2",
                "penalty": 8.470101583882387,
                "source": "HelpSteer2 train split; Laplace-smoothed violation odds",
                "expr": {"implies": [
                    {"ge": [{"var": "helpfulness"}, 3]},
                    {"ge": [{"var": "coherence"}, 2]},
                ]},
            },
            {
                "name": "max-correctness-prefers-helpfulness-3",
                "penalty": 4.244943582617576,
                "source": "HelpSteer2 train split; Laplace-smoothed violation odds",
                "expr": {"implies": [
                    {"ge": [{"var": "correctness"}, 4]},
                    {"ge": [{"var": "helpfulness"}, 3]},
                ]},
            },
        ],
    },
    description="HelpSteer2 hard rules plus train-estimated weighted preferences.",
    primary_metrics=_helpsteer_hard.primary_metrics,
)


def get_experiment(name: str) -> ExperimentSpec:
    try:
        return EXPERIMENTS[name]
    except KeyError as error:
        raise KeyError("unknown experiment %r; choose from %s" % (name, ", ".join(EXPERIMENTS))) from error
