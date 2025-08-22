import copy

from src.core.schemas import (
    EnrichedReviewJSON,
    EnhancedActionField,
    EnhancedProject,
    EnhancedMeasure,
    EnhancedIndicator,
    ConnectionWithConfidence,
)
from src.processing.graph_postprocessing import (
    remove_redundant_af_measure_shortcuts,
)


def _targets(connections):
    return [c.target_id for c in connections]


def test_af_to_measure_pruned_when_project_path_exists_with_idempotency():
    # AF → P and P → M exist; AF → M should be pruned
    af = EnhancedActionField(
        id="af_1",
        content={"title": "AF"},
        connections=[
            ConnectionWithConfidence(target_id="proj_1", confidence_score=0.91),
            ConnectionWithConfidence(target_id="msr_1", confidence_score=0.72),
        ],
    )
    proj = EnhancedProject(
        id="proj_1",
        content={"title": "P"},
        connections=[ConnectionWithConfidence(target_id="msr_1", confidence_score=0.8)],
    )
    msr = EnhancedMeasure(id="msr_1", content={"title": "M"}, connections=[])

    state = EnrichedReviewJSON(
        action_fields=[af], projects=[proj], measures=[msr], indicators=[]
    )

    # Keep snapshot for unaffected counts
    original_proj_conns = copy.deepcopy(proj.connections)

    # Action
    out = remove_redundant_af_measure_shortcuts(state)

    # Assertions: AF→M removed, AF→P remains with same confidence
    af_targets = _targets(out.action_fields[0].connections)
    assert af_targets == ["proj_1"]
    assert out.action_fields[0].connections[0].confidence_score == 0.91

    # Unaffected: Project → Measure unchanged
    assert _targets(out.projects[0].connections) == ["msr_1"]
    assert out.projects[0].connections[0].confidence_score == original_proj_conns[0].confidence_score

    # Idempotency: second run makes no further changes
    out2 = remove_redundant_af_measure_shortcuts(out)
    assert _targets(out2.action_fields[0].connections) == ["proj_1"]
    assert _targets(out2.projects[0].connections) == ["msr_1"]


def test_measure_to_af_pruned_when_project_path_exists():
    # Reverse case: M → AF removed when AF → P and P → M exist
    af = EnhancedActionField(
        id="af_2",
        content={"title": "AF2"},
        connections=[ConnectionWithConfidence(target_id="proj_2", confidence_score=0.93)],
    )
    proj = EnhancedProject(
        id="proj_2",
        content={"title": "P2"},
        connections=[ConnectionWithConfidence(target_id="msr_2", confidence_score=0.84)],
    )
    msr = EnhancedMeasure(
        id="msr_2",
        content={"title": "M2"},
        connections=[ConnectionWithConfidence(target_id="af_2", confidence_score=0.71)],
    )

    state = EnrichedReviewJSON(
        action_fields=[af], projects=[proj], measures=[msr], indicators=[]
    )

    out = remove_redundant_af_measure_shortcuts(state)

    # M→AF removed, project edges unaffected
    assert _targets(out.measures[0].connections) == []
    assert _targets(out.action_fields[0].connections) == ["proj_2"]
    assert _targets(out.projects[0].connections) == ["msr_2"]


def test_keeps_af_to_measure_when_no_project_path():
    # AF → M present but missing AF → P; should keep AF → M
    af = EnhancedActionField(
        id="af_3",
        content={"title": "AF3"},
        connections=[ConnectionWithConfidence(target_id="msr_3", confidence_score=0.66)],
    )
    proj = EnhancedProject(
        id="proj_3",
        content={"title": "P3"},
        connections=[ConnectionWithConfidence(target_id="msr_9", confidence_score=0.5)],
    )
    msr = EnhancedMeasure(id="msr_3", content={"title": "M3"}, connections=[])

    state = EnrichedReviewJSON(
        action_fields=[af], projects=[proj], measures=[msr], indicators=[]
    )

    out = remove_redundant_af_measure_shortcuts(state)

    # AF→M kept (no AF→P to complete the path)
    assert _targets(out.action_fields[0].connections) == ["msr_3"]
    assert out.action_fields[0].connections[0].confidence_score == 0.66
    # Unrelated project connection untouched
    assert _targets(out.projects[0].connections) == ["msr_9"]


def test_does_not_touch_other_edges():
    # Ensure non-target edges like AF→Indicator and Project→Indicator remain unchanged
    af = EnhancedActionField(
        id="af_4",
        content={"title": "AF4"},
        connections=[
            ConnectionWithConfidence(target_id="proj_4", confidence_score=0.9),
            ConnectionWithConfidence(target_id="ind_4", confidence_score=0.77),
        ],
    )
    proj = EnhancedProject(
        id="proj_4",
        content={"title": "P4"},
        connections=[
            ConnectionWithConfidence(target_id="msr_4", confidence_score=0.8),
            ConnectionWithConfidence(target_id="ind_4", confidence_score=0.7),
        ],
    )
    msr = EnhancedMeasure(id="msr_4", content={"title": "M4"}, connections=[])
    ind = EnhancedIndicator(id="ind_4", content={"title": "I4"}, connections=[])

    state = EnrichedReviewJSON(
        action_fields=[af], projects=[proj], measures=[msr], indicators=[ind]
    )

    out = remove_redundant_af_measure_shortcuts(state)

    # AF→Indicator and Project→Indicator are unchanged
    assert set(_targets(out.action_fields[0].connections)) == {"proj_4", "ind_4"}
    assert set(_targets(out.projects[0].connections)) == {"msr_4", "ind_4"}

