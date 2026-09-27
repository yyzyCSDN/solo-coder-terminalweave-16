from datetime import datetime, timedelta, timezone

import pytest

from terminalweave.core.errors import ConflictError, SafetyConstraintError, StateTransitionError
from terminalweave.hazmat.compatibility import CompatibilityMatrix
from terminalweave.hazmat.emergency import EmergencyGuide, EmergencyGuidebook
from terminalweave.hazmat.isolation import (
    ContainerObservation,
    IsolationIncident,
    ResponseResource,
)

UTC = timezone.utc
T0 = datetime(2026, 9, 27, 10, 0, tzinfo=UTC)
T1 = T0 + timedelta(minutes=5)
T2 = T0 + timedelta(minutes=10)
T3 = T0 + timedelta(minutes=20)


def book():
    # class: iso, evac, water-reactive, medium, ppe
    return EmergencyGuidebook([
        EmergencyGuide('3', 25.0, 100.0, False, 'foam', 2),
        EmergencyGuide('8', 10.0, 50.0, False, 'water', 1),
        EmergencyGuide('4.3', 30.0, 150.0, True, 'dry_powder', 3),
        EmergencyGuide('2.1', 50.0, 200.0, False, 'foam', 3),
    ])


def obs(cid, x, y, declared=(), placards=()):
    return ContainerObservation(cid, x, y, frozenset(declared), frozenset(placards))


def open_incident(**kw):
    defaults = dict(
        incident_id='INC-1',
        source=obs('SRC', 0, 0, declared=['8']),
        neighbours=[obs('N1', 8, 0, declared=['8'])],
        resources=[ResponseResource('R1', 'water', ppe_level=1, x_m=50.0, y_m=0.0)],
        guidebook=book(),
        opened_at=T0,
        actor='chief',
    )
    defaults.update(kw)
    return IsolationIncident(**defaults)


# --------------------------------------------------------------- plan inputs

def test_declared_vs_placard_contradiction_plans_union():
    # Manifest says corrosive (8); the placard on the box says flammable (3).
    inc = open_incident(source=obs('SRC', 0, 0, declared=['8'], placards=['3']))
    plan = inc.plan
    assert plan.effective_classes == frozenset({'3', '8'})
    # Most dangerous reading: class 3 drives the 25 m radius.
    assert plan.isolation_m == 25.0
    assert plan.evacuation_m == 100.0
    c = plan.contradictions[0]
    assert c.container_id == 'SRC'
    assert c.missing_on_placard == frozenset({'8'})
    assert c.extra_on_placard == frozenset({'3'})
    assert any('most dangerous reading' in b for b in plan.basis)


def test_neighbours_expand_perimeter_and_protected_set():
    # Neighbour sits at 20 m: outside the class-8 radius (10 m) but inside the
    # class-3 radius once the placard contradiction is taken at its worst.
    inc = open_incident(
        source=obs('SRC', 0, 0, declared=['8'], placards=['3']),
        neighbours=[obs('FAR', 20, 0, declared=['3']), obs('SAFE', 500, 0, declared=['8'])])
    assert inc.plan.protected_neighbour_ids == frozenset({'FAR'})
    assert any('FAR' in b and '20.0m' in b for b in inc.plan.basis)


def test_neighbour_with_worse_class_cascades_envelope():
    # Source is class 8; a neighbour just inside 10 m carries 2.1, dragging the
    # evacuation envelope to 200 m; a downwind neighbour is only reachable via
    # the evacuation distance after the envelope expands.
    inc = open_incident(
        source=obs('SRC', 0, 0, declared=['8']),
        neighbours=[obs('GAS', 9, 0, declared=['2.1']),
                    obs('RING', 0, 120, declared=['8'])],
        wind_direction_deg=0.0)
    assert inc.plan.isolation_m == 50.0
    assert inc.plan.evacuation_m == 200.0
    assert inc.plan.protected_neighbour_ids == frozenset({'GAS', 'RING'})


def test_downwind_neighbour_uses_evacuation_distance():
    # Bearing convention: wind toward +y is 0 deg. A box at 120 m downwind is
    # beyond the 50 m hot zone but inside the 200 m evacuation sector.
    inc = open_incident(
        source=obs('SRC', 0, 0, declared=['2.1']),
        neighbours=[obs('PLUME', 0, 120, declared=['8']),
                    obs('UPWIND', 0, -120, declared=['8'])],
        wind_direction_deg=0.0)
    assert inc.plan.protected_neighbour_ids == frozenset({'PLUME'})


def test_unknown_hazard_fails_closed():
    with pytest.raises(SafetyConstraintError):
        open_incident(source=obs('SRC', 0, 0))
    with pytest.raises(SafetyConstraintError):
        open_incident(neighbours=[obs('MYSTERY', 5, 0)])


def test_missing_emergency_guide_fails_closed():
    src = obs('SRC', 0, 0, declared=['9'])
    with pytest.raises(SafetyConstraintError):
        open_incident(source=src, neighbours=[])


# ---------------------------------------------------------------- resources

def test_resource_assessment_classifies_assets():
    inc = open_incident(
        source=obs('SRC', 0, 0, declared=['8'], placards=['3']),
        resources=[
            ResponseResource('FAR-FOAM', 'foam', ppe_level=2, x_m=80, y_m=0),
            ResponseResource('FAR-WATER', 'water', ppe_level=1, x_m=90, y_m=0),
            ResponseResource('TRAPPED', 'water', ppe_level=1, x_m=5, y_m=0),
            ResponseResource('WRONG', 'water', ppe_level=1,
                             applicable_classes=frozenset({'8'}), x_m=80, y_m=0),
            ResponseResource('LOST', 'foam', ppe_level=2),
        ])
    r = inc.plan.resources
    assert r.usable == ('FAR-FOAM', 'FAR-WATER')
    assert r.trapped == ('TRAPPED',)
    assert r.unsuitable == ('WRONG',)
    assert r.unpositioned == ('LOST',)
    assert r.required_ppe_level == 2
    assert not r.missing_media
    assert not r.missing_ppe


def test_missing_media_and_ppe_reported():
    # Class 3 needs foam and PPE 2; only a water unit with PPE 1 is usable.
    inc = open_incident(
        source=obs('SRC', 0, 0, declared=['3']),
        resources=[ResponseResource('W', 'water', ppe_level=1, x_m=60, y_m=0)])
    assert inc.plan.resources.missing_media == ('foam',)
    assert inc.plan.resources.missing_ppe


# ------------------------------------------------------------- move ordering

def test_movement_order_neighbours_before_source_and_outer_first():
    inc = open_incident(
        source=obs('SRC', 0, 0, declared=['8']),
        neighbours=[obs('NEAR', 5, 0, declared=['8']),
                    obs('FAR', 9, 0, declared=['8'])])
    ids = [(s.container_id, s.role) for s in inc.plan.movements]
    assert ids == [('FAR', 'neighbour'), ('NEAR', 'neighbour'), ('SRC', 'source')]


def test_prohibited_pair_and_water_reactive_take_priority():
    matrix = CompatibilityMatrix()
    matrix.set_incompatible('4.3', '8')
    inc = open_incident(
        source=obs('SRC', 0, 0, declared=['8'], placards=['3']),
        neighbours=[obs('REACTIVE', 8, 0, declared=['4.3']),
                    obs('IGNITION', 20, 0, declared=['3'])],
        matrix=matrix)
    roles = [(s.container_id, s.role) for s in inc.plan.movements]
    # Prohibited reaction pair first, then the ignition-source neighbour,
    # and the source last.
    assert roles[0] == ('REACTIVE', 'incompatible_neighbour')
    assert roles[1] == ('IGNITION', 'ignition_source')
    assert roles[-1] == ('SRC', 'source')


# ------------------------------------------------ revision / traceability

def test_escalation_is_recorded_in_trail():
    inc = open_incident()
    assert inc.level == 'level_1'
    inc.revise(T1, 'chief', source=obs('SRC', 0, 0, declared=['2.1']))
    assert inc.level == 'level_2'
    assert inc.plan.revision == 2
    kinds = [e.kind for e in inc.trail()]
    assert kinds == ['opened', 'escalated']
    esc = inc.trail()[-1]
    assert esc.detail['from_level'] == 'level_1'
    assert esc.detail['level'] == 'level_2'
    assert esc.seq == 2 and esc.at == T1 and esc.actor == 'chief'


def test_revision_cannot_shrink_live_perimeter():
    inc = open_incident(source=obs('SRC', 0, 0, declared=['3']))
    assert inc.plan.isolation_m == 25.0
    # "Corrected" paperwork claiming only class 8 must not silently downgrade.
    with pytest.raises(SafetyConstraintError):
        inc.revise(T1, 'chief', source=obs('SRC', 0, 0, declared=['8']))
    # Failed revision leaves plan and trail untouched.
    assert inc.plan.revision == 1
    assert [e.kind for e in inc.trail()] == ['opened']


def test_stand_down_then_reopen_allows_downgrade_and_is_traced():
    inc = open_incident(source=obs('SRC', 0, 0, declared=['3']))
    inc.stand_down(T1, 'on-scene-commander', 'leak contained, air monitoring clear')
    inc.reopen(T2, 'on-scene-commander', 'residual class-8 cleanup only')
    plan = inc.revise(T3, 'on-scene-commander',
                      source=obs('SRC', 0, 0, declared=['8']))
    assert plan.isolation_m == 10.0
    assert [e.kind for e in inc.trail()] == ['opened', 'stood_down', 'reopened', 'revised']
    # Monotonic again from the new baseline: re-expanding is fine, but
    # shrinking back is rejected without another recorded stand-down.
    inc.revise(T3, 'chief', source=obs('SRC', 0, 0, declared=['8'], placards=['3']))
    with pytest.raises(SafetyConstraintError):
        inc.revise(T3, 'chief', source=obs('SRC', 0, 0, declared=['8']))
    inc.stand_down(T3, 'on-scene-commander', 'site handed back')
    with pytest.raises(StateTransitionError):
        inc.stand_down(T3, 'x', 'again')


def test_revision_cannot_override_movement_already_entered():
    inc = open_incident(
        neighbours=[obs('A', 5, 0, declared=['8']),
                    obs('B', 9, 0, declared=['8'])])
    # Sequence: B (outer), A (inner), SRC. Complete B, i.e. a crew entered
    # the zone and moved it under revision 1.
    inc.start_movement(T1, 'crew-1', 1)
    inc.complete_movement(T2, 'crew-1', 1)
    # A new worse neighbour appears ahead of B in ordering; the executed
    # movement can neither be renumbered nor replaced.
    with pytest.raises(ConflictError, match='cannot be overridden'):
        inc.revise(T3, 'chief',
                   neighbours=[obs('A', 5, 0, declared=['8']),
                               obs('B', 9, 0, declared=['8']),
                               obs('Z', 9.5, 0, declared=['4.3'])])
    # A compatible revision that preserves B at step 1 carries its done state.
    plan = inc.revise(T3, 'chief',
                      neighbours=[obs('A', 5, 0, declared=['8']),
                                  obs('B', 9, 0, declared=['8']),
                                  obs('NEW', 2, 0, declared=['8'])])
    done = {s.step_no: s for s in plan.movements if s.state == 'done'}
    assert done[1].container_id == 'B'


def test_in_progress_movement_blocks_replanning():
    inc = open_incident()
    inc.start_movement(T1, 'crew-1', 1)
    with pytest.raises(ConflictError, match='in progress'):
        inc.revise(T2, 'chief', source=obs('SRC', 0, 0, declared=['3']))


def test_movement_state_machine_enforces_order():
    inc = open_incident()
    with pytest.raises(ConflictError):
        inc.complete_movement(T1, 'crew-1', 1)  # never started
    inc.start_movement(T1, 'crew-1', 1)
    with pytest.raises(ConflictError):
        inc.start_movement(T2, 'crew-1', 1)  # already in progress


def test_revision_monotonic_growth_is_supported():
    inc = open_incident(source=obs('SRC', 0, 0, declared=['8']))
    inc.revise(T1, 'chief', source=obs('SRC', 0, 0, declared=['8'], placards=['3']))
    # Re-read placard corrects 3 -> 2.1: the class set changes, but since the
    # physical envelope only grows (25 m -> 50 m) the revision is accepted.
    inc.revise(T2, 'chief', source=obs('SRC', 0, 0, declared=['8'], placards=['2.1']))
    assert inc.plan.isolation_m == 50.0
    assert [e.kind for e in inc.trail()] == ['opened', 'revised', 'escalated']
    assert inc.trail()[-1].detail['classes'] == ['2.1', '8']
