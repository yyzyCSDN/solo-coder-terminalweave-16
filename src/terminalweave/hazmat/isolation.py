"""Hazardous-materials incident isolation workflow.

The isolation plan is assembled from four inputs:

* declared DG classes (shipping documents / DG manifest);
* placards actually observed on the containers at the scene;
* neighbouring containers that sit inside the emergency distances;
* response resources (fire/rescue/spill) available on site.

When the inputs disagree the plan is built on the *most dangerous* reading:
every class suggested by either source is retained, and the contradiction is
recorded on the plan. Required safety information that is missing fails
closed with ``SafetyConstraintError``.

The workflow is a state machine with an append-only audit trail. Revising a
plan never rewrites actions responders have already entered: steps that are
``in_progress`` or ``done`` keep their identity and ordering, and any new plan
that would touch their containers is rejected. Every escalation, downgrade
(stand-down), reopening and movement execution appends a traceable event.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from terminalweave.core.clock import ensure_utc
from terminalweave.core.errors import (
    ConflictError,
    SafetyConstraintError,
    StateTransitionError,
    ValidationError,
)
from .compatibility import CompatibilityMatrix
from .emergency import EmergencyGuidebook

# Downwind plume sector: neighbours whose bearing is within this half-angle of
# the wind vector are planned against the evacuation distance.
_DOWNWIND_HALF_ANGLE_DEG = 45.0


@dataclass(frozen=True)
class ContainerObservation:
    """A container at the scene with its declared and physically seen hazards.

    Either side may be empty (e.g. paperwork missing, or placards burned off),
    but both being empty means the hazard is unknown and fails closed.
    """

    container_id: str
    x_m: float
    y_m: float
    declared_classes: frozenset[str] = frozenset()
    placard_classes: frozenset[str] = frozenset()

    @property
    def effective_classes(self) -> frozenset[str]:
        # Most dangerous reading: keep every class either source claims.
        return self.declared_classes | self.placard_classes

    @property
    def contradictions(self) -> frozenset[str]:
        return self.declared_classes ^ self.placard_classes


@dataclass(frozen=True)
class HazardContradiction:
    container_id: str
    missing_on_placard: frozenset[str]  # declared but not seen
    extra_on_placard: frozenset[str]    # seen but not declared


@dataclass(frozen=True)
class ResponseResource:
    """A response asset: foam unit, rescue team, spill kit, PPE cache, ..."""

    resource_id: str
    kind: str                       # foam_unit / fire_unit / rescue_team / spill_kit / ppe_cache / ...
    ppe_level: int = 1
    applicable_classes: frozenset[str] = frozenset()  # empty = all classes
    x_m: float | None = None
    y_m: float | None = None

    def __post_init__(self) -> None:
        if not self.resource_id.strip() or not self.kind.strip():
            raise ValidationError('resource id and kind required')
        if self.ppe_level not in (1, 2, 3):
            raise ValidationError('invalid ppe level')
        if (self.x_m is None) != (self.y_m is None):
            raise ValidationError('resource position must be fully given or absent')

    def handles(self, hazard_class: str) -> bool:
        return not self.applicable_classes or hazard_class in self.applicable_classes


@dataclass(frozen=True)
class MovementStep:
    """One ordered action in the safe-movement sequence."""

    step_no: int
    container_id: str
    role: str            # ignition_source / incompatible_neighbour / water_reactive / neighbour / source
    reason: str
    state: str = 'pending'   # pending / in_progress / done

    def with_state(self, state: str) -> 'MovementStep':
        return MovementStep(self.step_no, self.container_id, self.role, self.reason, state)


@dataclass(frozen=True)
class ResourceAssessment:
    usable: tuple[str, ...]          # applicable assets staging outside the perimeter
    trapped: tuple[str, ...]         # applicable assets inside the perimeter (unreachable)
    unsuitable: tuple[str, ...]      # present but not applicable to the worst-case hazards
    unpositioned: tuple[str, ...]    # no known position; held back until located
    required_ppe_level: int
    missing_media: tuple[str, ...]   # firefighting media no usable resource provides
    missing_ppe: bool                # no usable asset carries the required PPE level


@dataclass(frozen=True)
class IsolationPlan:
    revision: int
    basis: tuple[str, ...]           # human-readable worst-case decision basis
    effective_classes: frozenset[str]
    contradictions: tuple[HazardContradiction, ...]
    isolation_m: float               # hot-zone radius
    evacuation_m: float              # outer planning radius
    avoid_water: bool
    media: tuple[str, ...]
    required_ppe_level: int
    protected_neighbour_ids: frozenset[str]
    resources: ResourceAssessment
    movements: tuple[MovementStep, ...]


@dataclass(frozen=True)
class LevelBand:
    level: str
    max_isolation_m: float           # inclusive upper bound
    label: str


DEFAULT_LEVEL_BANDS: tuple[LevelBand, ...] = (
    LevelBand('level_1', 25.0, 'minor'),
    LevelBand('level_2', 100.0, 'serious'),
    LevelBand('level_3', math.inf, 'major'),
)


@dataclass(frozen=True)
class IncidentEvent:
    seq: int
    at: datetime
    actor: str
    kind: str                        # opened / revised / escalated / stood_down / reopened / move_started / move_done
    detail: dict[str, object] = field(default_factory=dict)


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def _bearing_deg(ax: float, ay: float, bx: float, by: float) -> float:
    angle = math.degrees(math.atan2(bx - ax, by - ay))
    return angle % 360.0


def _downwind(neighbour: ContainerObservation, source: ContainerObservation,
              wind_direction_deg: float | None) -> bool:
    if wind_direction_deg is None:
        return False
    bearing = _bearing_deg(source.x_m, source.y_m, neighbour.x_m, neighbour.y_m)
    delta = abs((bearing - wind_direction_deg + 180.0) % 360.0 - 180.0)
    return delta <= _DOWNWIND_HALF_ANGLE_DEG


class IsolationIncident:
    """State machine holding one incident's plan, movements and audit trail."""

    def __init__(self, incident_id: str, source: ContainerObservation,
                 neighbours: list[ContainerObservation],
                 resources: list[ResponseResource],
                 guidebook: EmergencyGuidebook,
                 opened_at: datetime, actor: str,
                 wind_direction_deg: float | None = None,
                 matrix: CompatibilityMatrix | None = None,
                 level_bands: tuple[LevelBand, ...] = DEFAULT_LEVEL_BANDS):
        if not incident_id.strip():
            raise ValidationError('incident id required')
        if source.container_id in {n.container_id for n in neighbours}:
            raise ValidationError('source is also listed as neighbour')
        if len({r.resource_id for r in resources}) != len(resources):
            raise ValidationError('duplicate resource id')
        self.incident_id = incident_id
        self.source = source
        self.neighbours = list(neighbours)
        self.resources = list(resources)
        self.guidebook = guidebook
        self.wind_direction_deg = wind_direction_deg
        self.matrix = matrix
        self.level_bands = level_bands
        self.status = 'active'
        self.level: str | None = None
        self.plan: IsolationPlan | None = None
        self.events: list[IncidentEvent] = []
        # The first revision after a recorded stand-down/reopen may shrink the
        # perimeter; monotonic worst-case growth is enforced between them.
        self._baseline_reset_pending = False
        self._revise(opened_at, actor, 'opened')

    # ------------------------------------------------------------------ plan

    def _guides_for(self, classes: frozenset[str]):
        try:
            return [self.guidebook.guides[c] for c in sorted(classes)]
        except KeyError as exc:
            raise SafetyConstraintError(f'missing emergency guide for {exc.args[0]}') from exc

    def _build_plan(self, revision: int) -> IsolationPlan:
        # Expand protection until the evacuation envelope stops growing:
        # neighbours dragged in may carry their own (worse) hazard classes.
        radius_iso = radius_evac = 0.0
        included_classes = set(self.source.effective_classes)
        included: dict[str, ContainerObservation] = {}
        if not included_classes:
            raise SafetyConstraintError(
                f'no hazard class known for source {self.source.container_id}')
        basis = [
            f'source {self.source.container_id}: '
            f'declared={sorted(self.source.declared_classes) or "none"} '
            f'placards={sorted(self.source.placard_classes) or "none"} -> '
            f'planned against union {sorted(included_classes)} (most dangerous reading)'
        ]
        while True:
            guides = self._guides_for(frozenset(included_classes))
            new_iso = max(g.isolation_m for g in guides)
            new_evac = max(g.evacuation_m for g in guides)
            grew = new_iso > radius_iso + 1e-09 or new_evac > radius_evac + 1e-09
            radius_iso, radius_evac = new_iso, new_evac
            added = False
            for n in self.neighbours:
                if n.container_id in included:
                    continue
                dist = _distance(self.source.x_m, self.source.y_m, n.x_m, n.y_m)
                threshold = radius_evac if _downwind(n, self.source, self.wind_direction_deg) else radius_iso
                if dist <= threshold + 1e-09:
                    if not n.effective_classes:
                        raise SafetyConstraintError(
                            f'neighbour {n.container_id} inside isolation radius '
                            f'with unknown hazard class')
                    included[n.container_id] = n
                    basis.append(
                        f'neighbour {n.container_id} at {dist:.1f}m '
                        f'(threshold {threshold:.1f}m) included; '
                        f'classes {sorted(n.effective_classes)}')
                    before = len(included_classes)
                    included_classes |= n.effective_classes
                    added = added or len(included_classes) > before
            if not (grew or added):
                break

        guides = self._guides_for(frozenset(included_classes))
        avoid_water = any(g.water_reactive for g in guides)
        media = tuple(sorted({g.firefighting_medium for g in guides}))
        required_ppe = max(g.min_ppe_level for g in guides)
        contradictions = []
        for obs in (self.source, *included.values()):
            if obs.contradictions:
                contradictions.append(HazardContradiction(
                    obs.container_id,
                    obs.declared_classes - obs.placard_classes,
                    obs.placard_classes - obs.declared_classes,
                ))
                basis.append(
                    f'contradiction on {obs.container_id}: declared '
                    f'{sorted(obs.declared_classes)} vs placard '
                    f'{sorted(obs.placard_classes)}; union retained')

        assessment = self._assess_resources(frozenset(included_classes), radius_iso,
                                            required_ppe, media, basis)
        movements = self._build_movements(included, guides)
        return IsolationPlan(
            revision, tuple(basis), frozenset(included_classes),
            tuple(contradictions), radius_iso, radius_evac, avoid_water, media,
            required_ppe, frozenset(included), assessment, tuple(movements))

    def _assess_resources(self, classes: frozenset[str], radius_iso: float,
                          required_ppe: int, media: tuple[str, ...],
                          basis: list[str]) -> ResourceAssessment:
        usable, trapped, unsuitable, unpositioned = [], [], [], []
        for r in self.resources:
            if not all(r.handles(c) for c in classes):
                unsuitable.append(r.resource_id)
                continue
            if r.x_m is None:
                unpositioned.append(r.resource_id)
                continue
            dist = _distance(self.source.x_m, self.source.y_m, r.x_m, r.y_m)
            if dist <= radius_iso + 1e-09:
                trapped.append(r.resource_id)
                basis.append(f'resource {r.resource_id} at {dist:.1f}m is inside the '
                             f'{radius_iso:.1f}m perimeter; not usable until extricated')
            else:
                usable.append(r.resource_id)
        usable_ids = set(usable)
        media_kinds = {'water', 'foam', 'dry_powder', 'co2', 'none'}
        provided = {r.kind for r in self.resources
                    if r.resource_id in usable_ids and r.kind in media_kinds}
        spill_available = any(r.kind == 'spill_kit' for r in self.resources
                              if r.resource_id in usable_ids)
        missing_media = []
        for m in media:
            if m == 'spill_kit':
                if not spill_available:
                    missing_media.append(m)
            elif m not in provided:
                missing_media.append(m)
        missing_ppe = not any(r.ppe_level >= required_ppe for r in self.resources
                              if r.resource_id in usable_ids)
        return ResourceAssessment(
            tuple(sorted(usable)), tuple(sorted(trapped)), tuple(sorted(unsuitable)),
            tuple(sorted(unpositioned)), required_ppe, tuple(missing_media), missing_ppe)

    def _build_movements(self, included: dict[str, ContainerObservation],
                         guides) -> list[MovementStep]:
        steps: list[MovementStep] = []

        def add(container_id: str, role: str, reason: str) -> None:
            steps.append(MovementStep(len(steps) + 1, container_id, role, reason))

        reactive = bool({g.firefighting_medium for g in guides} & {'water', 'foam'})
        ordered = self._order_neighbours(included, reactive)
        for role, container_id, reason in ordered:
            add(container_id, role, reason)
        add(self.source.container_id, 'source',
            'source container moved last, by specialist hazmat team')
        return steps

    def _order_neighbours(self, included: dict[str, ContainerObservation],
                          water_or_foam_planned: bool) -> list[tuple[str, str, str]]:
        reactive_classes = {g.hazard_class for g in self._guides_for(
            frozenset(self.guidebook.guides)) if g.water_reactive}

        def rank(n: ContainerObservation) -> tuple:
            dist = _distance(self.source.x_m, self.source.y_m, n.x_m, n.y_m)
            # 0 = must clear first due to a prohibited/violent reaction pair,
            # 1 = water-reactive neighbour while water/foam is on the media list,
            # 2 = ignition-source class (3 = flammable liquids),
            # 3 = ordinary neighbour; outer rings move before inner ones.
            priority = 3
            reason = 'neighbour moved clear before source is approached'
            if self.matrix is not None:
                for c in n.effective_classes:
                    for s in self.source.effective_classes:
                        try:
                            req = self.matrix.requirement(c, s)
                        except SafetyConstraintError:
                            # No segregation rule on file for this pair: the
                            # guides still bound the hazard, so it does not
                            # determine move priority.
                            continue
                        if req is None:
                            priority = 0
                            reason = f'prohibited reaction pair {c}/{s}; clear first'
            if priority and (n.effective_classes & reactive_classes) and water_or_foam_planned:
                priority = 1
                reason = 'water-reactive neighbour removed before any water/foam application'
            if priority > 2 and '3' in n.effective_classes:
                priority = 2
                reason = 'ignition-source neighbour moved clear before hot work'
            return (priority, -dist, n.container_id, reason)

        ranked = [(rank(n), n) for n in included.values()]
        return self._flatten_rank(ranked)

    @staticmethod
    def _flatten_rank(ranked) -> list[tuple[str, str, str]]:
        out = []
        for key, n in sorted(ranked, key=lambda item: item[0]):
            role = {0: 'incompatible_neighbour', 1: 'water_reactive',
                    2: 'ignition_source', 3: 'neighbour'}[key[0]]
            out.append((role, n.container_id, key[3]))
        return out

    # --------------------------------------------------------------- workflow

    def revise(self, at: datetime, actor: str, *,
               source: ContainerObservation | None = None,
               neighbours: list[ContainerObservation] | None = None,
               resources: list[ResponseResource] | None = None,
               wind_direction_deg: float | None = None,
               wind_unknown: bool = False) -> IsolationPlan:
        """Replan against fresh scene information (an escalation or same-level revision)."""
        if self.status != 'active':
            raise StateTransitionError('incident is stood down; reopen before revising')
        if source is not None:
            self.source = source
        if neighbours is not None:
            self.neighbours = list(neighbours)
        if resources is not None:
            self.resources = list(resources)
        if wind_unknown:
            self.wind_direction_deg = None
        elif wind_direction_deg is not None:
            self.wind_direction_deg = wind_direction_deg
        return self._revise(at, actor, 'revised')

    def _revise(self, at: datetime, actor: str, kind: str) -> IsolationPlan:
        at = ensure_utc(at)
        old_plan = self.plan
        candidate = self._build_plan((old_plan.revision + 1) if old_plan else 1)

        if old_plan is not None:
            # An action already entered the isolation zone: it cannot be covered
            # by a new plan. In-progress steps block replanning outright; done
            # steps must survive with the same step number and container.
            old_by_container = {s.container_id: s for s in old_plan.movements}
            new_by_container = {s.container_id: s for s in candidate.movements}
            for old in old_plan.movements:
                if old.state == 'in_progress':
                    raise ConflictError(
                        f'step {old.step_no} ({old.container_id}) is in progress; '
                        'finish or abort it before replanning')
                if old.state == 'done':
                    new = new_by_container.get(old.container_id)
                    if new is None or new.step_no != old.step_no:
                        raise ConflictError(
                            f'executed movement {old.container_id} cannot be overridden '
                            'by the new plan')
            # Carry executed states forward, preserving identity/ordering.
            candidate = IsolationPlan(
                candidate.revision, candidate.basis, candidate.effective_classes,
                candidate.contradictions, candidate.isolation_m, candidate.evacuation_m,
                candidate.avoid_water, candidate.media, candidate.required_ppe_level,
                candidate.protected_neighbour_ids, candidate.resources,
                tuple(s.with_state(old_by_container[s.container_id].state)
                      if s.container_id in old_by_container
                      and old_by_container[s.container_id].state == 'done'
                      else s for s in candidate.movements))
            # Worst-case information may never shrink the hot zone on a live
            # plan; the *physical envelope* must grow or hold. The hazard-class
            # set itself may change (a placard re-read can correct a class to a
            # different, more dangerous one) and is fully recorded in the trail.
            if not self._baseline_reset_pending:
                if candidate.isolation_m + 1e-09 < old_plan.isolation_m:
                    raise SafetyConstraintError(
                        'revision would shrink the isolation radius; stand down to downgrade')
        self._baseline_reset_pending = False

        self.plan = candidate
        new_level = self._level_for(candidate.isolation_m)
        detail = {'revision': candidate.revision,
                  'isolation_m': candidate.isolation_m,
                  'evacuation_m': candidate.evacuation_m,
                  'level': new_level,
                  'classes': sorted(candidate.effective_classes),
                  'contradictions': [c.container_id for c in candidate.contradictions],
                  'missing_media': list(candidate.resources.missing_media),
                  'missing_ppe': candidate.resources.missing_ppe}
        if old_plan is not None and self._band_rank(new_level) > self._band_rank(self.level):
            kind = 'escalated'
            detail['from_level'] = self.level
        self.level = new_level
        self._append(at, actor, kind, detail)
        return candidate

    def stand_down(self, at: datetime, actor: str, reason: str) -> IsolationPlan:
        """Downgrade/close. Records the explicit release; does not delete history."""
        if self.status != 'active':
            raise StateTransitionError('incident already stood down')
        if not reason.strip():
            raise ValidationError('stand-down reason required')
        self.status = 'stood_down'
        self._append(at, actor, 'stood_down', {'reason': reason, 'level': self.level})
        return self.plan

    def reopen(self, at: datetime, actor: str, reason: str) -> IsolationPlan:
        if self.status != 'stood_down':
            raise StateTransitionError('incident is active')
        if not reason.strip():
            raise ValidationError('reopen reason required')
        self.status = 'active'
        # Next revision is the new, re-evaluated worst-case baseline and may
        # shrink the perimeter once; subsequent revisions are monotonic again.
        self._baseline_reset_pending = True
        self._append(at, actor, 'reopened', {'reason': reason})
        return self.plan

    def start_movement(self, at: datetime, actor: str, step_no: int) -> MovementStep:
        self._require_active_step(step_no, {'pending'})
        old = self.plan.movements[step_no - 1]
        self._set_step(old.with_state('in_progress'))
        self._append(at, actor, 'move_started',
                     {'step_no': step_no, 'container_id': old.container_id,
                      'revision': self.plan.revision})
        return self.plan.movements[step_no - 1]

    def complete_movement(self, at: datetime, actor: str, step_no: int) -> MovementStep:
        self._require_active_step(step_no, {'in_progress'})
        old = self.plan.movements[step_no - 1]
        self._set_step(old.with_state('done'))
        self._append(at, actor, 'move_done',
                     {'step_no': step_no, 'container_id': old.container_id,
                      'revision': self.plan.revision})
        return self.plan.movements[step_no - 1]

    def _require_active_step(self, step_no: int, expected: set[str]) -> None:
        if self.status != 'active':
            raise StateTransitionError('incident is stood down')
        if self.plan is None or not 1 <= step_no <= len(self.plan.movements):
            raise ValidationError('unknown movement step')
        state = self.plan.movements[step_no - 1].state
        if state not in expected:
            raise ConflictError(f'step {step_no} is {state}, expected one of {sorted(expected)}')

    def _set_step(self, updated: MovementStep) -> None:
        steps = list(self.plan.movements)
        steps[updated.step_no - 1] = updated
        self.plan = IsolationPlan(
            self.plan.revision, self.plan.basis, self.plan.effective_classes,
            self.plan.contradictions, self.plan.isolation_m, self.plan.evacuation_m,
            self.plan.avoid_water, self.plan.media, self.plan.required_ppe_level,
            self.plan.protected_neighbour_ids, self.plan.resources, tuple(steps))

    def _level_for(self, isolation_m: float) -> str:
        for band in self.level_bands:
            if isolation_m <= band.max_isolation_m + 1e-09:
                return band.level
        raise SafetyConstraintError('no level band covers isolation radius')

    def _band_rank(self, level: str | None) -> int:
        order = [b.level for b in self.level_bands]
        return order.index(level) if level in order else -1

    # ------------------------------------------------------------------ audit

    def _append(self, at: datetime, actor: str, kind: str, detail: dict[str, object]) -> None:
        if not actor.strip():
            raise ValidationError('actor required')
        self.events.append(IncidentEvent(len(self.events) + 1, ensure_utc(at), actor, kind, detail))

    def trail(self) -> tuple[IncidentEvent, ...]:
        return tuple(self.events)
