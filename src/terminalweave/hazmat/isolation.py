from __future__ import annotations
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from terminalweave.core.clock import ensure_utc
from terminalweave.core.errors import ValidationError, StateTransitionError, NotFoundError, SafetyConstraintError
from terminalweave.hazmat.compatibility import CompatibilityMatrix
from terminalweave.hazmat.emergency import EmergencyGuidebook
from terminalweave.hazmat.placards import DeclaredHazmat, PlacardObservation

class IncidentLevel(str, Enum):
    WATCH = 'watch'
    ISOLATED = 'isolated'
    ESCALATED = 'escalated'
    STOOD_DOWN = 'stood_down'

class ActionState(str, Enum):
    PLANNED = 'planned'
    COMMITTED = 'committed'
    COMPLETED = 'completed'

@dataclass(frozen=True)
class ResponseResource:
    equipment_id: str
    medium: str
    available: bool = True

@dataclass(frozen=True)
class AdjacentContainer:
    container_id: str
    distance_m: float
    declared_classes: frozenset[str] = frozenset()
    observed_placards: frozenset[str] = frozenset()

    def __post_init__(self):
        if not self.container_id.strip() or self.distance_m < 0:
            raise ValidationError('invalid adjacent container')

    @property
    def worst_case_classes(self) -> frozenset[str]:
        return self.declared_classes | self.observed_placards

@dataclass(frozen=True)
class MoveAction:
    sequence: int
    container_id: str
    reason: str
    state: ActionState = ActionState.PLANNED

@dataclass(frozen=True)
class IsolationPlan:
    revision: int
    level: IncidentLevel
    hazard_classes: frozenset[str]
    conflicting_evidence: bool
    isolation_m: float
    evacuation_m: float
    avoid_water: bool
    media: tuple[str, ...]
    usable_resources: tuple[str, ...]
    move_order: tuple[MoveAction, ...]

@dataclass(frozen=True)
class AuditEntry:
    revision: int
    at: datetime
    actor: str
    event: str
    detail: str

    def __post_init__(self):
        object.__setattr__(self, 'at', ensure_utc(self.at))

@dataclass
class IsolationIncident:
    incident_id: str
    container_id: str
    created_at: datetime
    level: IncidentLevel = IncidentLevel.ISOLATED
    revision: int = 0
    plan: IsolationPlan | None = None
    history: list[AuditEntry] = field(default_factory=list)

    def __post_init__(self):
        self.created_at = ensure_utc(self.created_at)

    @property
    def active(self) -> bool:
        return self.level != IncidentLevel.STOOD_DOWN

class IsolationController:

    def __init__(self, guidebook: EmergencyGuidebook, matrix: CompatibilityMatrix | None=None):
        self.guidebook = guidebook
        self.matrix = matrix
        self._incidents: dict[str, IsolationIncident] = {}

    def get(self, incident_id: str) -> IsolationIncident:
        try:
            return self._incidents[incident_id]
        except KeyError as exc:
            raise NotFoundError('unknown incident') from exc

    def audit_trail(self, incident_id: str) -> tuple[AuditEntry, ...]:
        return tuple(self.get(incident_id).history)

    def actions_in_zone(self, incident_id: str) -> tuple[MoveAction, ...]:
        plan = self.get(incident_id).plan
        if not plan:
            return ()
        return tuple((a for a in plan.move_order if a.state == ActionState.COMMITTED))

    def open_incident(self, incident_id: str, container_id: str, declared: DeclaredHazmat, observed: PlacardObservation | None, adjacent: list[AdjacentContainer], resources: list[ResponseResource], at: datetime, actor: str, level: IncidentLevel=IncidentLevel.ISOLATED, wind_direction_deg: float | None=None) -> IsolationIncident:
        at = ensure_utc(at)
        if not incident_id.strip() or not actor.strip():
            raise ValidationError('incident id and actor required')
        if level == IncidentLevel.STOOD_DOWN:
            raise ValidationError('incident cannot open stood down')
        if incident_id in self._incidents:
            raise ValidationError('incident id already exists')
        self._check_container(container_id, declared, observed)
        incident = IsolationIncident(incident_id, container_id, at, level=level)
        incident.plan = self._build_plan(incident, declared, observed, adjacent, resources, keep=(), wind_direction_deg=wind_direction_deg)
        self._incidents[incident_id] = incident
        self._record(incident, at, actor, 'opened', f'level={level.value} classes={sorted(incident.plan.hazard_classes)}')
        return incident

    def replan(self, incident_id: str, declared: DeclaredHazmat, observed: PlacardObservation | None, adjacent: list[AdjacentContainer], resources: list[ResponseResource], at: datetime, actor: str, wind_direction_deg: float | None=None) -> IsolationIncident:
        incident = self._require_active(incident_id)
        at = ensure_utc(at)
        self._check_container(incident.container_id, declared, observed)
        keep = tuple((a for a in incident.plan.move_order if a.state != ActionState.PLANNED))
        incident.plan = self._build_plan(incident, declared, observed, adjacent, resources, keep=keep, wind_direction_deg=wind_direction_deg)
        self._record(incident, at, actor, 'replanned', f'kept {len(keep)} zone action(s)')
        return incident

    def commit_action(self, incident_id: str, container_id: str, at: datetime, actor: str) -> MoveAction:
        return self._advance_action(incident_id, container_id, ActionState.PLANNED, ActionState.COMMITTED, at, actor, 'action_committed')

    def complete_action(self, incident_id: str, container_id: str, at: datetime, actor: str) -> MoveAction:
        return self._advance_action(incident_id, container_id, ActionState.COMMITTED, ActionState.COMPLETED, at, actor, 'action_completed')

    def escalate(self, incident_id: str, at: datetime, actor: str, reason: str) -> IsolationIncident:
        incident = self._require_active(incident_id)
        at = ensure_utc(at)
        if not reason.strip():
            raise ValidationError('escalation reason required')
        order = [IncidentLevel.WATCH, IncidentLevel.ISOLATED, IncidentLevel.ESCALATED]
        index = order.index(incident.level)
        if index == len(order) - 1:
            raise StateTransitionError('incident already at highest level')
        incident.level = order[index + 1]
        incident.plan = replace(incident.plan, level=incident.level)
        self._record(incident, at, actor, 'escalated', f'{incident.level.value}: {reason}')
        return incident

    def stand_down(self, incident_id: str, at: datetime, actor: str, reason: str) -> IsolationIncident:
        incident = self._require_active(incident_id)
        at = ensure_utc(at)
        if not reason.strip():
            raise ValidationError('stand-down reason required')
        if self.actions_in_zone(incident_id):
            raise StateTransitionError('actions still inside the isolation zone')
        incident.level = IncidentLevel.STOOD_DOWN
        incident.plan = replace(incident.plan, level=incident.level)
        self._record(incident, at, actor, 'stood_down', reason)
        return incident

    def _advance_action(self, incident_id: str, container_id: str, expected: ActionState, target: ActionState, at: datetime, actor: str, event: str) -> MoveAction:
        incident = self._require_active(incident_id)
        at = ensure_utc(at)
        plan = incident.plan
        for action in plan.move_order:
            if action.container_id == container_id:
                if action.state != expected:
                    raise StateTransitionError(f'action is {action.state.value}, expected {expected.value}')
                updated = MoveAction(action.sequence, action.container_id, action.reason, target)
                order = tuple((updated if a.sequence == action.sequence else a for a in plan.move_order))
                incident.plan = replace(plan, move_order=order)
                self._record(incident, at, actor, event, container_id)
                return updated
        raise NotFoundError('container not in move order')

    def _build_plan(self, incident: IsolationIncident, declared: DeclaredHazmat, observed: PlacardObservation | None, adjacent: list[AdjacentContainer], resources: list[ResponseResource], keep: tuple[MoveAction, ...], wind_direction_deg: float | None) -> IsolationPlan:
        classes = frozenset(declared.classes) | (frozenset(observed.placards) if observed else frozenset())
        if not classes:
            raise ValidationError('hazard class required')
        conflicting = observed is not None and (set(declared.classes) != set(observed.placards) or set(declared.un_numbers) != set(observed.un_numbers))
        guide = self.guidebook.plan(set(classes), wind_direction_deg)
        isolation_m = guide['isolation_m']
        excluded = {'water'} if guide['avoid_water'] else set()
        usable = tuple(sorted((r.equipment_id for r in resources if r.available and r.medium not in excluded)))
        kept_ids = {a.container_id for a in keep}
        sequence = max((a.sequence for a in keep), default=0)
        move_order = list(keep)
        candidates = [b for b in adjacent if b.container_id != incident.container_id and b.container_id not in kept_ids and b.distance_m <= isolation_m]
        candidates.sort(key=lambda b: (self._risk_rank(b, classes), round(b.distance_m, 6), b.container_id))
        for box in candidates:
            sequence += 1
            reason = 'incompatible or under-separated cargo' if self._risk_rank(box, classes) == 0 else 'within isolation radius'
            move_order.append(MoveAction(sequence, box.container_id, reason))
        return IsolationPlan(incident.revision + 1, incident.level, classes, conflicting, isolation_m, guide['evacuation_m'], guide['avoid_water'], tuple(guide['media']), usable, tuple(move_order))

    def _risk_rank(self, box: AdjacentContainer, incident_classes: frozenset[str]) -> int:
        for own in box.worst_case_classes:
            for other in incident_classes:
                if self._pair_urgent(own, other, box.distance_m):
                    return 0
        return 1

    def _pair_urgent(self, a: str, b: str, distance_m: float) -> bool:
        if self.matrix is None or a == b:
            return False
        try:
            requirement = self.matrix.requirement(a, b)
        except SafetyConstraintError:
            return True
        if requirement is None:
            return True
        return distance_m + 1e-09 < requirement

    def _require_active(self, incident_id: str) -> IsolationIncident:
        incident = self.get(incident_id)
        if not incident.active:
            raise StateTransitionError('incident already stood down')
        return incident

    def _record(self, incident: IsolationIncident, at: datetime, actor: str, event: str, detail: str) -> None:
        incident.revision += 1
        incident.history.append(AuditEntry(incident.revision, at, actor, event, detail))

    @staticmethod
    def _check_container(container_id: str, declared: DeclaredHazmat, observed: PlacardObservation | None) -> None:
        if declared.container_id != container_id or (observed is not None and observed.container_id != container_id):
            raise ValidationError('container ids differ')
