from __future__ import annotations
from dataclasses import dataclass
from terminalweave.core.errors import ValidationError

@dataclass(frozen=True)
class EmergencyGuide:
    hazard_class: str
    isolation_m: float
    evacuation_m: float
    water_reactive: bool
    firefighting_medium: str
    min_ppe_level: int = 1

    def __post_init__(self):
        if min(self.isolation_m, self.evacuation_m) < 0 or self.evacuation_m < self.isolation_m:
            raise ValidationError('invalid emergency distances')
        if self.min_ppe_level not in (1, 2, 3):
            raise ValidationError('invalid ppe level')

class EmergencyGuidebook:

    def __init__(self, guides: list[EmergencyGuide]):
        self.guides = {g.hazard_class: g for g in guides}

    def plan(self, hazard_classes: set[str], wind_direction_deg: float | None=None) -> dict[str, object]:
        if not hazard_classes:
            raise ValidationError('hazard class required')
        selected = [self.guides[x] for x in hazard_classes if x in self.guides]
        if len(selected) != len(hazard_classes):
            raise ValidationError('missing emergency guide')
        return {'isolation_m': max((g.isolation_m for g in selected)), 'evacuation_m': max((g.evacuation_m for g in selected)), 'avoid_water': any((g.water_reactive for g in selected)), 'media': sorted({g.firefighting_medium for g in selected}), 'wind_direction_deg': wind_direction_deg}
