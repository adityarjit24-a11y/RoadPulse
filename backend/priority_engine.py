from dataclasses import dataclass, asdict
from typing import Literal

PriorityBand = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]

WEIGHTS = {
    "severity": 0.35,
    "road_importance": 0.20,
    "proximity": 0.15,
    "report_count": 0.15,
    "age": 0.10,
    "confidence": 0.05,
}

ROAD_IMPORTANCE_WEIGHTS = {
    "arterial": 1.0,
    "collector": 0.7,
    "residential": 0.5,
    "unknown": 0.5,
}

MAX_AGE_DAYS_FOR_FULL_SCORE = 60
REPORT_COUNT_SATURATION = 10

@dataclass
class PriorityBreakdown:
    severity_component: float
    road_importance_component: float
    proximity_component: float
    report_count_component: float
    age_component: float
    confidence_component: float
    total_score: float
    band: PriorityBand

    def as_dict(self):
        return asdict(self)

def _band_for_score(score: float) -> PriorityBand:
    if score <= 40: return "LOW"
    if score <= 65: return "MEDIUM"
    if score <= 85: return "HIGH"
    return "CRITICAL"

def compute_priority(
    severity_score: float,
    road_type: str,
    near_sensitive_site: bool,
    citizen_report_count: int,
    days_unresolved: int,
    detection_confidence: float,
) -> PriorityBreakdown:
    
    road_importance_weight = ROAD_IMPORTANCE_WEIGHTS.get(road_type, ROAD_IMPORTANCE_WEIGHTS["unknown"])
    proximity_weight = 1.0 if near_sensitive_site else 0.3
    report_count_factor = min(citizen_report_count / REPORT_COUNT_SATURATION, 1.0)
    age_factor = min(days_unresolved / MAX_AGE_DAYS_FOR_FULL_SCORE, 1.0)

    severity_component = WEIGHTS["severity"] * severity_score
    road_importance_component = WEIGHTS["road_importance"] * road_importance_weight * 100
    proximity_component = WEIGHTS["proximity"] * proximity_weight * 100
    report_count_component = WEIGHTS["report_count"] * report_count_factor * 100
    age_component = WEIGHTS["age"] * age_factor * 100
    confidence_component = WEIGHTS["confidence"] * detection_confidence * 100

    total = sum([severity_component, road_importance_component, proximity_component, report_count_component, age_component, confidence_component])
    total = round(min(total, 100.0), 2)

    return PriorityBreakdown(
        severity_component=round(severity_component, 2),
        road_importance_component=round(road_importance_component, 2),
        proximity_component=round(proximity_component, 2),
        report_count_component=round(report_count_component, 2),
        age_component=round(age_component, 2),
        confidence_component=round(confidence_component, 2),
        total_score=total,
        band=_band_for_score(total),
    )