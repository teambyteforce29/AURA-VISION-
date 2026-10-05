"""Condition Scoring Engine.

Maps damage severity metrics to overall asset condition score (0-100) and operational status.

Scoring Scale:
• 80–100: Excellent
• 60–79:  Good
• 40–59:  Fair (or Needs Maintenance)
• 20–39:  Poor
• 0–19:   Critical
"""

from backend.schemas.results import ConditionResult, SeverityResult


def calculate_condition(severity: SeverityResult, defect_count: int = 0) -> ConditionResult:
    """Calculates asset condition score and status label from severity results.

    Higher severity implies lower structural condition.
    Condition is calibrated so that an asset with 0 defects has score ~100 (Excellent),
    and severe damage degrades condition toward 0 (Critical).

    Args:
        severity: Evaluated SeverityResult containing score (0-100) and level.
        defect_count: Total number of identified defects.

    Returns:
        ConditionResult: Object containing condition score and human-readable status.
    """
    severity_score = severity.score

    # Compute base condition score as inverse of severity
    if defect_count == 0 and severity_score == 0:
        condition_score = 100
        status = "Excellent"
        return ConditionResult(score=condition_score, status=status)

    # Damage degradation curve
    condition_score = max(0, min(100, int(round(100 - (severity_score * 0.95)))))

    # Map score to standard infrastructure condition brackets
    if condition_score >= 80:
        status = "Excellent"
    elif condition_score >= 60:
        status = "Good"
    elif condition_score >= 40:
        # Standard maintenance trigger bracket (e.g. 58/100 -> "Needs Maintenance")
        status = "Needs Maintenance"
    elif condition_score >= 20:
        status = "Poor"
    else:
        status = "Critical"

    return ConditionResult(score=condition_score, status=status)
