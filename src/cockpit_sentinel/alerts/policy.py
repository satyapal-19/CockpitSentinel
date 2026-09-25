"""Configuration-backed alert scoring policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cockpit_sentinel.domain import SIGNAL_WEIGHTS, DriverSignals, RiskAssessment, assess_risk


class PolicyConfigurationError(ValueError):
    """Raised when alert policy configuration is incomplete or invalid."""


@dataclass(frozen=True, slots=True)
class AlertPolicy:
    """Risk weights and thresholds used by the monitoring pipeline."""

    weights: Mapping[str, int]
    caution_score: int
    warning_score: int
    critical_score: int
    time_window_seconds: float = 0.0

    def assess(self, signals: DriverSignals) -> RiskAssessment:
        return assess_risk(
            signals,
            weights=self.weights,
            caution_score=self.caution_score,
            warning_score=self.warning_score,
            critical_score=self.critical_score,
        )


DEFAULT_POLICY = AlertPolicy(
    weights=SIGNAL_WEIGHTS,
    caution_score=2,
    warning_score=4,
    critical_score=6,
    time_window_seconds=0.0,
)


def load_alert_policy(path: Path) -> AlertPolicy:
    """Load and validate alert settings from a YAML file."""

    with path.open(encoding="utf-8") as config_file:
        raw_config: Any = yaml.safe_load(config_file)

    if not isinstance(raw_config, dict):
        raise PolicyConfigurationError("Alert policy must be a YAML mapping.")

    weights = raw_config.get("weights")
    thresholds = raw_config.get("thresholds")
    if not isinstance(weights, dict) or not isinstance(thresholds, dict):
        raise PolicyConfigurationError("Alert policy needs weights and thresholds mappings.")

    expected_signals = set(SIGNAL_WEIGHTS)
    if set(weights) != expected_signals:
        raise PolicyConfigurationError("Alert policy weights must define every driver signal.")

    if not all(isinstance(value, int) and value > 0 for value in weights.values()):
        raise PolicyConfigurationError("Alert policy weights must be positive integers.")

    try:
        caution_score = thresholds["caution"]
        warning_score = thresholds["warning"]
        critical_score = thresholds["critical"]
    except KeyError as error:
        raise PolicyConfigurationError(f"Missing alert threshold: {error.args[0]}") from error

    scores = (caution_score, warning_score, critical_score)
    if not all(isinstance(value, int) and value > 0 for value in scores):
        raise PolicyConfigurationError("Alert thresholds must be positive integers.")
    if not caution_score < warning_score < critical_score:
        raise PolicyConfigurationError("Alert thresholds must increase from caution to critical.")

    time_window_raw = raw_config.get("time_window_seconds", 0.0)
    time_window_seconds = float(time_window_raw) if time_window_raw is not None else 0.0

    return AlertPolicy(
        weights=weights,
        caution_score=caution_score,
        warning_score=warning_score,
        critical_score=critical_score,
        time_window_seconds=time_window_seconds,
    )
