"""Demonstration of the floating-strike warrant Python binding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import floating_strike_warrant as fsw


@dataclass(frozen=True)
class Scenario:
    label: str
    focus: Optional[str]
    updates: Dict[str, Any]


BASELINE: Dict[str, Any] = {
    "initial_price": 100.0,
    "risk_free_rate": 0.02,
    "volatility": 0.25,
    "maturity": 1.0,
    "steps_per_year": 252,
    "strike_discount": 0.9,
    "strike_reset_steps": 5,
    "buyback_price": 25.0,
    "exercise_limit_fraction": 0.25,
    "exercise_limit_period_steps": 21,
    "next_limit_reset_step": 0,
    "exercised_fraction_current_period": 0.0,
    "num_paths": 3000,
    "seed": 7,
}


SCENARIOS = [
    Scenario("Baseline", None, {}),
    Scenario("Spot -5%", "Spot", {"initial_price": BASELINE["initial_price"] * 0.95}),
    Scenario("Spot +5%", "Spot", {"initial_price": BASELINE["initial_price"] * 1.05}),
    Scenario("Volatility 15%", "Volatility", {"volatility": 0.15}),
    Scenario("Volatility 35%", "Volatility", {"volatility": 0.35}),
    Scenario("Rate 0%", "Rate", {"risk_free_rate": 0.0}),
    Scenario("Rate 4%", "Rate", {"risk_free_rate": 0.04}),
    Scenario("Buyback 2", "Buyback", {"buyback_price": 2.0}),
    Scenario("Buyback 3", "Buyback", {"buyback_price": 3.0}),
    Scenario("Discount 85%", "Discount", {"strike_discount": 0.85}),
    Scenario("Discount 95%", "Discount", {"strike_discount": 0.95}),
    Scenario("Quota 10%", "Quota", {"exercise_limit_fraction": 0.10}),
    Scenario("Quota 40%", "Quota", {"exercise_limit_fraction": 0.40}),
]


def main() -> None:
    rows = []
    for scenario in SCENARIOS:
        params = {**BASELINE, **scenario.updates}
        price = fsw.price_warrant_py(**params)
        value = "-"
        if scenario.focus is not None:
            key_map = {
                "Spot": "initial_price",
                "Volatility": "volatility",
                "Rate": "risk_free_rate",
                "Buyback": "buyback_price",
                "Discount": "strike_discount",
                "Quota": "exercise_limit_fraction",
            }
            source_key = key_map[scenario.focus]
            value = f"{params[source_key]:.4f}"
        rows.append((scenario.label, value, price))

    print("| Scenario | Key value | Price |")
    print("| --- | --- | --- |")
    for label, value, price in rows:
        print(f"| {label} | {value} | {price:.6f} |")


if __name__ == "__main__":
    main()
