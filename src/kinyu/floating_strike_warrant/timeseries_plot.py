"""Plot the floating-strike warrant price path and contractual bounds."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import floating_strike_warrant as fsw


DOCS_DIR = Path(__file__).resolve().parent / "docs"
DEFAULT_OUTPUT = DOCS_DIR / "timeseries_plot.png"

BASELINE: Dict[str, Any] = {
    "initial_price": 100.0,
    "risk_free_rate": 0.02,
    "volatility": 0.25,
    "maturity": 1.0,
    "steps_per_year": 252,
    "strike_discount": 0.9,
    "strike_reset_steps": 5,
    "buyback_price": 25.0,
    "holder_put_trigger_price": 75.0,
    "holder_put_price": 6.0,
    "exercise_limit_fraction": 0.25,
    "exercise_limit_period_steps": 21,
    "next_limit_reset_step": 0,
    "exercised_fraction_current_period": 0.0,
    "num_paths": 3000,
    "seed": 7,
}


def fetch_timeseries(params: Dict[str, Any]) -> Tuple[float, List[Tuple[int, float, float, float, float, float]]]:
    """Run the Monte Carlo pricer and return the time series tuple list."""

    price, series = fsw.price_warrant_timeseries_py(**params)
    return price, series


def plot_series(
    series: Iterable[Tuple[int, float, float, float, float, float]],
    output_path: Path,
    show: bool = False,
) -> Path:
    """Render the time series chart for the warrant value and bounds.

    Parameters
    ----------
    series
        Iterable of tuples returned from the pricing routine.
    output_path
        Destination where the PNG file will be written.
    show
        Whether to display the plot interactively after saving.
    """

    _, times, expected, exercise_cap, put_floor, buyback_cap = zip(*series)
    buyback_line = list(buyback_cap)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, buyback_line, label="Issuer buyback cap", linestyle="--")
    ax.plot(times, exercise_cap, label="Holder exercise upper bound", linestyle=":")
    ax.plot(times, put_floor, label="Holder put lower bound", linestyle="-.")
    ax.plot(times, expected, label="Expected warrant price", linewidth=2.0)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Value")
    ax.set_title("Floating-Strike Warrant Monte Carlo valuation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a PNG plot of floating-strike warrant valuation bounds."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Path where the PNG file will be saved. Defaults to"
            f" {DEFAULT_OUTPUT}"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the generated plot after saving the PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = {**BASELINE}
    price, series = fetch_timeseries(params)
    print(f"Simulated warrant price: {price:.6f}")
    output = plot_series(series, args.output, show=args.show)
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()
