# Floating-Strike Warrant Timeseries Plot

The `timeseries_plot.py` helper generates a chart showing the simulated value of the
floating-strike warrant alongside the contractual bounds enforced during the
Longstaff–Schwartz pricing routine.

## Usage

From the repository root run:

```bash
python src/kinyu/floating_strike_warrant/timeseries_plot.py
```

By default the script writes `timeseries_plot.png` into the documentation folder.
You can override the destination or display the plot interactively:

```bash
python src/kinyu/floating_strike_warrant/timeseries_plot.py \
  --output /tmp/custom_plot.png \
  --show
```

## Output

![Floating-strike warrant bounds](timeseries_plot.png)
