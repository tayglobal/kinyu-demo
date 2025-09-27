# Interest Rate Curve Builder

This library provides tools to construct an interest rate discount curve from a set of market instruments using a global optimization approach. Instead of bootstrapping the curve instrument by instrument, this method finds a single curve that collectively best fits all the provided market data points.

## Core Concept: Global Optimization

The fundamental idea is to define the interest rate curve by a set of parameters and then use a numerical optimizer to find the parameters that minimize the difference between the market prices of the instruments and the prices calculated from our curve.

### 1. Curve Representation

The discount curve is represented by a set of zero-coupon rates at specific **pillar dates** ($T_1, T_2, ..., T_n$). Let the vector of these zero rates be $\vec{r} = (r_1, r_2, ..., r_n)$.

The discount factor $DF(t)$ for any time $t$ is calculated from the zero rate $Z(t)$ as:

$$
DF(t) = e^{-Z(t) \cdot t}
$$

The zero rate $Z(t)$ for a time $t$ that falls between two pillar dates $T_i$ and $T_{i+1}$ is found using linear interpolation on the zero rates.

### 2. The Objective Function (Cost Function)

The goal of the optimization is to find the vector of zero rates $\vec{r}$ that minimizes the pricing error across all instruments. We define a **cost function**, which is the sum of the squared differences between the observed market rate for each instrument and the rate implied by the curve defined by $\vec{r}$.

The objective is to solve:

$$
\min_{\vec{r}} \sum_{i=1}^{N} \left( \text{Rate}_{\text{market}, i} - \text{Rate}_{\text{curve}, i}(\vec{r}) \right)^2
$$

Where:
- $N$ is the number of market instruments.
- $\text{Rate}_{\text{market}, i}$ is the quoted market rate of the $i$-th instrument.
- $\text{Rate}_{\text{curve}, i}(\vec{r})$ is the par rate of the $i$-th instrument calculated using the curve defined by the zero rates $\vec{r}$.

This library uses the **Nelder-Mead** algorithm, a gradient-free optimization method, to perform this minimization.

## Instrument Pricing Formulas

Here are the formulas used to calculate the curve-implied par rate for each instrument type. Let $DF(t)$ be the discount factor for a future time $t$ derived from the curve.

### Overnight Index Swap (OIS)

An OIS is treated as a single-period swap. The par rate is derived from the discount factor at its maturity $T$.

$$
\text{Rate}_{\text{OIS}} = \frac{1}{\tau} \left( \frac{1}{DF(T)} - 1 \right)
$$

Where $\tau$ is the year fraction for the period of the OIS.

### Forward Rate Agreement (FRA)

A FRA is an agreement on an interest rate for a future period between $T_{start}$ and $T_{end}$. The forward rate implied by the curve is:

$$
\text{Rate}_{\text{FRA}} = \frac{1}{\tau} \left( \frac{DF(T_{start})}{DF(T_{end})} - 1 \right)
$$

Where $\tau$ is the year fraction of the forward period. Interest Rate Futures are priced using the same formula.

### Interest Rate Swap (IRS)

An interest rate swap has a fixed leg and a floating leg. The par swap rate is the fixed rate that makes the Net Present Value (NPV) of the swap equal to zero.

$$
\text{NPV} = \text{PV}_{\text{floating}} - \text{PV}_{\text{fixed}} = 0
$$

The Present Value (PV) of the floating leg (for a standard swap with unit notional) is:

$$
\text{PV}_{\text{floating}} = DF(T_{start}) - DF(T_{end})
$$

The PV of the fixed leg is the swap rate $R$ multiplied by the annuity factor $A$:

$$
\text{PV}_{\text{fixed}} = R \times A \quad \text{where} \quad A = \sum_{i=1}^{n} \tau_i \cdot DF(T_i)
$$

Setting the PVs equal gives the par swap rate:

$$
\text{Rate}_{\text{Swap}} = R = \frac{\text{PV}_{\text{floating}}}{A} = \frac{DF(T_{start}) - DF(T_{end})}{\sum_{i=1}^{n} \tau_i \cdot DF(T_i)}
$$

Where $T_i$ are the payment dates of the fixed leg and $\tau_i$ are the corresponding year fractions.