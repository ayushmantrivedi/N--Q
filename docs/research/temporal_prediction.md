# Temporal Prediction in Nāḍī-Q

The Nāḍī-Q framework is designed to evaluate longitudinal disease progression models with mathematical rigor to ensure **zero temporal data leakage**.

## The Mathematical Objective

We formalize longitudinal prediction as predicting future outcomes based on strictly left-truncated histories:

$$X_{\leq T} \rightarrow Y_{T+H}$$

Where:
- $T$ is the snapshot time (e.g., the date of a patient's baseline MCI visit).
- $X_{\leq T}$ is the set of all available data (measurements, demographics) collected strictly on or before $T$.
- $H$ is the prediction horizon (e.g., 24 months).
- $Y_{T+H}$ is the clinical outcome evaluated according to strict horizon semantics.

## Horizon Semantics

A horizon defines not just *when* to look for an outcome, but *how* to handle the interval. We implement three strict definitions:

### 1. EXACT
The outcome must be observed exactly at $T + H$ (within a configurable `window_days` margin).
- Example use case: Scheduled clinical trial visits where you want to predict the state at the exact 24-month check-in.
- **Censoring**: If a patient misses the $T+H$ visit, their outcome is `CENSORED`, even if they were observed at $T+12$ and $T+36$.

### 2. WITHIN
The outcome occurs at any point in the interval $(T, T+H]$.
- Example use case: Predicting whether an event (like MCI to AD conversion) will happen at any point in the next 24 months.
- **Positives**: The event is observed $\leq T+H$.
- **Negatives**: The event is *not* observed, AND we have explicit follow-up $\geq T+H$ confirming the patient did not convert.
- **Censoring**: The patient drops out of the study at $T+12$ without converting. We cannot safely call this a negative for a 24-month horizon. Their outcome is `CENSORED`.

### 3. AT_LEAST
The event happens at some point $> T$. We use $H$ solely to establish confidence in negative cases.
- **Positives**: Event observed at any point $> T$.
- **Negatives**: No event observed, and we have follow up of at least $H$.
- **Censoring**: No event observed, and follow up is $< H$.

## Leakage Prevention

The `TemporalLeakageChecker` acts as a rigid firewall enforcing the following guarantees:
- **Feature Time Travel**: No feature in $X$ can have a timestamp $> T$.
- **Label Time Travel**: If a patient converted to the target diagnosis $\leq T$, they are excluded from the snapshot since the event has already occurred.
- **Partition Contamination**: The train, validation, and test sets are strictly disjoint at the patient level. A patient cannot contribute a snapshot to the training set and a later snapshot to the test set.

## Limitations and Assumptions

- **LATEST Feature Policy**: Currently, `ClinicalSample` feature blocks are constructed using a flattened vector of the most recent measurement $\leq T$. True sequence modeling using full `TIME_SERIES` policies is an active research goal.
- **Date Granularity**: All timeline arithmetic is calculated using discrete days, approximating months as $30.4375$ days.
