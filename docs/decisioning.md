# Decision Logic

The model outputs a probability of default (PD).

Business rules convert PD into decisions:

- PD < low_threshold → APPROVE
- low_threshold ≤ PD < high_threshold → MANUAL REVIEW
- PD ≥ high_threshold → REJECT

This allows:
- Risk control
- Human review for borderline cases
- Automation for safe cases

Thresholds can be adjusted without retraining the model.
