# Hypothesis status

**Question.** Can automated, statistically grounded sensitivity analysis
identify deployment decisions that look favorable under ordinary offline
evaluation and become unsupported under stated changes to the evaluation
assumptions, the data distribution, or operational constraints?

**Status.** Unanswered. No reversal method is implemented. Experiment 001
checks IPS and SNIPS against a known synthetic policy value. A pass on that
experiment is a statement about those two estimators under the simulator's
assumptions. It is not evidence about deployment decisions.

**Constraint.** If a later reversal flag mostly restates a wide confidence
interval or a small effective sample size, the notes should say that. The
hypothesis does not get rewritten to match a positive table.
