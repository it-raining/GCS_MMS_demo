"""
shooting.py - Backward-compatible access to shooting-related factories.

The multiple-shooting transcription now lives in `constraint_layers.py`, and
the CasADi RK4/local-cost factory lives in `dynamics.py` so both optimizers use
one integration bundle.
"""

from dynamics import create_casadi_local_cost

__all__ = ["create_casadi_local_cost"]
