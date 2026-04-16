"""Model-risk backtesting framework.

Validates the accuracy of internal VaR / simulation models by running them
walk-forward against realised hypothetical P&L in EUR and applying the
Basel traffic-light test together with Kupiec and Christoffersen coverage
tests. See changes.txt for the design rationale.
"""
