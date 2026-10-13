"""API-free baseline: python -m examples.compare_regression."""
import json
from statistics_agent.tools import ols, bayesian

if __name__ == '__main__':
    print(json.dumps({'frequentist': ols(), 'bayesian': bayesian()}, indent=2))
