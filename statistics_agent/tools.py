"""Statistical calculations, independent of the language model and API."""
import numpy as np
from scipy.stats import norm, t


def sample_data(n: int = 100):
    if isinstance(n, bool) or not isinstance(n, int) or not 3 <= n <= 100:
        raise ValueError('n must be an integer from 3 to 100.')
    # Same legacy generator and full draw order as GradientDescentBasic.ipynb.
    random = np.random.RandomState(42)
    x = 2 * random.rand(100, 1)
    y = 4 + 3 * x + random.randn(100, 1)
    return x[:n, 0], y[:n, 0]


def _validate(level, prediction_x):
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError('level must be strictly between 0 and 1.')
    if not np.isfinite(prediction_x) or abs(prediction_x) > 1e6:
        raise ValueError('prediction_x must be finite and between -1e6 and 1e6.')


def describe_data(n: int = 100) -> dict:
    x, y = sample_data(n)
    return {'n': n, 'source': 'Seed-42 data from GradientDescentBasic.ipynb',
            'generating_intercept': 4, 'generating_slope': 3, 'generating_noise_sd': 1,
            'x_range': [float(x.min()), float(x.max())],
            'x': x.tolist(), 'y': y.tolist()}


def _summary(method, n, estimate, covariance, critical, level, prediction_x,
             noise_variance):
    se = np.sqrt(np.diag(covariance))
    phi = np.array([1., prediction_x])
    mean = float(phi @ estimate)
    mean_sd = float(np.sqrt(phi @ covariance @ phi))
    pred_sd = float(np.sqrt(mean_sd**2 + noise_variance))
    x, _ = sample_data(n)
    return {
        'method': method, 'n': n, 'level': level,
        'parameter_order': ['intercept', 'slope'], 'estimate': estimate.tolist(),
        'coefficient_sd': se.tolist(), 'covariance': covariance.tolist(),
        'coefficient_intervals': np.column_stack(
            [estimate - critical * se, estimate + critical * se]).tolist(),
        'prediction': {
            'x': prediction_x, 'mean': mean,
            'mean_response_interval': [mean-critical*mean_sd, mean+critical*mean_sd],
            'new_observation_interval': [mean-critical*pred_sd, mean+critical*pred_sd],
            'extrapolation': bool(prediction_x < x.min() or prediction_x > x.max())},
        'interval_scope': 'Pointwise; not a simultaneous band',
    }


def ols(n: int = 100, level: float = 0.95, prediction_x: float = 1.0) -> dict:
    _validate(level, prediction_x)
    x, y = sample_data(n)
    design = np.column_stack([np.ones(n), x])
    estimate = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ estimate
    df = n - 2
    noise_variance = float(residual @ residual / df)
    covariance = noise_variance * np.linalg.solve(design.T @ design, np.eye(2))
    result = _summary('OLS with estimated noise variance', n, estimate, covariance,
                      float(t.ppf((1+level)/2, df)), level, prediction_x, noise_variance)
    result.update({'noise_variance': noise_variance, 'residual_df': df,
                   'interval_type': 'Frequentist confidence / prediction intervals',
                   'assumptions': 'Linear mean; independent homoscedastic Gaussian errors'})
    return result


def bayesian(n: int = 100, prior_sd: float = 10.0, noise_sd: float = 1.0,
             level: float = 0.95, prediction_x: float = 1.0) -> dict:
    _validate(level, prediction_x)
    for name, value in [('prior_sd', prior_sd), ('noise_sd', noise_sd)]:
        if not np.isfinite(value) or not 1e-6 <= value <= 1e6:
            raise ValueError(f'{name} must be finite and between 1e-6 and 1e6.')
    x, y = sample_data(n)
    design = np.column_stack([np.ones(n), x])
    precision = np.eye(2)/prior_sd**2 + design.T @ design/noise_sd**2
    covariance = np.linalg.solve(precision, np.eye(2))
    estimate = np.linalg.solve(precision, design.T @ y/noise_sd**2)
    result = _summary('Gaussian Bayesian regression with known noise variance', n,
                      estimate, covariance, float(norm.ppf((1+level)/2)), level,
                      prediction_x, noise_sd**2)
    result.update({'prior_mean': [0., 0.], 'prior_sd': prior_sd,
                   'known_noise_sd': noise_sd,
                   'interval_type': 'Posterior credible / posterior predictive intervals',
                   'assumptions': 'Linear mean; independent Gaussian errors; fixed known noise SD; Gaussian coefficient prior'})
    return result
