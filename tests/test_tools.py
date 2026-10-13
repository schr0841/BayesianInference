import unittest
import numpy as np
from scipy.stats import linregress
from statistics_agent.tools import sample_data, ols, bayesian


class RegressionTests(unittest.TestCase):
    def test_original_data_and_prefix(self):
        np.random.seed(42)
        original_x = 2 * np.random.rand(100, 1)
        original_y = 4 + 3 * original_x + np.random.randn(100, 1)
        x, y = sample_data()
        np.testing.assert_array_equal(x, original_x.ravel())
        np.testing.assert_array_equal(y, original_y.ravel())
        np.testing.assert_array_equal(sample_data(5)[1], y[:5])

    def test_ols_against_scipy(self):
        x, y = sample_data()
        reference = linregress(x, y)
        result = ols()
        np.testing.assert_allclose(result['estimate'], [reference.intercept, reference.slope])
        np.testing.assert_allclose(result['coefficient_sd'], [reference.intercept_stderr, reference.stderr])
        self.assertEqual(result['residual_df'], 98)

    def test_bayesian_against_augmented_least_squares(self):
        x, y = sample_data()
        design = np.c_[np.ones(100), x]
        tau, sigma = 2., 1.5
        augmented = np.vstack([design / sigma, np.eye(2)/tau])
        target = np.r_[y/sigma, np.zeros(2)]
        reference = np.linalg.lstsq(augmented, target, rcond=None)[0]
        result = bayesian(prior_sd=tau, noise_sd=sigma)
        np.testing.assert_allclose(result['estimate'], reference)
        self.assertTrue(np.all(np.linalg.eigvalsh(result['covariance']) > 0))

    def test_prediction_adds_noise(self):
        result = bayesian()
        prediction = result['prediction']
        z = 1.959963984540054
        mean_half = np.diff(prediction['mean_response_interval'])[0]/2
        pred_half = np.diff(prediction['new_observation_interval'])[0]/2
        self.assertAlmostEqual((pred_half**2-mean_half**2)/z**2, 1.)
        self.assertTrue(bayesian(prediction_x=5)['prediction']['extrapolation'])

    def test_more_data_reduces_uncertainty(self):
        self.assertTrue(np.all(np.array(bayesian()['coefficient_sd']) < np.array(bayesian(n=5)['coefficient_sd'])))

    def test_invalid_inputs(self):
        for n in [0, 2, 101, True, 4.5]:
            with self.assertRaises(ValueError): sample_data(n)
        for kwargs in [{'prior_sd': 0}, {'noise_sd': float('nan')}, {'level': 1}, {'prediction_x': float('inf')}]:
            with self.assertRaises(ValueError): bayesian(**kwargs)


if __name__ == '__main__':
    unittest.main()
