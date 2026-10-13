import unittest
import httpx2 as httpx
from openai import RateLimitError
from statistics_agent.errors import explain_error


class ErrorTests(unittest.TestCase):
    def error(self, body):
        response = httpx.Response(429, request=httpx.Request('POST', 'https://api.openai.com/v1/responses'))
        return RateLimitError('Secret response text must not be printed', response=response, body=body)

    def test_quota_is_not_treated_as_transient_rate_limit(self):
        message = explain_error(self.error({'code': 'insufficient_quota'}))
        self.assertIn('billing', message)
        self.assertIn('Retrying alone', message)
        self.assertNotIn('Secret', message)

    def test_nested_rate_limit(self):
        message = explain_error(self.error({'error': {'code': 'rate_limit_exceeded'}}))
        self.assertIn('Wait before retrying', message)

    def test_unknown_429_remains_uncertain(self):
        self.assertIn('can mean', explain_error(self.error(None)))
