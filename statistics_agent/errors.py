"""Safe, actionable API diagnostics without dumping responses or credentials."""
from openai import RateLimitError


def explain_error(error):
    if isinstance(error, ValueError):
        return str(error)
    if isinstance(error, RateLimitError):
        body = error.body if isinstance(error.body, dict) else {}
        detail = body.get('error', body)
        if not isinstance(detail, dict):
            detail = {}
        code = detail.get('code')
        kind = detail.get('type')
        quota_codes = {
            'insufficient_quota', 'billing_hard_limit_reached',
            'billing_not_active', 'usage_limit_reached',
            'organization_usage_limit_exceeded', 'project_usage_limit_exceeded',
            'organization_spend_limit_exceeded', 'project_spend_limit_exceeded',
        }
        if code in quota_codes or kind == 'insufficient_quota':
            return ('OpenAI rejected the request because of API billing, credit, or quota limits. '
                    'Check billing and limits for the project that owns this key at '
                    'https://platform.openai.com/settings/organization/billing/overview . '
                    'Retrying alone will not restore quota.')
        if code == 'rate_limit_exceeded' or kind in {'requests', 'tokens', 'rate_limit_exceeded'}:
            return ('OpenAI request/token rate limit reached. Wait before retrying and check '
                    'your project/model limits at https://platform.openai.com/settings/organization/limits .')
        return ('OpenAI returned HTTP 429. This can mean a rate limit or a billing/quota limit. '
                'Check API billing, available credits, and project/model limits; '
                'the response did not identify a recognized category.')
    return (f'Agent could not run ({type(error).__name__}). '
            'Check your .env, model access, billing, and network.')
