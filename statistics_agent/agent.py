"""Model configuration and the SDK-managed tool-calling loop."""
import os
from pathlib import Path
from agents import Agent, ModelSettings, RunConfig, Runner, function_tool
from dotenv import load_dotenv
from .tools import describe_data, ols, bayesian

ROOT = Path(__file__).resolve().parents[1]


@function_tool
def get_sample_data(n: int = 100) -> dict:
    """Inspect the first n observations of the original 100-point seed-42 data."""
    return describe_data(n)


@function_tool
def fit_ols(n: int = 100, level: float = 0.95, prediction_x: float = 1.0) -> dict:
    """Fit frequentist regression; estimate noise and use Student-t intervals."""
    return ols(n, level, prediction_x)


@function_tool
def fit_bayesian_linear_regression(n: int = 100, prior_sd: float = 10.0,
                                   noise_sd: float = 1.0, level: float = 0.95,
                                   prediction_x: float = 1.0) -> dict:
    """Fit exact Gaussian posterior with zero-centered prior and known noise SD."""
    return bayesian(n, prior_sd, noise_sd, level, prediction_x)


def build_agent():
    load_dotenv(ROOT / '.env')
    if not os.getenv('OPENAI_API_KEY', '').strip():
        raise ValueError('Set OPENAI_API_KEY in your local .env file. See README.md.')
    return Agent(
        name='Statistics tutor',
        model=os.getenv('OPENAI_MODEL') or 'gpt-4.1-mini',
        instructions=(Path(__file__).with_name('instructions.md')).read_text(encoding='utf-8'),
        tools=[get_sample_data, fit_ols, fit_bayesian_linear_regression],
        model_settings=ModelSettings(max_tokens=1800),
    )


async def ask(question: str):
    if not question.strip() or len(question) > 6000:
        raise ValueError('Enter a nonempty question of at most 6000 characters.')
    # No hosted trace export; CLI logs below are local and ignored by Git.
    return await Runner.run(build_agent(), question, max_turns=6,
                            run_config=RunConfig(tracing_disabled=True))
