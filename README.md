# Statistics agent experiment

A basic Bayesian and frequentist statistics tutor using the OpenAI Agents SDK.
The SDK runs the loop: model chooses a tool, Python executes it, the result goes
back to the model, and the model answers or chooses another tool. The only
numerical scope currently implemented is simple linear regression on the exact
seed-42 dataset from `GradientDescentBasic.ipynb`.

## Files

```text
BayesianInference/
  BayesianLinearRegression.ipynb
  statistics_agent/
    instructions.md    # Behavior, assumptions, and interpretation rules
    agent.py           # Agent definition and three tool adapters
    tools.py           # Deterministic statistical calculations
    __main__.py        # Command-line interface and optional local event log
  examples/
    compare_regression.py
    prompts.md
  tests/
    test_agent.py      # Real SDK loop with a scripted, API-free model
    test_tools.py
```

## Windows setup (Python 3.10 or later)

Open a terminal in this directory. Commands do not require PowerShell activation:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m statistics_agent --offline
```

The offline demo runs the calculations directly. It does not simulate a language
model or make agent decisions, and costs no API credits.

The tests check the numerical tools against independent calculations and run the
real SDK loop with a scripted model that requests both fitting tools. That verifies
tool execution and returning outputs to the next model turn; it does not evaluate
the accuracy of a real model's explanations. A live API run still needs your key.

## Enable the actual agent

Create a separate project on the [OpenAI API platform](https://platform.openai.com/),
check its billing/credits and limits, and create a project API key. API access uses
your own key, including when free test access or promotional credits are available.
Do not assume that a ChatGPT subscription or a free account funds deployed API usage.
Account eligibility and available credit must be checked in your dashboard.

```powershell
Copy-Item .env.example .env
```

Edit `.env` locally and set `OPENAI_API_KEY`. Never paste the key into a chat or
commit it. `OPENAI_MODEL` defaults to `gpt-4.1-mini`, a small model supporting tool
calling; change it to a model available to your project if needed. This is a
learning prototype, not a demonstrated statistical expert.

```powershell
.\.venv\Scripts\python.exe -m statistics_agent "Compare Bayesian and frequentist slope estimates and explain the intervals." --show-tools --save-run
```

`--show-tools` displays actual tool-call arguments and outputs. `--save-run`
writes the question, events, and answer to a Git-ignored `agent_runs/` folder.
These events show actions, not hidden model reasoning. Hosted SDK tracing is
disabled. Prompts and tool outputs are still sent to OpenAI for the agent run.
Each command starts a fresh conversation. There is no persistent model memory.

There are at most six SDK turns per invocation and 1,800 output tokens per model
call. Multiple model calls can cost more than a single response; these controls
are not a dollar cap. Set platform spending controls and monitor usage.

## Statistical details

- OLS estimates noise variance using n-2 degrees of freedom and Student-t intervals.
- Bayesian fitting uses a zero-centered independent Gaussian prior for both
  coefficients, known noise SD, and the exact Gaussian posterior.
- Both provide pointwise coefficient intervals, mean-response intervals, and
  new-observation intervals at the requested x.
- Comparisons must disclose the different treatment of noise variance.
- Tools have no arbitrary file access, code execution, web retrieval, uploads,
  residual-diagnostic suite, or MCMC support. Fixed-data tools return JSON numbers.

## Eventually connecting GitHub Pages

GitHub Pages serves static HTML/CSS/JavaScript; it cannot run this Python agent.
The future deployment architecture is:

```text
GitHub Pages interface -> authenticated HTTPS backend -> agent/tools -> OpenAI
```

Keep this Python package on a separate backend host with the API key in its secret
manager. The Pages interface sends a question to that backend and receives the
answer and selected tool events. Never put the key in browser JavaScript, an HTML
file, a Pages build artifact, or a public repository. A GitHub Actions secret is
not safe if a build embeds it into files delivered to browsers.

Before public use, add real authentication, request-size limits, per-user rate
limits, timeouts, concurrency limits, and a server-enforced usage budget. CORS is
not authentication. Start with access restricted to you. No website or backend
has been deployed by this experiment.

## Documentation

- [Agents SDK quickstart](https://developers.openai.com/api/docs/guides/agents/quickstart)
- [OpenAI API setup](https://developers.openai.com/api/docs/quickstart)
- [API key and production practices](https://developers.openai.com/api/docs/guides/production-best-practices)
- [API pricing](https://developers.openai.com/api/docs/pricing)
- [GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)
