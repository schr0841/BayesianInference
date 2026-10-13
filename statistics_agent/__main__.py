import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from .tools import describe_data, ols, bayesian


def main():
    parser = argparse.ArgumentParser(description='Inspect a basic statistics agent.')
    parser.add_argument('question', nargs='?', default='Compare Bayesian and frequentist slope estimates on our sample data, and explain the intervals.')
    parser.add_argument('--offline', action='store_true', help='Run both tools directly; no model or API call.')
    parser.add_argument('--show-tools', action='store_true', help='Print the actual model tool calls and outputs.')
    parser.add_argument('--save-run', action='store_true', help='Save local question, tool events, and answer in agent_runs/.')
    args = parser.parse_args()
    if args.offline:
        print('OFFLINE DEMO: direct calculations, no agent decisions or API calls.')
        print(json.dumps({'source': describe_data()['source'], 'ols': ols(), 'bayesian': bayesian()}, indent=2))
        return
    from .agent import ask
    from agents.exceptions import AgentsException
    from openai import APIError
    try:
        result = asyncio.run(ask(args.question))
    except (ValueError, APIError, AgentsException) as error:
        # Do not dump HTTP headers, credentials, or the full API exception.
        print(f'Agent could not run ({type(error).__name__}). Check your .env, model access, billing, and network.')
        raise SystemExit(1)
    events = []
    for item in result.new_items:
        if item.type == 'tool_call_item':
            raw = item.raw_item
            events.append({'type': 'tool_call', 'name': raw.name, 'arguments': raw.arguments})
        elif item.type == 'tool_call_output_item':
            events.append({'type': 'tool_output', 'output': item.output})
    if args.show_tools:
        print(json.dumps(events, indent=2, default=str))
    print(result.final_output)
    if args.save_run:
        folder = Path(__file__).resolve().parents[1] / 'agent_runs'
        folder.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output = folder / f'{stamp}.json'
        output.write_text(json.dumps({'question': args.question, 'events': events,
                                     'answer': result.final_output}, indent=2, default=str), encoding='utf-8')
        print(f'Saved local run: {output}')


if __name__ == '__main__':
    main()
