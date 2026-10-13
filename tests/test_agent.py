"""Exercise the real SDK loop with a scripted model; no key or network needed."""
import json
import unittest
from agents import Agent, Model, ModelResponse, RunConfig, Runner, Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
from statistics_agent.agent import fit_ols, fit_bayesian_linear_regression


class ScriptedModel(Model):
    def __init__(self):
        self.calls = 0
        self.received_input = None

    async def get_response(self, *args, **kwargs):
        self.calls += 1
        self.received_input = kwargs.get('input', args[1] if len(args) > 1 else None)
        if self.calls == 1:
            output = [ResponseFunctionToolCall(
                id=f'fc_{i}', call_id=f'call_{i}', name=name,
                arguments=json.dumps({'n': 100, 'level': 0.95, 'prediction_x': 1.0,
                                      **extra}), type='function_call')
                for i, (name, extra) in enumerate([
                    ('fit_ols', {}),
                    ('fit_bayesian_linear_regression', {'prior_sd': 10., 'noise_sd': 1.})])]
        else:
            output = [ResponseOutputMessage(
                id='msg_1', role='assistant', status='completed', type='message',
                content=[ResponseOutputText(type='output_text', text='Scripted test complete.', annotations=[])])]
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError('Only non-streaming runs are tested.')
        yield


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_executes_tools_and_returns_outputs_to_model(self):
        model = ScriptedModel()
        agent = Agent(name='Test', model=model,
                      tools=[fit_ols, fit_bayesian_linear_regression])
        result = await Runner.run(agent, 'Compare regression fits.', max_turns=3,
                                  run_config=RunConfig(tracing_disabled=True))
        self.assertEqual(model.calls, 2)
        self.assertEqual(result.final_output, 'Scripted test complete.')
        outputs = [entry for entry in model.received_input
                   if entry.get('type') == 'function_call_output']
        self.assertEqual(len(outputs), 2)
        self.assertIn('2.770', str(outputs))
        self.assertEqual(sum(item.type == 'tool_call_item' for item in result.new_items), 2)


if __name__ == '__main__':
    unittest.main()
