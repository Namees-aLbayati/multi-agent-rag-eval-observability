import json

from fastapi.testclient import TestClient
from langchain_core.tools import tool

from backend import workflow
from backend.main import app


def run(query='My landlord kept my security deposit', retry=True):
    events = []
    result = workflow.execute(query, 'demo', retry, events.append)
    return result, events


def test_retry_appends_evidence_and_cases_run_once(monkeypatch):
    monkeypatch.setattr(workflow.time, 'sleep', lambda _: None)
    result, events = run()
    assert result['attempts'] == 2
    assert result['validation_status'] == 'RELEVANT'
    assert len(result['sources']) == 3
    tools = [e['step'] for e in events if e.get('phase') == 'running']
    assert tools == ['parse_query', 'validate_and_refine_parsed_query', 'policy_search_tool', 'cases_search_tool', 'validate_chunks_tool', 'policy_search_tool', 'validate_chunks_tool', 'reply_tool']
    requests = [e for e in events if e['step'] == 'tool_request']
    assert requests[3]['executed_args']['attempt'] == 2
    assert requests[3]['executed_args']['seen_ids'] == ['demo-policy-1']


def test_happy_path_and_no_external_calls(monkeypatch):
    monkeypatch.setattr(workflow.time, 'sleep', lambda _: None)
    def forbidden(*args, **kwargs):
        raise AssertionError('Demo must never call a provider')
    monkeypatch.setattr(workflow, 'model', forbidden)
    monkeypatch.setattr(workflow.retrieval, 'search', forbidden)
    result, _ = run(retry=False)
    assert result['attempts'] == 1
    assert result['trace_url'] is None
    assert 'DEMO' in result['answer']


def test_out_of_scope_stops_before_response_agent(monkeypatch):
    monkeypatch.setattr(workflow.time, 'sleep', lambda _: None)
    result, events = run('How do I bake a cake?')
    assert result['answer'].startswith('OUT_OF_SCOPE')
    assert result['sources'] == []
    assert not any(e['step'] == 'response_agent' for e in events)


def test_third_failed_validation_terminates(monkeypatch):
    monkeypatch.setattr(workflow.time, 'sleep', lambda _: None)
    original = workflow.make_tools

    @tool
    def always_insufficient(query: str, policy_chunks: list[dict], attempt: int) -> dict:
        """Simulate insufficient evidence on every attempt."""
        return {'status': 'NOT_RELEVANT', 'explanation': 'Missing jurisdiction'}

    def patched(demo, retry):
        tools = original(demo, retry)
        tools['validate_chunks_tool'] = always_insufficient
        return tools
    monkeypatch.setattr(workflow, 'make_tools', patched)
    result, events = run()
    assert result['attempts'] == 3
    assert 'sufficient' in result['answer']
    assert not any(e['step'] == 'reply_tool' for e in events)
    searches = [e for e in events if e['step'] == 'policy_search_tool' and e.get('phase') == 'done']
    assert [e['output']['candidates_requested'] for e in searches] == [5, 10, 15]


def test_policy_search_filters_chunk_ids(monkeypatch):
    docs = [{'id': 'a', 'text': 'old', 'source': 'p.pdf', 'page': 1}, {'id': 'b', 'text': 'new chunk on same page', 'source': 'p.pdf', 'page': 1}]
    monkeypatch.setattr(workflow.retrieval, 'search', lambda *args: docs)
    tool = workflow.make_tools(False, False)['policy_search_tool']
    result = tool.invoke({'query': 'deposit', 'attempt': 2, 'seen_ids': ['a']})
    assert result['new_ids'] == ['b']
    result = tool.invoke({'query': 'deposit', 'attempt': 3, 'seen_ids': ['a', 'b']})
    assert result['status'] == 'NO_NEW_POLICIES'


def test_api_stream_demo(monkeypatch):
    monkeypatch.setattr(workflow.time, 'sleep', lambda _: None)
    with TestClient(app) as client:
        response = client.post('/api/run', json={'query': 'My landlord kept my deposit', 'mode': 'demo'})
        assert response.status_code == 200
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[-1]['type'] == 'result'
        assert any(e.get('step') == 'retry' for e in events)
        assert client.get('/').status_code == 200
        assert client.get('/assets/app.js').status_code == 200


def test_api_validation_and_credentials(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with TestClient(app) as client:
        assert client.post('/api/run', json={'query': '   '}).status_code == 422
        assert client.post('/api/run', json={'query': 'test', 'mode': 'invalid'}).status_code == 422
        assert client.post('/api/run', json={'query': 'My rent', 'mode': 'live'}).status_code == 400
        assert client.post('/api/index').status_code == 400
        status = client.get('/api/status').json()
        assert not status['openai_configured']
        assert 'api_key' not in json.dumps(status)


def test_stream_errors_are_safe(monkeypatch):
    import backend.main as main
    def broken(*args, **kwargs):
        raise RuntimeError('secret-provider-detail')
    monkeypatch.setattr(main, 'execute', broken)
    with TestClient(app) as client:
        response = client.post('/api/run', json={'query': 'My rent'})
        assert 'secret-provider-detail' not in response.text
        assert json.loads(response.text.splitlines()[-1])['type'] == 'error'


def test_live_dispatch_uses_state_not_model_arguments(monkeypatch):
    """Exercise the live branch with fake model responses; no provider calls."""
    from langchain_core.messages import AIMessage
    monkeypatch.setattr(workflow, 'tracing_enabled', lambda: False)
    bound_tools = []

    class FakeModel:
        def with_structured_output(self, schema):
            class Structured:
                def invoke(self, messages):
                    if schema is workflow.ScopeCheck:
                        return workflow.ScopeCheck(in_scope=True, explanation='In scope', refined_output='Security deposit question')
                    return workflow.EvidenceCheck(sufficient=True, explanation='Policy covers this question')
            return Structured()

        def bind_tools(self, tools, tool_choice, parallel_tool_calls):
            assert parallel_tool_calls is False
            assert tools[0].name == tool_choice
            bound_tools.append(tool_choice)
            class Bound:
                def invoke(self, messages):
                    return AIMessage(content='', tool_calls=[{'name': tool_choice, 'args': {'attempt': 99, 'query': 'ignored model argument'}, 'id': 'call_' + str(len(bound_tools))}])
            return Bound()

        def invoke(self, messages):
            return AIMessage(content='Grounded answer [policy.pdf, p. 1]')

    searched = []
    def fake_search(kind, query, k):
        searched.append((kind, query, k))
        return [{'id': kind + '-1', 'text': 'Evidence', 'source': kind + '.pdf', 'page': 1, 'distance': 0.1}]
    monkeypatch.setattr(workflow, 'model', lambda **kwargs: FakeModel())
    monkeypatch.setattr(workflow.retrieval, 'search', fake_search)
    events = []
    result = workflow.execute('My landlord kept my deposit', 'live', False, events.append)
    assert bound_tools == ['policy_search_tool', 'cases_search_tool', 'validate_chunks_tool', 'reply_tool']
    assert searched[0][2] == 5
    assert searched[1][2] == 4
    assert 'ignored model argument' not in searched[0][1]
    assert result['answer'] == 'Grounded answer [policy.pdf, p. 1]'
    validation = next(e for e in events if e['step'] == 'validate_chunks_tool' and e.get('phase') == 'running')
    assert all(c['source'] == 'policy.pdf' for c in validation['inputs']['policy_chunks'])
