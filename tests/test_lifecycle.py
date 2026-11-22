"""Iteration 6: durable generations, explicit context fences and bounded concurrency."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest
from fastapi.testclient import TestClient

from app.contracts.v2.lifecycle import Activate, CycleRequest, Decision, InstallationUpdate
from app.main import create_app
from app.preparation.models import PrepareCandidate
from app.preparation.service import PreparationService
from app.preparation.storage import PreparationError
from tests.test_preparation import AUTH, setup, store_and_prepare
from tests.test_execution import bound_setup, onnx_identity
from tests.test_safety import upgrade


@pytest.fixture
def managed(bound_setup):
    upgrade(bound_setup[1])
    return bound_setup


def declare(client, config, key="context", context_id=None, activation_id=None):
    response = client.put('/admin/v1/installation', headers={**AUTH, "Idempotency-Key": key}, json=dict(
        schema_version="installation-update.v1-draft", expected_context_id=context_id,
        expected_activation_id=activation_id, installation=config["installation"]))
    assert response.status_code == 200, response.text
    return response.json()["context_id"]


def activate(client, candidate, context_id, previous=None, key="activate"):
    body = dict(schema_version="activation.v2-draft", candidate_id=candidate["candidate_id"],
                expected_context_id=context_id, expected_activation_id=previous)
    response = client.post('/admin/v1/activations', headers={**AUTH, "Idempotency-Key": key}, json=body)
    assert response.status_code == 202, response.text
    operation = response.json()["operation_id"]
    return client.get('/admin/v1/operations/' + operation, headers=AUTH).json()


def cycle(client, activation_id, context_id, load=5):
    return client.post('/inference/v2', headers=AUTH, json=dict(expected_activation_id=activation_id,
        expected_context_id=context_id, cycle_id="cycle-10", features={"observations": {"non_shiftable_load": load}}))


def ready(client, config, request):
    candidate = store_and_prepare(client, config, request)
    context = declare(client, config)
    receipt = activate(client, candidate, context)
    assert receipt["status"] == "succeeded", receipt
    return candidate, context, receipt["resource_id"]


def new_candidate(client, fixture, revision="configuration-2", value=7):
    _, config, request, _, bundle, _ = fixture
    changed = copy.deepcopy(config)
    changed["revision"] = revision
    (bundle / 'policy.json').write_text(json.dumps({"default_actions": {"dispatch": value}, "rules": []}))
    candidate = store_and_prepare(client, changed, {**request, "configuration_revision": revision}, key=revision)
    assert candidate["state"] == "ready", candidate
    return candidate, changed


def test_activation_serves_audited_decisions_and_reuses_loaded_runtime(managed, monkeypatch):
    settings, config, request, *_ = managed
    with TestClient(create_app(config=settings)) as client:
        assert client.get('/ready').status_code == 503
        candidate, context, activation_id = ready(client, config, request)
        def forbidden(*args, **kwargs): raise AssertionError("Inference must reuse the loaded runtime")
        monkeypatch.setattr(client.app.state.preparation, 'build_execution', forbidden)
        result = cycle(client, activation_id, context, 10)
        assert result.status_code == 200, result.text
        doc = result.json()
        Decision.parse_obj(doc)
        assert doc['schema_version'] == 'decision.v2-draft' and 'evaluation_only' not in doc
        assert doc['proposed_actions'][0]['proposed_value'] == 2
        assert doc['actions'][0]['value'] == 1
        assert doc['activation']['activation_id'] == activation_id and doc['activation']['context_id'] == context
        assert doc['context']['candidate_id'] == candidate['candidate_id'] and doc['cycle_id'] == 'cycle-10'
        assert client.get('/ready').status_code == 200
        assert client.get('/health').json()['configured']
        assert client.post('/inference', json={'features': {}}).status_code == 409


def test_configuration_only_swap_and_replay_do_not_reactivate_old_generation(managed):
    settings, config, request, *_ = managed
    with TestClient(create_app(config=settings)) as client:
        first, context, old = ready(client, config, request)
        second, changed = new_candidate(client, managed)
        receipt = activate(client, second, context, old, 'second')
        assert receipt['status'] == 'succeeded', receipt
        new = receipt['resource_id']
        assert cycle(client, new, context).json()['actions'][0]['value'] == 6
        assert cycle(client, old, context).status_code == 409
        replay = activate(client, first, context)
        assert replay['resource_id'] == old
        assert client.get('/ready').json()['activation_id'] == new
        assert client.get('/admin/v1/activations/'+old, headers=AUTH).json()['state'] == 'superseded'
        assert client.get('/admin/v1/activations/'+new, headers=AUTH).json()['usable']


def test_restart_recovers_committed_generation_from_its_own_snapshot(managed):
    settings, config, request, _, bundle, _ = managed
    with TestClient(create_app(config=settings)) as client:
        candidate, context, activation_id = ready(client, config, request)
        before = cycle(client, activation_id, context).json()
    # Recovery must not trust mutable delivery or a subsequently damaged candidate.
    (bundle/'policy.json').write_text('{}')
    path = settings.configuration_store_dir / 'candidates' / candidate['candidate_id'] / 'files/policy.json'
    path.chmod(0o600); path.write_text('{}')
    with TestClient(create_app(config=settings)) as client:
        result = cycle(client, activation_id, context)
        assert result.status_code == 200, result.text
        after = result.json()
        assert after['actions'] == before['actions'] and after['activation'] == before['activation']
        assert client.get('/admin/v1/operations/'+activation_id, headers=AUTH).json()['status'] == 'succeeded'


def test_corrupt_committed_files_report_unavailable_without_old_fallback(managed):
    settings, config, request, *_ = managed
    with TestClient(create_app(config=settings)) as client:
        _, context, old = ready(client, config, request)
        second, _ = new_candidate(client, managed)
        new = activate(client, second, context, old, 'second')['resource_id']
    path = settings.configuration_store_dir / 'generations' / new / 'files/policy.json'
    path.chmod(0o600); path.write_text('{}')
    with TestClient(create_app(config=settings)) as client:
        assert client.get('/health').status_code == 200
        status = client.get('/ready')
        assert status.status_code == 503 and status.json()['state'] == 'recovery_failed'
        assert status.json()['activation_id'] == new
        assert cycle(client, new, context).status_code == 503
        assert client.get('/admin/v1/operations/'+new, headers=AUTH).json()['status'] == 'succeeded'


def test_new_context_suspends_old_and_disallows_unsafe_rollback(managed):
    settings, config, request, *_ = managed
    with TestClient(create_app(config=settings)) as client:
        candidate, old_context, old = ready(client, config, request)
        changed = copy.deepcopy(config)
        changed['revision'] = 'new-limit'
        changed['installation']['revision'] = 'installation-2'
        changed['installation']['channels'][-1]['physical_bounds']['upper'] = 4
        new_context = declare(client, changed, 'new-context', old_context, old)
        assert client.get('/ready').status_code == 503
        assert cycle(client, old, old_context).status_code == 409
        assert cycle(client, old, new_context).status_code == 503
        failed = activate(client, candidate, new_context, old, 'unsafe-rollback')
        assert failed['status'] == 'failed' and failed['error']['code'] == 'installation_changed'
        # Reusing the old bundle is allowed only after preparation against current context.
        updated = store_and_prepare(client, changed, {**request, 'configuration_revision': 'new-limit'}, key='new-limit')
        receipt = activate(client, updated, new_context, old, 'reviewed')
        assert receipt['status'] == 'succeeded', receipt
        assert cycle(client, receipt['resource_id'], new_context).status_code == 200
    with TestClient(create_app(config=settings)) as client:
        assert client.get('/ready').status_code == 200


def test_restart_during_context_transition_does_not_resurrect_previous_context(managed):
    settings, config, request, *_ = managed
    with TestClient(create_app(config=settings)) as client:
        _, context, old = ready(client, config, request)
        changed = copy.deepcopy(config)
        changed['installation']['revision'] = 'house-changed'
        new_context = declare(client, changed, 'changed', context, old)
    with TestClient(create_app(config=settings)) as client:
        assert client.get('/ready').json()['state'] == 'context_changed'
        assert cycle(client, old, new_context).status_code == 503


@pytest.mark.parametrize('failure', ['hash', 'runtime'])
def test_failed_activation_preserves_previous_generation(managed, monkeypatch, failure):
    settings, config, request, *_ = managed
    with TestClient(create_app(config=settings)) as client:
        _, context, old = ready(client, config, request)
        second, _ = new_candidate(client, managed)
        if failure == 'hash':
            path = settings.configuration_store_dir/'candidates'/second['candidate_id']/'files/policy.json'
            path.chmod(0o600); path.write_text('{}')
        else:
            def fail(*args, **kwargs): raise RuntimeError('native detail')
            monkeypatch.setattr(client.app.state.preparation, 'build_execution', fail)
        receipt = activate(client, second, context, old, 'failure')
        assert receipt['status'] == 'failed'
        assert client.get('/ready').json()['activation_id'] == old
        assert cycle(client, old, context).status_code == 200


def test_required_preconditions_authentication_and_idempotency_conflicts(managed):
    settings, config, request, *_ = managed
    with TestClient(create_app(config=settings)) as client:
        candidate, context, old = ready(client, config, request)
        body = dict(schema_version='activation.v2-draft', candidate_id=candidate['candidate_id'], expected_context_id=context, expected_activation_id=old)
        assert client.post('/admin/v1/activations', json=body).status_code == 401
        assert client.post('/inference/v2', json=dict(features={}, cycle_id='c', expected_context_id=context, expected_activation_id=old)).status_code == 401
        for key in ('expected_context_id', 'expected_activation_id'):
            bad = dict(body); del bad[key]
            assert client.post('/admin/v1/activations', json=bad, headers={**AUTH,'Idempotency-Key':'bad'}).status_code == 422
        assert client.post('/admin/v1/activations', json=body, headers={**AUTH,'Idempotency-Key':'activate'}).status_code == 409
        body['expected_activation_id'] = None
        response = client.post('/admin/v1/activations', json=body, headers={**AUTH,'Idempotency-Key':'stale'})
        assert response.status_code == 409 and response.json()['detail']['code'] == 'precondition_failed'


def test_context_declaration_replay_is_not_a_new_physical_change(managed):
    settings, config, request, *_ = managed
    with TestClient(create_app(config=settings)) as client:
        _, context, activation_id = ready(client, config, request)
        assert declare(client, config) == context  # Replay includes original null fences.
        assert declare(client, config, 'same-body', context, activation_id) == context
        assert client.get('/ready').status_code == 200


def test_context_change_while_loading_prevents_publication(managed, monkeypatch):
    settings, config, request, *_ = managed
    entered, release = threading.Event(), threading.Event()
    with TestClient(create_app(config=settings)) as client:
        _, context, old = ready(client, config, request)
        second, _ = new_candidate(client, managed)
        service = client.app.state.preparation
        build = service.build_execution
        def paused(*args, **kwargs):
            result = build(*args, **kwargs); entered.set(); assert release.wait(10); return result
        monkeypatch.setattr(service, 'build_execution', paused)
        with ThreadPoolExecutor() as pool:
            future = pool.submit(activate, client, second, context, old, 'race')
            assert entered.wait(10)
            changed = copy.deepcopy(config); changed['installation']['revision'] = 'new-composition'
            new_context = declare(client, changed, 'race-context', context, old)
            release.set(); receipt = future.result(10)
        assert receipt['status'] == 'failed' and receipt['error']['code'] == 'precondition_failed'
        assert service.lifecycle.status()['error'] is None
        assert service.lifecycle.status()['activation_id'] == old
        assert cycle(client, old, new_context).status_code == 503


def test_inflight_cycle_keeps_generation_and_retirement_is_bounded(managed, monkeypatch):
    settings, config, request, *_ = managed
    entered, release = threading.Event(), threading.Event()
    with TestClient(create_app(config=settings)) as client:
        _, context, old = ready(client, config, request)
        second, _ = new_candidate(client, managed)
        service = client.app.state.preparation
        engine = service.lifecycle.active.execution
        evaluate = engine.evaluate
        def paused(features): entered.set(); assert release.wait(10); return evaluate(features)
        monkeypatch.setattr(engine, 'evaluate', paused)
        with ThreadPoolExecutor() as pool:
            future = pool.submit(cycle, client, old, context)
            assert entered.wait(10)
            new = activate(client, second, context, old, 'second')['resource_id']
            assert new and cycle(client, new, context).status_code == 200
            blocked = activate(client, second, context, new, 'third')
            assert blocked['status'] == 'failed' and blocked['error']['code'] == 'generation_busy'
            assert service.lifecycle.status()['retired_requests'] == 1
            release.set(); result = future.result(10).json()
        assert result['activation']['activation_id'] == old and result['actions'][0]['value'] == 2
        assert service.lifecycle.status()['retired_requests'] == 0
        assert not (settings.configuration_store_dir/'generations'/old).exists()


def test_bounded_inference_admission(managed, monkeypatch):
    settings, config, request, *_ = managed
    settings.max_concurrent_inferences = 1
    entered, release = threading.Event(), threading.Event()
    with TestClient(create_app(config=settings)) as client:
        _, context, activation_id = ready(client, config, request)
        engine = client.app.state.preparation.lifecycle.active.execution
        evaluate = engine.evaluate
        def paused(features): entered.set(); assert release.wait(10); return evaluate(features)
        monkeypatch.setattr(engine, 'evaluate', paused)
        with ThreadPoolExecutor() as pool:
            pending = pool.submit(cycle, client, activation_id, context)
            assert entered.wait(10)
            assert cycle(client, activation_id, context).status_code == 429
            release.set(); assert pending.result(10).status_code == 200


def test_preparation_and_activation_admission_are_bounded_and_idempotent(managed):
    settings, config, request, *_ = managed
    settings.max_pending_preparations = 1
    with TestClient(create_app(config=settings)) as client:
        candidate, context, old = ready(client, config, request)
        service = client.app.state.preparation
        parsed = PrepareCandidate.parse_obj(request)
        operation = service.submit('pending', parsed)
        assert service.submit('pending', parsed) == operation
        with pytest.raises(PreparationError, match='admission'): service.submit('overflow', parsed)
        service.prepare(operation)
        body = Activate(schema_version='activation.v2-draft', candidate_id=candidate['candidate_id'], expected_context_id=context, expected_activation_id=old)
        op = service.lifecycle.submit('pending', body)
        assert service.lifecycle.submit('pending', body) == op
        with pytest.raises(PreparationError, match='already admitted'): service.lifecycle.submit('overflow', body)
        service.lifecycle.activate(op)
        assert service.receipt(op)['status'] == 'succeeded'


def test_native_build_gate_does_not_block_existing_inference(managed):
    settings, config, request, *_ = managed
    with TestClient(create_app(config=settings)) as client:
        candidate, context, activation_id = ready(client, config, request)
        service = client.app.state.preparation
        with service.work():
            result = client.post(f'/admin/v1/candidates/{candidate["candidate_id"]}/evaluations', json={'features':{}}, headers=AUTH)
            assert result.status_code == 429
            assert cycle(client, activation_id, context).status_code == 200


@pytest.mark.parametrize('kind', ['storage', 'history'])
def test_explicit_storage_and_candidate_retention_limits(managed, kind):
    settings, config, request, *_ = managed
    with TestClient(create_app(config=settings)) as client:
        ready(client, config, request)
        if kind == 'storage': settings.max_snapshot_storage_bytes = 1
        else: settings.max_stored_candidates = 1
        response = client.post('/admin/v1/candidates', json=request, headers={**AUTH,'Idempotency-Key':'full'})
        if kind == 'history': assert response.status_code == 507
        else:
            receipt = client.get('/admin/v1/operations/'+response.json()['operation_id'], headers=AUTH).json()
            assert receipt['status'] == 'failed' and receipt['error']['code'] == 'snapshot_storage_full'
        assert client.get('/ready').status_code == 200


@pytest.mark.parametrize('point', ['before_commit', 'after_commit'])
def test_real_process_exit_recovers_exact_journal_boundary(managed, tmp_path, point):
    settings, config, request, *_ = managed
    with TestClient(create_app(config=settings)) as client:
        _, context, old = ready(client, config, request)
        second, _ = new_candidate(client, managed)
    args = tmp_path/'crash.json'
    args.write_text(json.dumps(dict(root=str(settings.allowed_bundle_root), state=str(settings.configuration_store_dir),
        token=settings.admin_api_token.get_secret_value(), candidate=second['candidate_id'], context=context, old=old, point=point)))
    marker = tmp_path/'operation.txt'
    code = '''
import json,os,sys
from pathlib import Path
from app.settings import Settings
from app.preparation.service import PreparationService
from app.contracts.v2.lifecycle import Activate
v=json.loads(Path(sys.argv[1]).read_text())
s=PreparationService(Settings(_env_file=None,manifest_path=None,allowed_bundle_root=Path(v['root']),configuration_store_dir=Path(v['state']),installation_site_id='example-house',admin_api_token=v['token']))
l=s.lifecycle
request=Activate(schema_version='activation.v2-draft',candidate_id=v['candidate'],expected_context_id=v['context'],expected_activation_id=v['old'])
op=l.submit('crash',request)
Path(sys.argv[2]).write_text(op)
commit=l.store.commit
def crash(*args):
    if v['point']=='before_commit': os._exit(17)
    commit(*args)
    os._exit(17)
l.store.commit=crash
l.activate(op)
'''
    result = subprocess.run([sys.executable, '-c', code, str(args), str(marker)], capture_output=True, timeout=30)
    assert result.returncode == 17, result.stderr.decode()
    operation = marker.read_text()
    with TestClient(create_app(config=settings)) as client:
        receipt = client.get('/admin/v1/operations/'+operation, headers=AUTH).json()
        expected = old if point == 'before_commit' else operation
        assert receipt['status'] == ('uncertain' if point == 'before_commit' else 'succeeded')
        assert client.get('/ready').json()['activation_id'] == expected
        response = cycle(client, expected, context)
        assert response.status_code == 200, response.text
        assert response.json()['actions'][0]['value'] == (2 if point == 'before_commit' else 6)


def test_lost_commit_ack_disables_live_serving_but_preserves_success_receipt(managed, monkeypatch):
    settings, config, request, *_ = managed
    with TestClient(create_app(config=settings)) as client:
        _, context, old = ready(client, config, request)
        second, _ = new_candidate(client, managed)
        store = client.app.state.preparation.lifecycle.store
        commit = store.commit
        def lost_ack(*args): commit(*args); raise OSError('ack lost')
        monkeypatch.setattr(store, 'commit', lost_ack)
        receipt = activate(client, second, context, old, 'lost-ack')
        assert receipt['status'] == 'succeeded'
        assert client.get('/ready').status_code == 503
        new = receipt['resource_id']
    with TestClient(create_app(config=settings)) as client:
        assert cycle(client, new, context).status_code == 200


def test_activation_deadline_prevents_late_publication(managed, monkeypatch):
    settings, config, request, *_ = managed
    with TestClient(create_app(config=settings)) as client:
        _, context, old = ready(client, config, request)
        second, _ = new_candidate(client, managed)
        service = client.app.state.preparation
        build = service.build_execution
        now = service.lifecycle.store.now()
        def slow(*args, **kwargs):
            result = build(*args, **kwargs)
            monkeypatch.setattr(service.lifecycle.store, 'now', lambda: now + 1000)
            return result
        monkeypatch.setattr(service, 'build_execution', slow)
        receipt = activate(client, second, context, old, 'expired')
        assert receipt['status'] == 'failed' and receipt['error']['code'] == 'activation_expired'
        assert client.get('/ready').json()['activation_id'] == old
        assert cycle(client, old, context).status_code == 200


def test_onnx_activation_and_restart_retain_safety_and_engine(managed):
    settings, config, request, *_ = managed
    onnx_identity(managed)
    with TestClient(create_app(config=settings)) as client:
        _, context, activation_id = ready(client, config, request)
        result = cycle(client, activation_id, context, 9).json()
        assert result['proposed_actions'][0]['proposed_value'] == 9
        assert result['actions'][0]['value'] == 2
    with TestClient(create_app(config=settings)) as client:
        result = cycle(client, activation_id, context, 9)
        assert result.status_code == 200, result.text
        assert result.json()['actions'][0]['value'] == 2


def test_rejected_message_adapter_candidate_keeps_active_available(managed):
    settings, config, request, *_ = managed
    with TestClient(create_app(config=settings)) as client:
        _, context, old = ready(client, config, request)
        changed = copy.deepcopy(config); changed['revision'] = 'new-message'
        changed['message']['adapter_revision'] = 'unsupported-new-percepta'
        bad = store_and_prepare(client, changed, {**request,'configuration_revision':'new-message'}, key='new-message')
        assert bad['state'] == 'rejected'
        response = client.post('/admin/v1/activations', headers={**AUTH,'Idempotency-Key':'rejected'}, json=dict(
            schema_version='activation.v2-draft', candidate_id=bad['candidate_id'], expected_context_id=context, expected_activation_id=old))
        assert response.status_code == 409
        assert cycle(client, old, context).status_code == 200


def test_one_to_two_chargers_requires_compatible_bundle_and_context(managed):
    settings, config, request, manifest, bundle, write = managed
    with TestClient(create_app(config=settings)) as client:
        candidate, context, old = ready(client, config, request)
        changed = copy.deepcopy(config)
        changed['revision'] = 'two-chargers'; changed['installation']['revision'] = 'installation-2'
        equipment = copy.deepcopy(changed['installation']['equipment'][0]); equipment['equipment_id'] = 'EVC02'
        changed['installation']['equipment'].append(equipment)
        channel = copy.deepcopy(changed['installation']['channels'][-1]); channel.update(channel_id='charger-2-setpoint',equipment_id='EVC02')
        changed['installation']['channels'].append(channel)
        changed['installation']['aggregate_limits'][0]['terms'].append({'channel_id':'charger-2-setpoint','coefficient':1})
        new_context = declare(client, changed, 'two', context, old)
        bad = store_and_prepare(client, changed, {**request,'configuration_revision':'two-chargers'}, key='two-bad')
        assert bad['state'] == 'rejected'
        assert any(i['code']=='uncovered_action' for i in bad['compatibility']['issues'])
        assert cycle(client, old, new_context).status_code == 503
        changed['revision'] = 'two-complete'
        changed['bundle']['outputs'].append({'name':'dispatch_2','meaning':copy.deepcopy(channel['meaning'])})
        changed['bindings']['outputs'].append({'signal_name':'dispatch_2','channel_id':'charger-2-setpoint'})
        changed['safety']['action_rules'].append({'rule_id':'second-bounds','channel_id':'charger-2-setpoint','decimal_places':1})
        changed['safety']['aggregate_rule']['curtailment_order'].append('charger-2-setpoint')
        manifest['environment']['action_names'].append('dispatch_2')
        manifest['environment']['action_bounds'][0]['low'].append(0)
        manifest['environment']['action_bounds'][0]['high'].append(7.4)
        write(); changed['bundle']['manifest_sha256'] = config['bundle']['manifest_sha256']
        (bundle/'policy.json').write_text('{"default_actions":{"dispatch":2,"dispatch_2":3},"rules":[]}')
        good = store_and_prepare(client, changed, {**request,'configuration_revision':'two-complete'}, key='two-good')
        assert good['state'] == 'ready', good
        activated = activate(client, good, new_context, old, 'two-good')
        assert activated['status'] == 'succeeded', activated
        result = cycle(client, activated['resource_id'], new_context).json()
        assert {a['equipment_id'] for a in result['actions']} == {'EVC01','EVC02'}
        # The original one-charger candidate cannot roll back the declared house.
        refused = activate(client, candidate, new_context, activated['resource_id'], 'rollback')
        assert refused['status'] == 'failed' and refused['error']['code'] == 'installation_changed'


def test_fsync_failure_before_commit_preserves_previous_activation(managed, monkeypatch):
    import app.lifecycle.service as module
    settings, config, request, *_ = managed
    with TestClient(create_app(config=settings)) as client:
        _, context, old = ready(client, config, request)
        second, _ = new_candidate(client, managed)
        def fail(*args): raise OSError('storage write failed')
        monkeypatch.setattr(module, 'sync_tree', fail)
        result = activate(client, second, context, old, 'fsync-failure')
        assert result['status'] == 'failed'
        assert cycle(client, old, context).status_code == 200
        assert client.get('/ready').json()['activation_id'] == old


def test_concurrent_activation_submissions_have_single_winner(managed):
    settings, config, request, *_ = managed
    with TestClient(create_app(config=settings)) as client:
        candidate, context, old = ready(client, config, request)
        lifecycle = client.app.state.preparation.lifecycle
        body = Activate(schema_version='activation.v2-draft', candidate_id=candidate['candidate_id'],
                        expected_context_id=context, expected_activation_id=old)
        def submit(key):
            try: return lifecycle.submit(key, body)
            except PreparationError as exc: return exc.code
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(submit, ['a','b']))
        assert results.count('activation_busy') == 1
        operation = next(x for x in results if x.startswith('act_'))
        lifecycle.activate(operation)
        assert lifecycle.receipt(operation)['status'] == 'succeeded'


def test_rollback_same_installation_is_a_new_validated_activation(managed):
    settings, config, request, *_ = managed
    with TestClient(create_app(config=settings)) as client:
        first, context, old = ready(client, config, request)
        second, _ = new_candidate(client, managed)
        new = activate(client, second, context, old, 'second')['resource_id']
        rollback = activate(client, first, context, new, 'rollback')
        assert rollback['status'] == 'succeeded' and rollback['resource_id'] not in (old,new)
        result = cycle(client, rollback['resource_id'], context).json()
        assert result['actions'][0]['value'] == 2
        assert result['activation']['previous_activation_id'] == new


def test_native_percepta_observes_one_installation_over_its_own_credential(managed):
    from pydantic import SecretStr
    settings, config, request, *_ = managed
    settings.deployment_model_id = "energy"
    settings.percepta_cycle_token = SecretStr("cycle-only-key-for-local-test-123456789")
    site = settings.installation_site_id
    url = f"/inference/sites/{site}/models/energy/observe"
    headers = {"Authorization": "Bearer " + settings.percepta_cycle_token.get_secret_value(),
               "X-Installation-Context": "pending"}
    channel = next(c for c in config["installation"]["channels"] if c.get("message_path"))
    channel["message_path"] = ["entities", "meter", "parameters", "load", "final_values", "0", "value"]
    message = {"period_start": "2026-11-22 03:45:00 +0000", "period_end": "2026-11-22 04:00:00 +0000",
               "entities": {"meter": {"parameters": {"load": {"measurement_unit": "kW", "final_values": [{"value": 5}]}}}}}
    with TestClient(create_app(config=settings)) as client:
        assert client.get(url, headers=headers).status_code == 503
        _, context, activation_id = ready(client, config, request)
        assert client.post(url, headers=headers, json=message).status_code == 409
        headers["X-Installation-Context"] = context
        assert client.get(url, headers=headers).json() == dict(site_id=site, model_id="energy", ready=True, context_id=context)
        assert client.get(url.replace("/energy/", "/other/"), headers=headers).status_code == 404
        assert client.get(url.replace(f"/{site}/", "/other-house/"), headers=headers).status_code == 404
        assert client.get("/admin/v1/status", headers=headers).status_code == 401
        assert client.post(url, headers={**headers, **AUTH}, json=message).status_code == 401
        assert client.post(url, headers=headers, json={**message, "period_end": message["period_start"]}).status_code == 422
        observed = client.post(url, headers=headers, json=message)
        assert observed.status_code == 200, observed.text
        assert set(observed.json()) == {"decision"}
        decision = observed.json()["decision"]
        assert decision["activation"]["activation_id"] == activation_id
        assert decision["context"]["site_id"] == site and decision["cycle_id"].startswith(site + "/")
        def reference(active_id):
            response = client.post("/inference/v2", headers=AUTH, json=dict(
                expected_activation_id=active_id, expected_context_id=context, cycle_id="reference", features=message))
            assert response.status_code == 200, response.text
            return response.json()["actions"]
        assert decision["actions"] == reference(activation_id)
        candidate, _ = new_candidate(client, managed)
        updated = activate(client, candidate, context, activation_id, key="new-version")
        assert updated["status"] == "succeeded"
        second = client.post(url, headers=headers, json=message).json()["decision"]
        assert second["activation"]["activation_id"] == updated["resource_id"]
        assert second["cycle_id"] == decision["cycle_id"]
        assert second["actions"] == reference(updated["resource_id"])
        assert second["actions"] != decision["actions"]
