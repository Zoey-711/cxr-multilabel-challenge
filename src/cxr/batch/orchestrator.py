"""One authorized launch for the frozen 22-task queue and four dependent G/H tasks."""
import json
from pathlib import Path

from cxr.batch.plan import write_json


def _base_complete(root, plan):
    state_file = Path(root)/'queue_state.json'
    if not state_file.is_file():
        return False
    state = json.loads(state_file.read_text(encoding='utf-8'))
    return all(state['tasks'].get(task['id'], {}).get('state') == 'complete'
               for task in plan['tasks'] if task['enabled'])


def _extension_complete(root, plan):
    state_file = Path(root)/'extension_state.json'
    if not state_file.is_file():
        return False
    state = json.loads(state_file.read_text(encoding='utf-8'))
    return all(state['tasks'].get(slot['id'], {}).get('state') == 'complete'
               for slot in plan['conditional_extension']['reserved_slots'])


def run_all(plan, plan_path, root, gpus, *, total_budget, gh_budget,
            max_concurrent, preflight_gpu_hours, gpu_preflight=True):
    """Run base first; extension starts only after all frozen prerequisites pass.

    Reinvocation resumes the existing queue and never retrains complete tasks.
    Failed prerequisites preserve base artifacts and mark G/H BLOCKED.
    """
    root = Path(root)
    if not plan['conditional_extension']['enabled']:
        raise ValueError('The reviewed 26-task extension is not enabled in the sealed plan')
    if not (total_budget > gh_budget > 0 and preflight_gpu_hours >= 0):
        raise ValueError('Friend must approve a positive total 26-task budget and separate G/H subbudget')
    base_budget = total_budget - gh_budget
    if preflight_gpu_hours >= base_budget:
        raise ValueError('Preflight already exhausts the base-task GPU-hour budget')
    from cxr.batch.runner import run_queue
    from cxr.batch.extension_queue import run_extension_queue
    if not _base_complete(root, plan):
        try:
            run_queue(plan, plan_path, root, gpus, base_budget,
                      resume=(root/'queue_state.json').is_file(),
                      initial_gpu_hours=preflight_gpu_hours,
                      max_concurrent=max_concurrent)
        except Exception as exc:
            blocked = {'state': 'BLOCKED', 'reason': f'Base queue failed: {type(exc).__name__}: {exc}',
                       'completed_base_results_preserved': True,
                       'tasks': {slot['id']: {'state': 'BLOCKED'} for slot in
                                 plan['conditional_extension']['reserved_slots']}}
            write_json(root/'reports/extension_blocked.json', blocked)
            raise
    if _extension_complete(root, plan):
        return json.loads((root/'extension_state.json').read_text(encoding='utf-8'))
    return run_extension_queue(root, plan, gpus, total_budget=total_budget,
                               gh_budget=gh_budget, max_concurrent=max_concurrent,
                               resume=(root/'extension_state.json').is_file(),
                               gpu_preflight=gpu_preflight)
