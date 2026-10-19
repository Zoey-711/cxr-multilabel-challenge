"""Single-host CPU/RAM and ZIP random-read admission check."""
import os
import random
import time
import zipfile
from pathlib import Path


def available_memory_bytes():
    if os.name == 'posix' and Path('/proc/meminfo').is_file():
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1])*1024
    raise RuntimeError('Cannot establish available host RAM; configure concurrency after inspection')


def check_resources(archive, concurrency, plan):
    tasks=[t for t in plan['tasks'] if t['enabled']]
    per_task_cores=max(t.get('estimated_cpu_cores',6) for t in tasks)
    per_task_ram=max(t.get('estimated_host_ram_gib',12) for t in tasks)
    cores=os.cpu_count() or 0
    free=available_memory_bytes()
    if cores < concurrency*per_task_cores:
        raise RuntimeError(f'{cores} logical CPUs insufficient for {concurrency} concurrent loaders; '
                           f'require {concurrency*per_task_cores} or lower authorized concurrency')
    if free < concurrency*per_task_ram*(1024**3):
        raise RuntimeError(f'{free/(1024**3):.1f}GiB available RAM insufficient for {concurrency} tasks; '
                           f'require {concurrency*per_task_ram}GiB or lower authorized concurrency')
    path=Path(archive)
    with zipfile.ZipFile(path if path.is_file() else path/'train.zip') as handle:
        members=[n for n in handle.namelist() if '/train/' in n.lower() and n.lower().endswith(('.jpg','.png'))]
        if not members:
            raise ValueError('No official Train images for random-read preflight')
        sample=random.Random(42).sample(members,min(24,len(members)))
        start=time.monotonic()
        total=sum(len(handle.read(name)) for name in sample)
        seconds=max(time.monotonic()-start,1e-6)
    throughput=total/seconds/1e6
    if throughput < 20:
        raise RuntimeError(f'ZIP random-read sample only {throughput:.1f} MB/s; '
                           '8 loaders may starve GPUs. Move data to authorized faster storage or lower concurrency')
    return {'logical_cpus':cores,'available_ram_gib':round(free/(1024**3),2),
            'concurrent_tasks':concurrency,'estimated_cpu_per_task':per_task_cores,
            'estimated_ram_gib_per_task':per_task_ram,'random_read_sample_files':len(sample),
            'random_read_mb_per_s':round(throughput,2),
            'note':'Short random-read probe; monitor actual 8-loader throughput during formal run'}
