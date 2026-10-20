"""Short CUDA allocation probe used by the staged remote preflight only."""
import argparse
import json
import os
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-token', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.expected_token:
        parser.error('GPU binding differs from the authorized token')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible CUDA device per worker')
    name = torch.cuda.get_device_name(0)
    x = torch.randn((1024, 1024), device='cuda')
    for _ in range(3):
        x = x @ x
    torch.cuda.synchronize()
    time.sleep(0.5)
    print(json.dumps({'token': args.expected_token, 'name': name,
                      'peak_allocated_bytes': torch.cuda.max_memory_allocated()}))


if __name__ == '__main__':
    main()
