"""Validate four-column, full-ten-class competition submission without Test labels."""
import argparse
import json
from pathlib import Path

from cxr.batch.test_results import read_test_probabilities, validate_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path, help='Batch plan YAML')
    parser.add_argument('--submission', required=True, type=Path)
    parser.add_argument('--test-probabilities', type=Path)
    parser.add_argument('--expected-studies', type=int, default=1414)
    args = parser.parse_args()
    from cxr.batch.plan import load_plan
    load_plan(args.config)
    expected = (read_test_probabilities(args.test_probabilities, expected_studies=args.expected_studies)['keys']
                if args.test_probabilities else None)
    print(json.dumps(validate_submission(args.submission, expected, args.expected_studies)))


if __name__ == '__main__':
    main()
