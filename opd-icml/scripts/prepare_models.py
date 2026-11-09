"""Download/cache pinned OPD model pairs without starting training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_assets import model_profiles, prepare_models, read_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=model_profiles(), required=True)
    parser.add_argument('--student', type=Path)
    parser.add_argument('--teacher', type=Path)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--model-dir', type=Path, help='Root for Gemma text-only exports; defaults to models/')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(dict(scale=args.scale, models=read_models(args.scale)), indent=2))
        return
    try:
        paths = prepare_models(args.scale, args.student, args.teacher, args.cache_dir, args.model_dir)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(dict(scale=args.scale, **paths), indent=2))


if __name__ == '__main__':
    main()
