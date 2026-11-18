"""Compatibility entry point for the original 4B math Base evaluation."""
import os
from evaluate_base import main


if __name__ == '__main__':
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    main()
