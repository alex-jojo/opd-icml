#!/usr/bin/env bash
set -euo pipefail
# Run inside the activated Linux training/evaluation environment.
# LCB's official judging core is vendored at a pinned commit; provider SDKs are unnecessary.
python -m pip install 'evalplus==0.3.1'
python -m pip check
