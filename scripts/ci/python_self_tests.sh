#!/usr/bin/env bash
# The Python scripts' own self-tests: pure helpers and the parity fixture, no torch, no download.
# Run with `-I`, as the service and the Rust tests run them. A missing numpy or Pillow is an error
# here, because CI installed them (scripts/ci/requirements-test.txt): the self-tests of the JPEG
# path would otherwise pass by leaving it out.
# Usage: PYTHON=<interpreter, default python> scripts/ci/python_self_tests.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=${PYTHON:-python}

"$PY" -I -c 'import sys, numpy, PIL; print(sys.executable, sys.version.split()[0], "numpy", numpy.__version__, "pillow", PIL.__version__)'
"$PY" -I scripts/autolabel.py --self-test --parity tests/data/overlaps_parity.json
"$PY" -I scripts/import_footage.py --self-test
"$PY" -I scripts/research/eval_autolabel.py --self-test
echo "python self-tests ok"
