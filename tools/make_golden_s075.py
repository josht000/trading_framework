"""Compatibility CLI for the generic golden copier.

  python tools/make_golden_s075.py [--variant big10]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.make_golden import make_golden  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="baseline")
    args = parser.parse_args()
    make_golden("s075_residual_reversal", args.variant, root=ROOT)


if __name__ == "__main__":
    main()
