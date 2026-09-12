#!/usr/bin/env python3
"""Thin wrapper so the tool runs as `python generate.py "a prompt"`."""

import sys

from sdchat.cli import main

if __name__ == "__main__":
    sys.exit(main())
