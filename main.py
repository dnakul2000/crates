#!/usr/bin/env python3
"""Crates — AI-Powered Sample Pack Factory."""

import multiprocessing
import sys

from crates.app import run


def main():
    multiprocessing.freeze_support()
    sys.exit(run())


if __name__ == "__main__":
    main()
