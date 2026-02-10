#!/usr/bin/env python3
"""Wrapper to run lm_eval with OLMo model registration.

Imports hf_olmo to register OLMo's custom AutoConfig/AutoModel/AutoTokenizer
classes, then delegates to lm_eval's CLI.
"""

import sys

# Register OLMo custom HF classes before lm_eval tries to load the model
import hf_olmo  # noqa: F401

from lm_eval.__main__ import cli_evaluate

cli_evaluate()
