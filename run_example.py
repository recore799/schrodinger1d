#!/usr/bin/env python3

import os
import sys
import subprocess

# Set PYTHONPATH to include the src directory
os.environ['PYTHONPATH'] = os.path.abspath('src')

run = 'examples/HO_gs.py'

# Run the example script
subprocess.run([sys.executable, 'examples/HO_gs.py'])
