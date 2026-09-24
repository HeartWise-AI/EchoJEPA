import os
import sys

JEPA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(JEPA_ROOT)
# The manifest scripts live in data/ as standalone scripts, not a package.
sys.path.append(os.path.join(JEPA_ROOT, "data"))
