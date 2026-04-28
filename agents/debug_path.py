import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("SYSPATH:", sys.path)
try:
    from tools.registry import registry
    print("SUCCESS: tools.registry imported")
except ImportError as e:
    print("FAILURE: tools.registry import failed:", e)
