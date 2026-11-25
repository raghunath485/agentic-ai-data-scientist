# ==============================================================================
# 🌌 Streamlit Cloud Entry Point
# ==============================================================================
# Streamlit Cloud looks for `streamlit_app.py` by default.
# This file simply executes the main application module (app.py).

import os, sys

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Run the main application
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py"), encoding="utf-8").read())
