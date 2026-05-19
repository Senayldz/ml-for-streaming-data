
import importlib

libs = ["wfdb", "river", "streamlit", "plotly", "pandas", "numpy", "scipy"]
for lib in libs:
    try:
        importlib.import_module(lib)
        print(f"{lib}: Installed")
    except ImportError:
        print(f"{lib}: NOT INSTALLED")
