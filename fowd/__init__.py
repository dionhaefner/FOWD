"""
__init__.py

Initialize global setup
"""

# we use multiprocessing, so prevent other libraries from using threads
import os
os.environ['OMP_NUM_THREADS'] = '1'
del os

# get version
try:
    from fowd._version import version as __version__  # noqa: F401
except ImportError:
    # package is not installed
    raise RuntimeError(
        'FOWD has not been installed correctly. Please run `pip install -e .` or '
        '`python setup.py develop` in the FOWD package folder.'
    ) from None
