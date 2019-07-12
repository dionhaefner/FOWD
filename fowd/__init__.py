"""
__init__.py

Initialize global setup
"""

# we use multiprocessing, so prevent other libraries from using threads
import os
os.environ['OMP_NUM_THREADS'] = '1'
del os

# get version
from pkg_resources import get_distribution, DistributionNotFound  # noqa: E402

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    raise RuntimeError(
        'FOWD has not been installed correctly. Please run `pip install -e .` or '
        '`python setup.py develop` in the FOWD package folder.'
    ) from None

del get_distribution, DistributionNotFound
