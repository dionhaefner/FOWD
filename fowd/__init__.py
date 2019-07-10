"""
__init__.py

Initialize global setup
"""

# get version
try:
    from fowd._version import version as __version__  # noqa: F401
except ImportError:
    # package is not installed
    raise RuntimeError(
        'FOWD has not been installed correctly. Please run `pip install -e .` or '
        '`python setup.py develop` in the FOWD package folder.'
    ) from None
