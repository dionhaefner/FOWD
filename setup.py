from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    # metadata
    name='fowd',
    description='A free ocean wave dataset, ready for your ML application',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/dionhaefner/FOWD',
    author='Dion Häfner',
    author_email='dion.haefner@nbi.ku.dk',
    # module
    packages=find_packages(exclude=['docs', 'tests']),
    python_requires='>=3.6',
    use_scm_version={
        'write_to': 'fowd/_version.py'
    },
    # dependencies
    setup_requires=[
        'setuptools_scm',
        'setuptools_scm_git_archive',
        'numpy'
    ],
    install_requires=[
        'bottleneck',
        'click',
        'filelock',
        'matplotlib',
        'netcdf4',
        'numpy',
        'scipy>=1.0',
        'tqdm',
        'xarray',
    ],
    extras_require={
        'testing': [
            'pytest',
        ]
    },
    # CLI
    entry_points='''
        [console_scripts]
        fowd=fowd.cli:entrypoint
    ''',
)
