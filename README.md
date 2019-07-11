# FOWD
:ocean: A free ocean wave dataset, ready for your ML application.

## Usage

After installing the Python code (e.g. via `pip install git+https://github.com/dionhaefner/FOWD.git`),
you can use the command line tool `fowd` to create a FOWD dataset from a raw source.

Currently, the only supported source is CDIP data:

```bash
$ fowd from-cdip 433p1
```

will process all data located in the `433p1` folder.
