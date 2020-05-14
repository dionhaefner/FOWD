# FOWD
:ocean: A free ocean wave dataset, ready for your ML application.

## Usage

After installing the Python code (e.g. via `pip install git+https://github.com/dionhaefner/FOWD.git`),
you can use the command line tool `fowd` to create a FOWD dataset from a raw source.

Currently, the best supported source is CDIP data:

```bash
$ fowd process-cdip 433p1 -o fowd-cdip-out
```

will process all data located in the `433p1` folder.

You can also process generic netCDF input files:

```bash
$ fowd process-generic infile.nc -o outdir
```

Generic inputs must have the following structure:

```
Variables:
- time
- displacement

Attributes:
- sampling_rate
- water_depth
- longitude
- latitude
```

## Post-processing

Data processing writes QC information in JSON format. You can visualize records in that QC file by using

```bash
$ fowd postprocess qcfile.json
```

## Testing

Run tests and sanity checks via

```bash
$ fowd run-tests
```

Test results are checked automatically, but sanity checks have to be inspected manually.
