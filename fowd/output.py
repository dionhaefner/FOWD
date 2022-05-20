"""
output.py

Output metadata and I/O handling.
"""

import uuid
import datetime

from .constants import FREQUENCY_INTERVALS, SEA_STATE_INTERVALS
from .operators import get_proc_version

import numpy as np
import netCDF4


# times are measured in milliseconds since this date
TIME_ORIGIN = '1980-01-01'

# fill values to use
FILL_VALUE_NUMBER = -9999
FILL_VALUE_STR = 'MISSING'

# chunk sizes to use for each dimension
CHUNKSIZES = {
    'meta_station_name': 1,
    'wave_id_local': 10_000,
    'meta_frequency_band': len(FREQUENCY_INTERVALS),
}

DATASET_VARIABLES = dict(
    # metadata
    meta_source_file_name=dict(
        dims=('wave_id_local',),
        dtype=str,
        attrs=dict(
            long_name='File name of raw input data file',
        )
    ),
    meta_source_file_uuid=dict(
        dims=('wave_id_local',),
        dtype=str,
        attrs=dict(
            long_name='UUID of raw input data file',
        )
    ),

    # wave data
    wave_start_time=dict(
        dims=('wave_id_local',),
        dtype='int64',
        attrs=dict(
            long_name='Wave start time',
            units=f'milliseconds since {TIME_ORIGIN}',
        ),

    ),

    wave_end_time=dict(
        dims=('wave_id_local',),
        dtype='int64',
        attrs=dict(
            long_name='Wave end time',
            units=f'milliseconds since {TIME_ORIGIN}',
        ),
    ),

    wave_zero_crossing_period=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Wave zero-crossing period relative to 30m sea surface elevation',
            units='seconds',
            comment='Zero-crossings determined through linear interpolation',
        )
    ),

    wave_zero_crossing_wavelength=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Wave zero-crossing wavelength relative to 30m sea surface elevation',
            units='meters',
        )
    ),

    wave_raw_elevation=dict(
        dims=('wave_id_local',),
        dtype='vlen',
        attrs=dict(
            long_name='Raw surface elevation relative to 30m sea surface elevation',
            units='meters',
            comment='Spacing in time as given by meta_sampling_rate',
        )
    ),

    wave_crest_height=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Wave crest height relative to 30m sea surface elevation',
            units='meters',
        )
    ),

    wave_trough_depth=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Wave trough depth relative to 30m sea surface elevation',
            units='meters',
        )
    ),

    wave_height=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Absolute wave height relative to 30m sea surface elevation',
            units='meters',
        )
    ),

    wave_ursell_number=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Ursell number',
            units='1',
            valid_min=0,
        )
    ),

    wave_maximum_elevation_slope=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Maximum slope of surface elevation in time',
            units='m s-1',
        )
    ),

    # station metadata
    meta_deploy_latitude=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Deploy latitude of instrument',
            units='degrees_north',
        )
    ),

    meta_deploy_longitude=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Deploy longitude of instrument',
            units='degrees_east',
        )
    ),

    meta_water_depth=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Water depth at deployment location',
            units='meters',
            positive='down',
        )
    ),

    meta_sampling_rate=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Measurement sampling frequency in time',
            units='hertz',
        )
    ),
)

# sea state parameter metadata
for interval in SEA_STATE_INTERVALS:
    if not isinstance(interval, str):
        interval = f'{interval}m'

    if interval == 'dynamic':
        DATASET_VARIABLES.update({
            'sea_state_dynamic_window_length': dict(
                dims=('wave_id_local',),
                dtype='int64',
                attrs=dict(
                    long_name='Length of dynamically computed sea state window',
                    units='minutes',
                )
            )
        })

    DATASET_VARIABLES.update({
        f'sea_state_{interval}_start_time': dict(
            dims=('wave_id_local',),
            dtype='int64',
            attrs=dict(
                long_name='Sea state aggregation start time',
                units=f'milliseconds since {TIME_ORIGIN}',
            )
        ),
        f'sea_state_{interval}_end_time': dict(
            dims=('wave_id_local',),
            dtype='int64',
            attrs=dict(
                long_name='Sea state aggregation end time',
                units=f'milliseconds since {TIME_ORIGIN}',
            )
        ),
        f'sea_state_{interval}_significant_wave_height_spectral': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Significant wave height estimated from wave spectrum (Hm0)',
                units='meters',
            )
        ),
        f'sea_state_{interval}_significant_wave_height_direct': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Significant wave height estimated from wave history (H1/3)',
                units='meters',
            )
        ),
        f'sea_state_{interval}_maximum_wave_height': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Maximum wave height estimated from wave history',
                units='meters',
            )
        ),
        f'sea_state_{interval}_rel_maximum_wave_height': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name=(
                    'Maximum wave height estimated from wave history '
                    'relative to spectral significant wave height'
                ),
                units='1',
            )
        ),
        f'sea_state_{interval}_mean_period_direct': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Mean zero-crossing period estimated from wave history',
                units='seconds',
            )
        ),
        f'sea_state_{interval}_mean_period_spectral': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Mean zero-crossing period estimated from wave spectrum',
                units='seconds',
            )
        ),
        f'sea_state_{interval}_skewness': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Skewness of sea surface elevation',
                units='1',
            )
        ),
        f'sea_state_{interval}_kurtosis': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Excess kurtosis of sea surface elevation',
                units='1',
            )
        ),
        f'sea_state_{interval}_valid_data_ratio': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Ratio of valid measurements to all measurements',
                valid_min=0,
                valid_max=1,
                units='1',
            )
        ),
        f'sea_state_{interval}_peak_wave_period': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Dominant wave period',
                units='seconds',
            )
        ),
        f'sea_state_{interval}_peak_wavelength': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Dominant wavelength',
                units='meters',
            )
        ),
        f'sea_state_{interval}_steepness': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Dominant wave steepness',
                units='1',
            )
        ),
        f'sea_state_{interval}_bandwidth_peakedness': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name=(
                    'Spectral bandwidth estimated through spectral peakedness '
                    '(quality factor)',
                ),
                units='1',
            )
        ),
        f'sea_state_{interval}_bandwidth_narrowness': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Spectral bandwidth estimated through spectral narrowness',
                units='1',
            )
        ),
        f'sea_state_{interval}_benjamin_feir_index_peakedness': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Benjamin-Feir index estimated through steepness and peakedness',
                units='1',
            )
        ),
        f'sea_state_{interval}_benjamin_feir_index_narrowness': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Benjamin-Feir index estimated through steepness and narrowness',
                units='1',
            )
        ),
        f'sea_state_{interval}_crest_trough_correlation': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Crest-trough correlation parameter (r) estimated from spectral density',
                units='1',
                valid_min=0,
                valid_max=1,
            )
        ),
        f'sea_state_{interval}_energy_in_frequency_interval': dict(
            dims=('wave_id_local', 'meta_frequency_band'),
            dtype='float32',
            attrs=dict(
                long_name='Total energy density contained in frequency band',
                units='J m-2',
            )
        ),
        f'sea_state_{interval}_rel_energy_in_frequency_interval': dict(
            dims=('wave_id_local', 'meta_frequency_band'),
            dtype='float32',
            attrs=dict(
                long_name='Relative energy contained in frequency band',
                units='1',
                valid_min=0,
                valid_max=1,
            )
        ),
        f'sea_state_{interval}_spectrum': dict(
            dims=('wave_id_local',),
            dtype='vlen',
            attrs=dict(
                long_name='Wave spectral density',
                units='m2 s-1',
                comment=f'Frequency resolution is given by sea_state_{interval}_spectrum_df',
            )
        ),
        f'sea_state_{interval}_spectrum_df': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Wave spectral density - frequency resolution',
                units='Hz',
            )
        ),
    })

# directional parameter metadata
DIRECTIONAL_VARIABLES = dict(
    direction_sampling_time=dict(
        dims=('wave_id_local',),
        dtype='int64',
        attrs=dict(
            long_name='Time at which directional quantities are sampled',
            units=f'milliseconds since {TIME_ORIGIN}',
        ),
    ),
    direction_dominant_spread_in_frequency_interval=dict(
        dims=('wave_id_local', 'meta_frequency_band'),
        dtype='float32',
        attrs=dict(
            long_name='Dominant directional spread in frequency band',
            units='degrees',
            valid_min=0,
            valid_max=90,
        )
    ),
    direction_dominant_direction_in_frequency_interval=dict(
        dims=('wave_id_local', 'meta_frequency_band'),
        dtype='float32',
        attrs=dict(
            long_name='Dominant wave direction in frequency band',
            units='degrees',
            valid_min=0,
            valid_max=360,
        )
    ),
    direction_peak_wave_direction=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Peak wave direction relative to normal-north',
            units='degrees',
            valid_min=0,
            valid_max=360
        )
    ),
    direction_directionality_index=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name=(
                'Directionality index R (squared ratio of directional spread and '
                'spectral bandwidth)'
            ),
            units='1',
            valid_min=0,
        )
    )
)


freq_lower, freq_upper = list(zip(*FREQUENCY_INTERVALS))

# additional output variables that are constant across stations
EXTRA_VARIABLES = dict(
    meta_frequency_band_lower=dict(
        data=np.array(freq_lower, dtype='float32'),
        dims=('meta_frequency_band',),
        attrs=dict(
            long_name='Lower limit of frequency band',
            units='hertz',
        ),
    ),
    meta_frequency_band_upper=dict(
        data=np.array(freq_upper, dtype='float32'),
        dims=('meta_frequency_band',),
        attrs=dict(
            long_name='Upper limit of frequency band',
            units='hertz',
        ),
    ),
)


# attributes added to coordinate variables
COORD_ATTRS = dict(
    meta_station_name=dict(
        long_name='Name of original measurement station',
    ),
    wave_id_local=dict(
        long_name='Incrementing wave ID for given station',
        comment=(
            'This ID is not guaranteed to denote the same wave between data versions.'
        ),
    ),
    meta_frequency_band=dict(
        long_name='Index of frequency band',
        comment=(
            'Frequency ranges are given by '
            '(meta_frequency_band_lower, meta_frequency_band_upper)'
        ),
    ),
)


def get_dataset_metadata(station_name, start_time, end_time, extra_metadata=None):
    """Get all metadata attributes related to the whole dataset."""
    dataset_metadata = dict(
        id=f'FOWD_{station_name}',
        title=f'Free Ocean Wave Dataset (FOWD), station {station_name}',
        summary=(
            'A catalogue of ocean waves and associated sea states, derived from in-situ '
            'measurement data.'
        ),
        project='Free Ocean Wave Dataset (FOWD)',
        keywords=(
            'EARTH SCIENCE, OCEANS, OCEAN WAVES, GRAVITY WAVES, WIND WAVES, '
            'SIGNIFICANT WAVE HEIGHT, WAVE FREQUENCY, WAVE PERIOD, WAVE SPECTRA'
        ),
        processing_version=get_proc_version(),
        processing_url='https://github.com/dionhaefner/FOWD',
        date_created=f'{datetime.datetime.utcnow():%Y-%m-%dT%H:%M:%S.%f}',
        uuid=str(uuid.uuid4()),
        creator_name='NBI Copenhagen',
        creator_url='https://www.nbi.ku.dk/english/research/pice/oceanography/',
        creator_email='dion.haefner@nbi.ku.dk',
        institution='Niels Bohr Institute, University of Copenhagen',
        geospatial_lat_units='degrees_north',
        geospatial_lat_resolution=1e-5,
        geospatial_lon_units='degrees_east',
        geospatial_lon_resolution=1e-5,
        geospatial_vertical_units='meters',
        geospatial_vertical_origin='sea surface height',
        geospatial_vertical_positive='up',
        time_coverage_start=str(start_time),
        time_coverage_end=str(end_time),
        source='insitu observations',
        license='These data may be redistributed and used without restriction.',
    )

    if extra_metadata is not None:
        for key, val in extra_metadata.items():
            dataset_metadata[key] = ''.join([dataset_metadata.get(key, ''), str(val)])

    return dataset_metadata


def write_records(wave_record_iterator, filename, station_name, extra_metadata=None,
                  include_direction=False):
    """Write given wave records in FOWD's netCDF4 output format.

    First argument is an iterable of chunks of wave records.
    """

    dimension_data = (
        # (name, dtype, data)
        ('meta_station_name', str, np.array([np.string_(station_name)])),
        ('wave_id_local', 'int64', None),
        ('meta_frequency_band', 'uint8', np.arange(len(FREQUENCY_INTERVALS))),
    )

    variables = DATASET_VARIABLES

    if include_direction:
        variables.update(DIRECTIONAL_VARIABLES)

    with netCDF4.Dataset(filename, 'w') as f:
        # create variable length dtype
        vlen_type = f.createVLType('float32', 'float_array')

        # create dimensions
        for dim, dtype, val in dimension_data:
            if val is None:
                f.createDimension(dim, None)
            else:
                f.createDimension(dim, len(val))

            extra_args = dict(
                zlib=True,
                fletcher32=True,
                chunksizes=[CHUNKSIZES[dim]]
            )

            v = f.createVariable(dim, dtype, (dim,), **extra_args)

            if val is not None:
                v[:] = val

        for name, meta in variables.items():
            # add meta_station_name as additional scalar dimension
            dims = ('meta_station_name',) + meta['dims']

            extra_args = dict(
                zlib=True,
                fletcher32=True,
                chunksizes=[CHUNKSIZES[dim] for dim in dims]
            )

            # determine dtype
            if meta['dtype'] == 'vlen':
                dtype = vlen_type
            else:
                dtype = meta['dtype']

            # add correct fill value
            is_number = np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer)
            if is_number and dtype is not vlen_type:
                extra_args.update(fill_value=FILL_VALUE_NUMBER)
            elif np.issubdtype(dtype, np.character):
                extra_args.update(fill_value=FILL_VALUE_STR)

            # create variable
            v = f.createVariable(name, dtype, dims, **extra_args)

            # attach attributes
            for attr, val in meta['attrs'].items():
                setattr(v, attr, val)

        # write data
        start_time = end_time = None
        current_wave_idx = 0
        for chunk in wave_record_iterator:
            if not chunk:
                continue

            chunk_length = len(chunk['wave_id_local'])
            chunk_slice = slice(
                current_wave_idx,
                current_wave_idx + chunk_length
            )
            current_wave_idx += chunk_length

            f.variables['wave_id_local'][chunk_slice] = chunk['wave_id_local']

            for name, meta in variables.items():
                v = f.variables[name]
                data = np.asarray(chunk[name])

                if name == 'wave_start_time':
                    if start_time is None or np.min(data) < start_time:
                        start_time = np.min(data)
                elif name == 'wave_end_time':
                    if end_time is None or np.max(data) > end_time:
                        end_time = np.max(data)

                # convert datetimes to ms since time origin
                if np.issubdtype(data.dtype, np.datetime64):
                    data = (data - np.datetime64(TIME_ORIGIN)) / np.timedelta64(1, 'ms')

                # convert timedelta64 to target unit
                if np.issubdtype(data.dtype, np.timedelta64):
                    unit = meta.get('attrs', {}).get('units')

                    if unit == 'seconds':
                        target_unit = 's'
                    elif unit == 'minutes':
                        target_unit = 'm'
                    elif unit == 'hours':
                        target_unit = 'h'
                    else:
                        raise ValueError(f'Got unrecognized time unit {unit}')

                    data = data / np.timedelta64(1, target_unit)

                if meta['dtype'] == 'vlen':
                    obj_data = np.empty(data.shape[0], dtype="object")
                    obj_data[...] = list(data)
                    data = obj_data

                v[0, chunk_slice, ...] = data

        # set global metadata
        dataset_metadata = get_dataset_metadata(
            station_name, start_time, end_time, extra_metadata=extra_metadata
        )
        for attr, val in dataset_metadata.items():
            setattr(f, attr, val)

        # add extra variables
        for name, meta in EXTRA_VARIABLES.items():
            extra_args = dict(
                zlib=True,
                fletcher32=True,
                chunksizes=[CHUNKSIZES[dim] for dim in meta['dims']]
            )
            v = f.createVariable(name, meta['data'].dtype, meta['dims'], **extra_args)
            v[:] = meta['data']
            for attr, val in meta['attrs'].items():
                setattr(v, attr, val)

        # add coordinate attributes
        for coord, attrs in COORD_ATTRS.items():
            for attr, val in attrs.items():
                setattr(f[coord], attr, val)
