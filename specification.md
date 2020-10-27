# A comprehensive wave dataset

## Data sources

Our primary data source will be the CDIP buoys:

- 151 buoys
- XYZ displacement at measurement frequency 1.28Hz
- Total data size 519GB containing more than 1e10 waves
- Most buoys in coastal regions, some in deep water

However, this specification should be flexible enough to incorporate additional future datasets should they become available.


## Container format

netCDF4, one file per location.

Reasons for netCDF4:

- De-facto standard in scientific community
- Readable by all major data analysis tools
- Supports attaching metadata directly

## Preprocessing

- Reconstruct time from sampling rate and counts
- Mask out all data that has a primary flag > 2 or secondary flag > 0
- Compute and subtract 30m running mean SSH, use this to find individual waves
- Zero-crossing intervals containing any missing values are discarded

## Data schema

### Attributes

Similar to original CDIP metadata. Example:

 ```
 Attributes:
  naming_authority:                edu.ucsd.cdip
  keywords_vocabulary:             Global Change Master Directory (GCMD) Ea...
  date_created:                    2013-11-12T23:46:47Z
  date_issued:                     2013-11-12T23:46:47Z
  date_modified:                   2018-03-15T22:06:45Z
  creator_name:                    Coastal Data Information Program, SIO/UCSD
  creator_url:                     http://cdip.ucsd.edu
  creator_email:                   www@cdip.ucsd.edu
  publisher_name:                  Coastal Data Information Program, SIO/UCSD
  publisher_url:                   http://cdip.ucsd.edu
  publisher_email:                 www@cdip.ucsd.edu
  institution:                     Scripps Institution of Oceanography, Uni...
  project:                         Coastal Data Information Program (CDIP)
  processing_level:                QA/QC information available at http://cd...
  standard_name_vocabulary:        CF Standard Name Table (v23, 23 March 2013)
  Metadata_Conventions:            Unidata Dataset Discovery v1.0
  license:                         These data may be redistributed and used...
  featureType:                     timeSeries
  references:                      http://cdip.ucsd.edu/documentation
  Conventions:                     CF-1.6
  uuid:                            A09CE288-6634-4F25-996D-172B75F4F932
  nodc_template_version:           NODC_NetCDF_TimeSeries_Orthogonal_Templa...
  title:                           Directional wave and sea surface tempera...
  summary:                         Directional wave and sea surface tempera...
  keywords:                        EARTH SCIENCE, OCEANS, OCEAN WAVES, GRAV...
  id:                              CDIP_028p1_20000316-20010328
  cdm_data_type:                   Station
  history:                         2018-03-15T22:06:45Z: wnc_edit_att() - c...
  comment:                         All values are decoded directly from the...
  acknowledgment:                  CDIP is supported by the U.S. Army Corps...
  metadata_link:                   http://cdip.ucsd.edu/metadata/028p1
  contributor_name:                CDIP, CDBW/USACE
  contributor_role:                station operation, station funding
  geospatial_lat_min:              33.844467
  geospatial_lat_max:              33.862198
  geospatial_lat_units:            degrees_north
  geospatial_lat_resolution:       1e-05
  geospatial_lon_min:              -118.64234
  geospatial_lon_max:              -118.620995
  geospatial_lon_units:            degrees_east
  geospatial_lon_resolution:       1e-05
  geospatial_vertical_min:         0.0
  geospatial_vertical_max:         0.0
  geospatial_vertical_units:       meters
  geospatial_vertical_origin:      sea surface
  geospatial_vertical_positive:    up
  geospatial_vertical_resolution:  1.0
  time_coverage_start:             2000-03-16T19:00:00Z
  time_coverage_end:               2001-03-28T13:59:59Z
  time_coverage_duration:          P376DT19H00M
  time_coverage_resolution:        PT30M
  time_coverage_units:             minutes
  source:                          insitu observations
  platform:                        metaPlatform
  instrument:                      metaInstrumentation
 ```

Additional metadata:

- Processing code git commit hash
- Processing version
- Contact information / URL
- UUID

### Dimensions

- Station name (scalar)
- Local wave ID
- Frequency bracket

### Variables

By "wave" we mean the series of surface elevations (relative to the 30 minute mean elevation) from any given zero upcrossing to the next zero upcrossing (some waves might be excluded due to quality control criteria, see below).

Frequency bands:

1. below 0.05Hz (tides and seiches)
2. 0.05Hz to 0.1Hz (swell)
3. 0.1Hz to 0.25Hz (long-wave wind sea)
4. above 0.25Hz (short-wave wind sea)
5. 0.08 Hz to 0.5Hz (total wind sea)

#### Wave-specific

1. Station name (scalar)
2. Source filename
3. Source file UUID
4. Local wave ID (unique for each station)
5. Wave start time
6. Wave end time
7. Zero-crossing period
8. Zero-crossing wavelength
9.  Raw elevation over 30m SSH (variable length variable)
10. Crest height over 30m SSH
11. Trough depth below 30m SSH
12. Wave height
13. Ursell number
14. Maximum elevation slope
15. Buoy latitude
16. Buoy longitude
17. Total water depth
18. Sampling rate

#### Aggregates

Quantities are computed directly from the raw displacements. All time averages stop at wave start time.

19. Aggregate start (30m)
21. Aggregate start (10m)
22. Aggregate end (30m)
23. Aggregate end (10m)
24. Significant wave height Hm0 (30m)
25. Significant wave height Hm0 (10m)
26. Significant wave height H1/3 (30m)
27. Significant wave height H1/3 (10m)
28. Maximum wave height (30m)
29. Maximum wave height (10m)
30. Relative maximum wave height (30m)
31. Relative maximum wave height (10m)
32. Sea surface height (30m)
33. Sea surface height (10m)
34. Skewness (30m)
35. Skewness (10m)
36. Excess kurtosis (30m)
37. Excess kurtosis (10m)
38. Spectral bandwidth - peakedness (30m)
39. Spectral bandwidth - peakedness (10m)
40. Spectral bandwidth - narrowness (30m)
41. Spectral bandwidth - narrowness (10m)
42. Characteristic wave steepness (30m)
43. Characteristic wave steepness (10m)
44. Mean zero-crossing period (30m)
45. Mean zero-crossing period (10m)
46. Mean spectral period (30m)
47. Mean spectral period (10m)
48. Benjamin-Feir index - peakedness (30m)
49. Benjamin-Feir index - peakedness (10m)
50. Benjamin-Feir index - narrowness (30m)
51. Benjamin-Feir index - narrowness (10m)
52. Crest-trough correlation (30m)
53. Crest-trough correlation (10m)
54. Valid data ratio (30m)
55. Valid data ratio (10m)
56. Peak wave period (30m)
57. Peak wave period (10m)
58. Peak wave length (30m)
59. Peak wave length (10m)
60. Total energy in frequency intervals (30m)
61. Total energy in frequency intervals (10m)
62. Relative energy in frequency intervals (30m)
63. Relative energy in frequency intervals (10m)

#### Directional information

Quantities are not re-computed, but taken from the 30m CDIP data products.
Always take the closest data point (in time), so the maximum offset is 15m.

64. Sampling time for directional quantities
65. Dominant directional spread in frequency intervals
66. Dominant wave mean direction in frequency intervals
67. Peak wave direction

Dominant quantities consist of frequency average weighted by spectral energy density.


## Quality control

Loosely after Christou and Ewans (2014):

- a. Individual waves with a zero-crossing wave period >25s.
- b. The rate of change of surface elevation exceeds the limit rate of change by a factor of 2.
- c. Ten consecutive data points of the same value.
- d. Any absolute crest or trough elevation is greater than 8 times the normalized median absolute deviation of the surface elevation.
- e. Surface elevations are not equally spaced in time (but they may contain missing data).
- f. The ratio of missing to valid data exceeds 5%.
- g. Less than 100 individual recorded zero-crossings.

Waves where the sea state over the last 30m does not pass QC are excluded from the final dataset.
