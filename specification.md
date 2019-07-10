# A comprehensive wave dataset

## Data sources

Our primary data source will be the CDIP buoys:

- 151 buoys
- XYZ displacement at measurement frequency 1.28Hz
- Total data size 518GB containing more than 1e10 waves
- Most buoys in coastal regions, some in deep water (how many?)

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

### Dimensions

- Station name (scalar)
- Local wave ID
- Frequency bracket
- Sample time (only used for raw elevation, fixed to 32 values)

### Variables

By "wave" we mean the series of surface elevations (relative to the 30 minute mean elevation) from any given zero downcrossing to the next zero downcrossing (some waves might be excluded due to quality control criteria, see below).

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
4. Global wave ID (hash computed from source file MD5, start time, end time, processing git hash)
5. Local wave ID (unique for each station)
6. Wave start time
7. Wave end time
8. Zero-crossing period
9. Zero-crossing wavelength
10. Raw elevation over 30m SSH (2D variable)
11. Crest height over 30m SSH
12. Trough depth below 30m SSH
13. Wave height
14. Maximum elevation slope
15. Buoy latitude
16. Buoy longitude
17. Total water depth
18. Sampling rate

#### Aggregates

Quantities are computed directly from the raw displacements. All time averages stop at wave start time.

19. Aggregate start
20. Aggregate end
21. Significant wave height Hm0 (30m)
22. Significant wave height Hm0 (10m)
23. Significant wave height H1/3 (30m)
24. Significant wave height H1/3 (10m)
25. Sea surface height (30m)
26. Sea surface height (10m)
27. Skewness (30m)
28. Skewness (10m)
29. Excess kurtosis (30m)
30. Excess kurtosis (10m)
31. Spectral bandwidth (30m)
32. Spectral bandwidth (10m)
33. Characteristic wave steepness (30m)
34. Characteristic wave steepness (10m)
35. Mean zero-crossing period (30m)
36. Mean zero-crossing period (10m)
37. Mean spectral period (30m)
38. Mean spectral period (10m)
39. Benjamin-Feir index (30m)
40. Benjamin-Feir index (10m)
41. Valid data ratio (30m)
42. Valid data ratio (10m)
43. Peak wave period (30m)
44. Peak wave period (10m)
45. Peak wave length (30m)
46. Peak wave length (10m)
47. Total energy in frequency interval 1 (30m)
48. Total energy in frequency interval 1 (10m)
49. Total energy in frequency interval 2 (30m)
50. Total energy in frequency interval 2 (10m)
51. Total energy in frequency interval 3 (30m)
52. Total energy in frequency interval 3 (10m)
53. Total energy in frequency interval 4 (30m)
54. Total energy in frequency interval 4 (10m)
55. Total energy in frequency interval 5 (30m)
56. Total energy in frequency interval 5 (10m)

#### Directional information

Quantities are not re-computed, but taken from the 30m CDIP data products.
Always take the closest data point (in time), so the maximum offset is 15m.

57. Sampling time for directional quantities
58. Dominant directional spread at frequency interval 1
59. Dominant directional spread at frequency interval 2
60. Dominant directional spread at frequency interval 3
61. Dominant directional spread at frequency interval 4
62. Dominant directional spread at frequency interval 5
63. Dominant wave mean direction at frequency interval 1
64. Dominant wave mean direction at frequency interval 2
65. Dominant wave mean direction at frequency interval 3
66. Dominant wave mean direction at frequency interval 4
67. Dominant wave mean direction at frequency interval 5
68. Peak wave direction

Dominant quantities consist of frequency average weighted by spectral energy density.


## Quality control

Loosely after Christou and Ewans (2014):

- a. Individual waves with a zero-crossing wave period >25s.
- b. The rate of change of surface elevation exceeds the limit rate of change by a factor of 2.
- c. Ten consecutive data points of the same value.
- d. Any absolute crest or trough elevation is greater than 8 times the standard deviation of the 30-min water surface elevation.
- e. Surface elevations are not equally spaced in time (but they may contain missing data).
- f. The ratio of missing to valid data exceeds 5%.
- g. Less than 100 individual recorded zero-crossings.

Waves where the sea state over the last 30m does not pass QC are excluded from the final dataset.
