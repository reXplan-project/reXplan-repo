import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

from climada.hazard import StormEurope
from climada.util.constants import WS_DEMO_NC

storm_instance = StormEurope.from_footprints(WS_DEMO_NC)

# WISC_files = '/path/to/folder/C3S_WISC_FOOTPRINT_NETCDF_0100/fp_era[!er5]*_0.nc'
# storm_instance = StormEurope.from_footprints(WISC_files)