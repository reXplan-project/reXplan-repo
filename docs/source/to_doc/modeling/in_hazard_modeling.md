# Hazard Modeling

The hazard needs to be modeled to generate the probability of failure of the power elements with the use of fragility curves. Depending on the provided dataset, the hazard can be defined in four different ways.

## Method 1: Generate a hazard element by reading a .nc file

The most precise method is using data from a [netCDF](https://www.unidata.ucar.edu/software/netcdf/) file. NetCDF is a standard file format for scientific climate data storage and exchange, which allows to store multidimensional data with self-describing metadata. Once imported, the tool can analyse the data to identify the highest intensity at each point of the geographical grid. By mapping the power elements against this grid, reXplan can generate probabilities of failure for each element based on the intensity of the climate data and the equipment fragility curves. This 
approach allows for a comprehensive assessment of the potential impacts of climate-related hazards on power equipment, providing insights into which elements are most at risk and where mitigation efforts should be focused. 

## Method 2: Generate a hazard element by reading a trajectory .csv file

If the hazard has a trajectory like Hurricanes, intensity can be described using a .csv file containing the time-series trajectory and radius of the event. The hazard model consists of a circle of varying radius with the point of maximum intensity at its centre. The intensity decreases as you move away from the centre. The centre of the circle also called the epicentre can be moved along geographical coordinates defined by the user, allowing for a more flexible and simple approach to hazard modelling. This method offers a way to visualize the spatial distribution of the hazard intensity over time, which can be useful in assessing the potential impact of the hazard on a specific location or region. By incorporating user-defined geographical coordinates, this approach can be tailored to specific events and locations, allowing for a more accurate and relevant analysis of the hazard.

```{figure} ../../_static/trajectory.gif
:name: equation-mensah
:width: 400px

#TODO Get current gif for sphinx doc; Scale for Windspeed!
```

## Method 3: Generate a static hazard element by providing epicenter data

For simpler analysis or hazards with static geolocation like earthquakes, a constant value of the hazard intensity may be utilized. In this scenario, the same level of intensity for the extreme event is applied to each equipment’s fragility curve to generate failure probabilities.

#TODO Difference to Method 1? Fileformat?

## Method 4: Simulate multiple events according a given return period

Another common way ([see an example from hydro studies](https://hydro-informatics.com/exercises/ex-floods.html#terminology)) in which hazard events are defined is by their return period. This method involves estimating the average time between events such as earthquakes, floods, landslides, or river discharge flows. 

Different return periods can be allocated to certain areas of interest. A great source for return period data is [the Worldbank](https://climateknowledgeportal.worldbank.org/country/germany/extremes) which provides data for extreme events.
 
Unlike other input methods in reXplan, which rely on a specific evolution of the hazard intensity, the return period method uses a stratified probability density function - derived from the return period - to sample from a wide range of intensities. ReXplan optimizes the sampling process through stratification, resulting in accurate Monte Carlo analysis with fewer data samples. This stratification approach employs the R library [SamplingStrata](https://CRAN.R-project.org/package=SamplingStrata) as its backend. A notable advantage of this approach is its ability to find optimal number of strata and the optimal stratification boundaries for multivariate problems or multiple fragility curves, which is particularly useful in addressing resiliency problems.

```{figure} ../../_static/return_period.png
:name: return_period
:width: 500px

Figure from [Basic Example](../gettingstarted/basic_example_sphinx.ipynb).
```