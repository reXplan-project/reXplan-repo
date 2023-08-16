# Hazard Modeling

## Methodologies

Depending on the provided dataset, the user can choose a methodology.

### Method 1: Generate a hazard element by reading a .nc file

The most precise method is using data from a [netCDF](https://www.unidata.ucar.edu/software/netcdf/) file. NetCDF is a standard file format for scientific data storage and exchange, which allows to store multidimensional data with self-describing metadata.

### Method 2: Generate a hazard element by reading a trajectory .csv file

If the hazard has a trajectory like hurricanes, a .csv file can be supplied. Data can be easily changed manually.

```{figure} ../../_static/trajectory.gif
:name: equation-mensah
:width: 400px

#TODO Get current gif for sphinx doc; Scale for Windspeed!
```

### Method 3: Generate a static hazard element by providing epicenter data
For hazards with a specified perimeter and epicenter like earthquakes. #TODO Difference to Method 1?

### Method 4: Simulate multiple events according a given return period

A return period is an important tool for risk assessment and probability analysis. It refers to the average time between events with a defined intensity. The occurrence of a "100-year strom" has the same probability for each year, which is 1/100. Visit [hydro-informatics.com](https://hydro-informatics.com/exercises/ex-floods.html#terminology) for more information on return periods.

Different return periods can be allocated to certain areas of interest. A great source for return period data is [the Worldbank](https://climateknowledgeportal.worldbank.org/country/germany/extremes) which provides data for extreme events.

```{figure} ../../_static/return_period.png
:name: return_period
:width: 800px

Figure from [Basic Example](../gettingstarted/basic_example_sphinx.ipynb).
```

## Initialize Functions

The initialize model functions {py:func}`resiliencyTool.simulation.Sim.initialize_model_rp` and
{py:func}`resiliencyTool.simulation.Sim.initialize_model_sh` **create an outage schedule for each electrical equipment in the network as a database with timeseries**. It contains the operating information of all elements in the grid (“in service”, “out of service”, “awaiting repair”, “under repair”), which will be **used for the Monte Carlo simulation**.

---

In a **standard Monte Carlo simulation**, random samples of the hazard **intensity are selected from the entire range** of expected hazards. Since higher intensity events are less common, there will be a smaller representation of those events among the Monte Carlo simulation. To achieve better accuracy a large quantity of iterations must be selected. This is not always possible for bigger networks as it quickly becomes **computationally expensive**.

To counter this problem, a **stratification** method is used to divide the total range of possible intensities from the hazard into stratas with the objective being to minimize the total variance of the Monte Carlo samples within each strata. A separate Monte Carlo simulation can then be performed for each of the defined stratas with the number of samples as iterations. This method can achieve a similar accuracy compared to the Monte Carlo simulation without stratification with significantly fewer samples and thus **reducing the computational cost**.

Each Monte Carlo iteration will select a random intensity value within the strata intensity boundaries. The probability of failure of each equipment type is then determined using the fragility curves defined for each type of electrical equipment. A random probability value is then selected for each electrical equipment and is compared to the probability of failure to determine which elements will fail in that specific Monte Carlo iteration.

The time of failure is also selected randomly during the hazard period to avoid having all elements marked for failure, failing at the exact same time.

Using a queue system with priorities for different elements of the network that can be defined by the user, a repair schedule is prepared. The repair schedule considers the number of crew available as well as the provided repair times for each of the electrical equipment.

The equipment failure schedule in combination with the repair schedule form the `outage schedule`. One outage schedule is generated for each Monte Carlo iteration and are all stored in the `montecarlo_database.csv` file.