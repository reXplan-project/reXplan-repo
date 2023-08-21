# Fragility Curve

A fragility curve is a common tool used in [hydrology](https://wntr.readthedocs.io/en/stable/fragility.html) and seismic studies for risk assessment. It represents graphically how vulnerable a system or structure is to a particular hazard. Fragility curves are developed through data analysis of historical events, simulation and statistical methods. It provides valuable information for risk assessment, loss estimation, and decision-making processes related to hazard mitigation and emergency preparedness.

In the context of power grids, fragility curves are employed to measure the resiliency of grid elements, like bus stations, lines or generators. The measured intensity can be defined for wind speeds, ground acceleration, water levels, etc. in theier respective unit system.

```{important} 
Type and unit system of intensity of fragility curve need to match the hazard definition.
```


## Finding Data
 
Fragility curve data, which provides crucial insights into the vulnerability of structures like wind turbines under varying conditions, is often not readily accessible due to the reluctance of companies to release their proprietary information.

Therefore, finding suitable data is an important task for the modeling of resilience. In the paper [Resilience Assessment of Electric Grids and Distributed Wind Generation under Hurricane Hazards](https://docslib.org/doc/4598583/resilience-assessment-of-electric-grids-and-distributed-wind-generation-under-hurricane-hazards) an equation can be found for the resiliency of transmission support towers. This can be used to model different types of towers.

```{figure} ../../_static/Equation_Mensah.png
:name: equation-mensah
:width: 800px

[Mensah Equation](https://docslib.org/doc/4598583/resilience-assessment-of-electric-grids-and-distributed-wind-generation-under-hurricane-hazards) #TODO write in Latex
```

With generated datasets from this equation, reXplan interpolates the user definded range to create continous data. The dataset gets build with 

```
network = rx.network.Network(simulationName);
```
and can be accessed with 

```
xnew = np.linspace(0, 100, num=100, endpoint=True)
fig, ax = rx.fragilitycurve.plotFragilityCurves(network.fragilityCurves, xnew)
```
as shown in the [Basic Example](../gettingstarted/basic_example_sphinx.ipynb):

```{figure} ../../_static/TowersWithFrag.png
:name: figure-towers

Different types of towers equal different fragility curves. Intensity in m/s windspeed.
```