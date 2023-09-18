
# Functions for Users

## Network

```{eval-rst}
.. autofunction:: reXplan.network.Network
```

## Simulation

```{eval-rst}
.. autofunction:: reXplan.simulation.Sim
```

### Initialize Functions

For explainatory text see [Initialize & Run](../modeling/mid_init_n_run.md#initialize-functions---asset-based-kpis).

#### Methodology Return Periods

The random samples of the hazard intensity for the methodology of return periods are selected from the `ref_return_period`.


```{eval-rst}
.. autofunction:: reXplan.simulation.Sim.initialize_model_rp
```


#### Methodology Single Hazard

```{eval-rst}
.. autofunction:: reXplan.simulation.Sim.initialize_model_sh
```

### Run

Running the optimal power flow -> cost analysis, load flow result

```{eval-rst}
.. autofunction:: reXplan.simulation.Sim.run
```

## Hazard

```{eval-rst}
.. autofunction:: reXplan.hazard.Hazard.hazardFromNC
```

```{eval-rst}
.. autofunction:: reXplan.hazard.Hazard.hazardFromTrajectory
```

```{eval-rst}
.. autofunction:: reXplan.hazard.Hazard.hazardFromStaticInput
```

