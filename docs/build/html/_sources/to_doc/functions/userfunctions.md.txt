
# Functions for Users

## Network

```{eval-rst}
.. autoclass:: reXplan.network.Network
    :noindex:
```

## Simulation

```{eval-rst}
.. autoclass:: reXplan.simulation.Sim
    :noindex:
```

### Initialize Functions

For explainatory text see [Initialize & Run](../modeling/mid_init_n_run.md#initialize-functions---asset-based-kpis).

#### Methodology Return Periods

The random samples of the hazard intensity for the methodology of return periods are selected from the `ref_return_period`.


```{eval-rst}
.. autofunction:: reXplan.simulation.Sim.initialize_model_rp
    :noindex:
```


#### Methodology Single Hazard

```{eval-rst}
.. autofunction:: reXplan.simulation.Sim.initialize_model_sh
    :noindex:
```

### Run

Running the optimal power flow -> cost analysis, load flow result

```{eval-rst}
.. autofunction:: reXplan.simulation.Sim.run
    :noindex:
```

## Hazard

```{eval-rst}
.. autofunction:: reXplan.hazard.Hazard.hazardFromNC
    :noindex:
```

```{eval-rst}
.. autofunction:: reXplan.hazard.Hazard.hazardFromTrajectory
    :noindex:
```

```{eval-rst}
.. autofunction:: reXplan.hazard.Hazard.hazardFromStaticInput
    :noindex:
```

