# (I/0) - Output Data


It is convenient to think about the results of the analysis in two categories of results:

- **Asset-based results**
- **OPF-based results**

The **asset-based results are efficiently calculated before running OPF**, so that **computational effort is reduced** at minimum. Asset based results can provide a **first impression of the resiliency of the grid** in terms of available asset. 

Output data is created whenever functions {py:func}`reXplan.simulation.Sim.run`, {py:func}`reXplan.simulation.Sim.initialize_model_rp` or {py:func}`reXplan.simulation.Sim.initialize_model_sh` were executed successfully. Data is stored at path

```
-/reXplan/jupyter_notebooks/file/output/[simulationName]
```
as the following files:

- 📗 **montecarlo_database.csv**, created by [`initialize function`](../functions/userfunctions.md#initialize-functions) -> Asset-based
- 📗 **engine_database.csv**, created by [`run function`](../functions/userfunctions.md#run)
 -> OPF-based



