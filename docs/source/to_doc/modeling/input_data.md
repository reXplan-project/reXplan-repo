# (I/0) - Input Data

The [basic example](../gettingstarted/basic_example_sphinx.ipynb) can be accessed via path

```
path: ../reXplan/jupyter_notebooks/
```

as `basic_example.ipynb`. If a new project is created, the .ipynb file should be kept in this directory.
The input data, as explained in the basic example, is located at the following directory:

```
path: ../reXplan/jupyter_notebooks/file/input/[simulationName]
# in the case of basic example, the input data of 'basic_example_tim' is used:
# ->    ../reXplan/jupyter_notebooks/file/input/basic_example_tim
```
If inside the directory, the following folders and file should show:

- 📁 **fragilityCurves**
- 📁 **hazards** (depending on project)
- 📁 **returnPeriods** (depending on project)
- 📗 **network.xlsx**

If new .ipynb file is created, the according files and folders need to be created manually. These can be copied from other projects and changed to desired needs. Depending on the Simulation method, either the *hazards* or *returnPeriods* folder needs to exist with according datasets.<br>
As previously described, the necessary datasets for reXplan are [Grid Modeling](in_grid_modeling.md), [Fragility Curves](in_fragility_curve.md) and [Hazard Modeling](in_hazard_modeling.md).