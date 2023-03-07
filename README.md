# reXplan
A research project to create a tool that evaluates the resiliency of power grids to extreme events.

## Installation
With pip go to the source folder and run the following command: `pip install -e .`

<!-- ### Modify pandapower
Before using pandapower, a modification must be introduced in the source code.

Go to pandapower's installation folder *~\Envs\env_name\lib\site-packages\pandapower*, where ~ is the user folder and *env_name* is the name of the environment where the package was installed.

Replace line 543 in *auxiliary.py*:
```
is_elements["line_is_idx"] = net["line"].index[net["line"].in_service.values]
```
by
```
is_elements["line_is_idx"] = net["line"].index[net["line"].in_service.values.astype(bool)]
``` -->
## Importing
Python: `import resiliencyTool`
