# RnD_Resiliency
A research project to create a tool that evaluates the resiliency of power grids to extreme events.

## Installation:
In the source folder using pip: pip install -e

### Modify pandapower:
Go to pandapower's installation folder (in windows, ~\Envs\test_resiliency\lib\site-packages\pandapower, where ~ is your user folder)
Modify auxiliary.py, line 543 
is_elements["line_is_idx"] = net["line"].index[net["line"].in_service.values]
by
is_elements["line_is_idx"] = net["line"].index[net["line"].in_service.values.astype(bool)]

## Importing
On Python: import resiliencyTool
