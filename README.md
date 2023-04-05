# reXplan
A research project to create a tool that evaluates the resiliency of power grids to extreme events.

## Installation
run the following code on cmd inside the reXplan directory to create and activate a new virtual environment:
```
python -m venv venv
venv\Scripts\activate.bat
```
Then run the following to install all the necessary python packages
```
pip install -r requirements.txt
```
Finally run the following to install reXplan 
```
pip install -e .
```

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

# Needed Julia packages
After installing Julia (latest version tested was 1.8.5), in the python shell type the following commands:
```
import julia
julia.install()
```

Then in the Julia shell, go to the package manager using `]` and run the following commands
```
add PowerModules@0.19.8
add PandaModules@0.7.1
```

# Needed R packages
After installing R (latest tested version was 4.2.2 (2022-10-31 ucrt)) run the following command in the R shell
```
install.packages("SamplingStrata")
```
The tested version for the SamplingStrata package is 1.5-4.

## Importing
Python: `import resiliencyTool`
