# Welcome to the reXplan repository!

**Simulating Power System Resilience**

ReXplan is a python library that evaluates the resiliency of power grids to extreme events, enabling a better decision making process for the infrastructure of power grids. <br>
Created by the R&D-Team of the Energy System Consulting Department of Tractebel Engineering GmbH.

<img src="./docs/source/_static/ENGIE_tractebel_solid_BLUE_RGB_300.png" alt="tractebel_logo" width="200"/>

Please see below for the installation, documentation and more.

## Installation Guide
### Step 1: Prerequisites
1.1 **Python 3.10** [(Download, win x64)](https://www.python.org/ftp/python/3.10.10/python-3.10.10-amd64.exe)\
1.2 **Julia 1.8.5** [(Download, win x64)](https://julialang-s3.julialang.org/bin/winnt/x64/1.8/julia-1.8.5-win64.exe)\
1.3 **R 4.2.2** [(Download, win x64)](https://ftp.fau.de/cran/bin/windows/base/old/4.2.2)\
1.4 **GitHub Repository** [(here)](https://github.com/FirasJrad/reXplan)

### Step 2: Imports and Module Management Julia & R
2.1 Import and Install Julia **through Python Shell**:
```
import julia
julia.install()
```

2.2 Add Julia Packages **through Julia Shell**, using the integrated Package Manager, which is accessable using " `]` " _(AltGr + 9)_:
```
add PowerModules@0.19.8
add PandaModules@0.7.1
```

2.3 Add R Package **through R Shell**:
```
install.packages("SamplingStrata")
```

### Step 3: Creating a Virtual Environment

3.1 Go to repository. Either use Windows Command Prompt (CMD)...
```
cd ..\reXplan
```

...or VSC Terminal with Command Prompt Profile (recommended).

---

3.2 Create and activate Environment:
```
py -3.10 -m venv venv
venv\Scripts\activate.bat
```

3.3 Upgrade Pip and install Packages & Dependencies:
```
py -m pip install --upgrade pip
pip install .
```

## Documentation
In development. Contact tim.hoffmann@tractebel.engie.com for further information.

## Contribute
[Become a Contributor on GitHub!](https://github.com/FirasJrad/reXplan)

## License
tba