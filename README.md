# Welcome to the reXplan repository!

**Simulating Power System Resilience**

ReXplan is a python library that evaluates the resiliency of power grids to extreme events, enabling a better decision making process for the infrastructure of power grids. <br>
Created by the R&D-Team of the Energy System Consulting Department of Tractebel Engineering GmbH.

<img src="./docs/source/_static/ENGIE_tractebel_solid_BLUE_RGB_300.png" alt="tractebel_logo" width="200"/>

Please see below for the installation, documentation and more.

## Installation Guide

> **Note**<br>
Succesfully tested setup on Windows 11.

### Step 1: Install Prerequisites

> **Important**<br>
Note during setup: set a PATH variable with Python, Julia and R!

1.1 **Python 3.10** [(Download, win x64)](https://www.python.org/ftp/python/3.10.10/python-3.10.10-amd64.exe)\
1.2 **Julia 1.8.5** [(Download, win x64)](https://julialang-s3.julialang.org/bin/winnt/x64/1.8/julia-1.8.5-win64.exe)\
1.3 **R 4.2.2** [(Download, win x64)](https://ftp.fau.de/cran/bin/windows/base/old/4.2.2/R-4.2.2-win.exe)\
1.4 **Git** [(Download, win x64)](https://github.com/git-for-windows/git/releases/download/v2.43.0.windows.1/Git-2.43.0-64-bit.exe)\
1.5 **GitHub Repository** [(here)](https://github.com/Tractebel-Engineering/reXplan-repo)

### Step 2: Imports and Module Management Julia & R


2.1 Add Julia Packages **through Julia Shell**, using the integrated Package Manager, which is accessable using " `]` " _(AltGr + 9)_:
```
>>> import julia
>>> julia.install()
```

2.2 Add Julia Packages **through Julia Shell**, using the integrated Package Manager:
```
julia> using Pkg
julia> ENV["HTTP_PROXY"]=”http://10.42.32.29:8080” # this if the Julia Proxi needs to be specified
julia> ENV["HTTPS_PROXY"]=”http://10.42.32.29:8080” # this if the Julia Proxi needs to be specified
julia> Pkg.add("PowerModels@0.19.8")
julia> Pkg.add("PandaModels@0.7.1")

```

2.3 Add R Package **through R Shell**:
```
> install.packages("SamplingStrata")
```

### Step 2: Creating a Virtual Environment

3.1 Go to the repository. Either use Windows Command Prompt (CMD)...
```
> cd ..\reXplan
```

...or VSC terminal with command prompt profile (recommended).

---

3.2 Create and activate Environment:

- If using venv:
```
> py -3.10 -m venv venv
> venv\Scripts\activate.bat
```

- If using virtualenvwrapper:
```
> mkvirtualenv --python = python3.10 reXplan
> workon reXplan
```

3.3 Upgrade Pip and install Packages & Dependencies:

```
> cd C:\path to reXplan-repo\ # this to change to the local directory of the reXplan repository
> py -m pip install --upgrade pip
> pip install .
```

## Documentation
In development. Contact reXplan@tractebel.engie.com for further information.

## Contribute
[Become a contributor on GitHub!](https://github.com/Tractebel-Engineering/reXplan-repo)

## License
tba
