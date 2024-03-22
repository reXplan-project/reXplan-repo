# Installation Guide

```{note}
Succesfully tested setup on Windows 11.
```

### Step 1: Install Prerequisites

```{important}
Note during setup: set a PATH variable with Python, Julia and R!
```

1.1 **Python 3.10** [(Download, win x64)](https://www.python.org/ftp/python/3.10.10/python-3.10.10-amd64.exe)\
1.2 **Julia 1.8.5** [(Download, win x64)](https://julialang-s3.julialang.org/bin/winnt/x64/1.8/julia-1.8.5-win64.exe)\
1.3 **R 4.2.2** [(Download, win x64)](https://ftp.fau.de/cran/bin/windows/base/old/4.2.2/R-4.2.2-win.exe)\
1.4 **Git** [(Download, win x64)](https://github.com/git-for-windows/git/releases/download/v2.43.0.windows.1/Git-2.43.0-64-bit.exe)\
1.5 **GitHub Repository** [(here)](https://github.com/Tractebel-Engineering/reXplan-repo)

### Step 2: Imports and Module Management Julia & R

2.1 Add Julia Packages **through Julia Shell**, using the integrated Package Manager, which is accessable using " `]` " _(AltGr + 9)_:
```
add PowerModels@0.19.8
add PandaModels@0.7.1
```

2.2 Add R Package **through R Shell**:
```
install.packages("SamplingStrata")
```

### Step 2: Creating a Virtual Environment

3.1 Go to the repository. Either use Windows Command Prompt (CMD)...
```
cd ..\reXplan
```

...or VSC terminal with command prompt profile (recommended).

---

3.2 Create and activate Environment:
```
py -3.10 -m venv venv
venv\Scripts\activate.bat
```

3.3 Upgrade pip and install packages & dependencies:
```
py -m pip install --upgrade pip
pip install -e . # For developers
pip install . # For users
```

3.4 Install Julia via cmd/VSC Terminal
```
py
import julia
julia.install()
```

## Documentation
In development. Contact tim.hoffmann@tractebel.engie.com for further information.

## Contribute
[Become a contributor on GitHub!](https://github.com/Tractebel-Engineering/reXplan-repo)

## License
tba
