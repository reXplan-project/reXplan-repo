# Sphinx Setup instructions

### Step 1: Install Pandoc

**pandoc** [(Download, win x64)](https://www.python.org/ftp/python/3.10.10/python-3.10.10-amd64.exe)\

### Step 2: Install Sphinx

```
pip install -e.[doc]
```

### Step 3: Use Sphinx
```
sphinx-autobuild docs/source docs/build/html  
```

SEE GITBOOK FOR SETUP PROCESS