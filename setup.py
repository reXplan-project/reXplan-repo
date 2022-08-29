import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="resiliencyTool",
    version="0.0.2",
    packages=['resiliencyTool'],
    zip_safe=False,
    author="ENGIE impact",
    author_email="devteam@engie.com",
    description="This modules provides tool to evaluate the resiliency of electrical grids against extreme events",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/pypa/sampleproject",
    #project_urls={
    #    "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    #},
    #classifiers=[
    #    "Programming Language :: Python :: 3",
    #    "License :: OSI Approved :: MIT License",
    #    "Operating System :: OS Independent",
    #],
    #package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=[
        'dash>=2.6.0',
        # 'matplotlib>=3.5.2',
        'netCDF4>=1.6.0',
        'networkx>=2.8.4',
        'numba>=0.55.2',
        'numpy>=1.22.4',
        # 'pandapower>=2.9.0',
        'pandas>=1.4.3',
        'plotly>=5.9.0',
        # 'nbformat>=4.2.0',
        'basemap>=1.3.3',
        'openpyxl>=3.0.10'

    ]
)
# pip install numba --upgrade --user
# pip install pandas = 1.2.3