import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="resiliencyTool",
    version="0.0.1",
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
)