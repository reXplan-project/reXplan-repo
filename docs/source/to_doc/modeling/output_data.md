# (I/0) - Output Data

```
path: ../reXplan/jupyter_notebooks/file/output/[simulationName]
```
- 🔰 **montecarlo_database.csv**, created by [`initialize function`](../functions/userfunctions.md#initialize-functions)
- 🔰 **engine_database.csv**, created by [``run function``](../functions/userfunctions.md#run)

[#TODO Set Links, introductory text]



In this section of the paper, some of the resiliency analysis results are described, together with the relevant KPIs.
It is convenient to think about the results of the analysis, divided in the following two categories of results:
• Asset-based results
• OPF-based results
The asset-based results are efficiently calculated before running OPF, so that computational effort is reduced at 
minimum. Asset based results can provide a first impression of the resiliency of the grid in terms of available asset, for example showing the number of line in service at any given time, like in diagram of Fig. 4. Here, statistical analysis, is used to summarize quantiles over the raw data of the MC analysis.

