# examplesI

Link to a doc like this {doc}`../gettingstarted/installation`.
Link to a header like this [ref](../gettingstarted/installation.md).

Docstring under "reXplan.simulation.Sim.initialize_model_rp"

		# DEPRECATED
		'''
		# Create Unique list fo combinations of fragility curves and return periods
		fc_rp_list = []
		for _, value in network.powerElements.items():
			if value.fragilityCurve != None and value.return_period != None:
				temp = [value.fragilityCurve, value.return_period]
				if temp not in fc_rp_list:
					fc_rp_list.append(temp)

		# Calculate default bounderies for the event intensity
		for j, fc_rp in enumerate(fc_rp_list):
			xp = network.fragilityCurves[fc_rp[0]].projected_intensity(network.returnPeriods[fc_rp[1]], 
																		network.returnPeriods[ref_return_period],
																		network.fragilityCurves[fc_rp[0]].x_data)
			if j == 0:
				xmin = min(xp)
				xmax =max(xp)
			else:
				xmin = min(xmin, min(xp))
				xmax = max(xmax, max(xp))
		'''

		# Calculate default bounderies for the event intensity

Big Test

```{eval-rst}
.. automodule:: reXplan.network
	:members:
	:noindex:
```