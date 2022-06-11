import resiliencyTool as rt
import pandapower as pp
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import datetime


network = rt.Network(1,'spain','testNetwork4nodes')

print(network.pp_network)
#print(network.bus)
#print(network.line)
#print(network.load)

#tic = time.time()
#pp.rundcopp(network, delta=1e-16)
#pp.runopp(network, delta=1e-16)
#pp.runpp(network, delta=1e-16)
#tac = time.time()
#print(f"Duration of runopp = {tac-tic}")
#print(network.res_ext_grid)
#print(network.res_gen)

fc = rt.FragilityCurve('fc1.csv')

#print(fc.name)
#fc.plot()
#plt.show()

h1 = rt.Hazard('hazard1', 'hazard\\LKA.nc')
start = pd.to_datetime(datetime.datetime(2020, 4, 20))
end = pd.to_datetime(datetime.datetime(2040, 4, 20))
t, pr = h1.read_attribute('pr', 80, 6, start, end)
print(h1.get_lon())
print(h1.get_lat())
plt.plot(t,pr)
#h1.plot()
plt.show()




#import netCDF4 as nc
#ncdf = nc.Dataset(r'D:\\Documents\\R&D\\RnD_Resiliency\\Test Scripts\\LKA.nc', mode='r')
#lons = ncdf.variables['lon']
#lats = ncdf.variables['lat']
#weather_elements = ncdf.variables['pr']
#print(weather_elements)
#time_values = ncdf.variables['time']
#var = np.where(time_values)[0]
#dtime = nc.num2date(time_values[:],time_values.units)

#print(var)
#ncdf.close()