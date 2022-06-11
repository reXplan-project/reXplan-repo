import math

def getDistanceFromLatLonInKm(lat1,lon1,lat2,lon2):
    # Radius of the earth in km
    R = 6371; 
    # deg2rad below
    dLat = deg2rad(lat2-lat1)  
    dLon = deg2rad(lon2-lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)) 
    # Distance in km
    d = R * c
    return d

def deg2rad(deg):
    return deg*(math.pi/180)

lat1 = 50.200387
lon1 = 8.518401
lat2 = 50.204676
lon2 = 8.504855

print(getDistanceFromLatLonInKm(lat1,lon1,lat2,lon2))