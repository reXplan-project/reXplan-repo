import os
import pandas as pd

def checkPath(path):
    if not os.path.exists(path):
        os.mkdir(path)

class Path():
    def __init__(self):
        # parent_ = os.path.join(parent, 'file')
        parent_ = 'file'
        # testFolder = os.path.join(parent_,'test')
        toolFolder = os.path.abspath(os.path.dirname(__file__ )) # aboslute file path of config.py
        self.inputFolder = os.path.join(parent_,'input')
        self.outputFolder = os.path.join(parent_,'output')
        self.inputFieldsMapFile = os.path.join(toolFolder, 'fields_map.csv')

        checkPath(self.outputFolder)
        
        self.network = 'network.xlsx'
        self.metricDatabase = 'metric_database.csv'
        self.montecarloDatabase = 'montecarlo_database.csv'
        self.engineDatabase = 'engine_database.csv'
        self.fcDatabase = 'fragilityCurves'
        self.hazardFolder = 'hazards'
        self.hazardGifFolder = 'gif'
        #self.hazard = 'trajectory.csv'

    def networkFile(self, simulationName):
        return os.path.join(self.inputFolder, simulationName, self.network)
    def fragilityCurveFolder(self, simulationName):
        return os.path.join(self.inputFolder, simulationName, self.fcDatabase)
    def hazardFile(self, simulationName, filename):
        return os.path.join(self.inputFolder, simulationName, self.hazardFolder, filename)
    def hazardGifFile(self, simulationName, filename):
        checkPath(os.path.join(self.inputFolder, simulationName, self.hazardFolder, self.hazardGifFolder))
        return os.path.join(self.inputFolder, simulationName, self.hazardFolder, self.hazardGifFolder, filename)
    def outputFolderPath(self, simulationName):
        checkPath(os.path.join(self.outputFolder, simulationName))
        return os.path.join(self.outputFolder, simulationName) 
    def metricDatabaseFile(self, simulationName):
        checkPath(os.path.join(self.outputFolder, simulationName))
        return os.path.join(self.outputFolder, simulationName, self.metricDatabase)
    def engineDatabaseFile(self, simulationName):
        checkPath(os.path.join(self.outputFolder, simulationName))
        return os.path.join(self.outputFolder, simulationName, self.engineDatabase)
    def montecarloDatabaseFile(self, simulationName):
        checkPath(os.path.join(self.outputFolder, simulationName))
        return os.path.join(self.outputFolder, simulationName, self.montecarloDatabase)

path = Path()

