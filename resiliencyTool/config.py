import os
import pandas as pd

def checkPath(path):
    if not os.path.exists(path):
        os.mkdir(path)

class Path():
    def __init__(self):

        parent = 'file'
        testFolder = os.path.join(parent,'test')
        toolFolder = 'resiliencyTool'

        self.inputFolder = os.path.join(parent,'input')
        self.outputFolder = os.path.join(parent,'output')
        self.inputFieldsMapFile = os.path.join(toolFolder, 'fields_map.csv')

        checkPath(self.outputFolder)
        
        self.network = 'network.xlsx'
        self.metricDatabase = 'metric_database.csv'
        self.montecarloDatabase = 'montecarlo_database.csv'
        self.engineDatabase = 'engine_database.csv'

    def networkFile(self, simulationName):
        return os.path.join(self.inputFolder, simulationName, self.network)
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

