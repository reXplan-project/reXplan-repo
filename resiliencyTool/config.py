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
        self.globalDatabase = 'database.csv'


    def networkFile(self, simulationName):
        return os.path.join(self.inputFolder, simulationName, self.network)
    def outputFolderPath(self, simulationName):
        checkPath(os.path.join(self.outputFolder, simulationName))
        return os.path.join(self.outputFolder, simulationName) 
    def globalDatabaseFile(self, simulationName):
        checkPath(os.path.join(self.outputFolder, simulationName))
        return os.path.join(self.outputFolder, simulationName, self.globalDatabase)
    

path = Path()
inputFieldsMap =  pd.read_csv(path.inputFieldsMapFile, index_col = 'input_file_field')
