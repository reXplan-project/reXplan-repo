import os

def checkPath(path):
    if not os.path.exists(path):
        os.mkdir(path)

class Path():
    def __init__(self):
        parent = 'file'
        testFolder = os.path.join(parent,'test')
        self.inputFolder = os.path.join(parent,'input')
        self.outputFolder = os.path.join(parent,'output')
        checkPath(self.outputFolder)
        self.network = 'network.xlsx'
        self.globalDatabase = 'database.csv'
        
    def networkFile(self, simulationName):
        return os.path.join(self.inputFolder, simulationName, self.network)
    def globalDatabaseFile(self, simulationName):
        checkPath(os.path.join(self.outputFolder, simulationName))
        return os.path.join(self.outputFolder, simulationName, self.globalDatabase)
    

path = Path()