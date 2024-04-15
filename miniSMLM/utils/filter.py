import pandas as pd
import numpy as np

class Filter:
    def __init__(self, spots):
        self.spots = spots

    def filter(self):
        self.spots = self.spots.loc[(self.spots['x_mle'] > 0 ) & (self.spots['y_mle'] > 0) & (self.spots['conv'] == True) & (self.spots['N0'] > 10)]
        self.spots = self.spots.loc[(self.spots['x_mle'] < 300) & (self.spots['y_mle'] < 300) & (self.spots['N0'] < 10000)]
        coords = self.spots[['x_mle','y_mle']].values

        x = coords[:, 1]
        x = self.findOutliers(x)
        self.spots.drop(x, inplace = True)

        y = coords[:, 0]
        y = self.findOutliers(y); 
        self.spots.drop(y, inplace = True)

        return self.spots

    def findOutliers(self, data):
        outlierIndices = []
        q1 = np.quantile(data, 0.25)
        q3 = np.quantile(data, 0.75)
        iqr = q3 - q1
        for i in range(len(data)):
            num = data[i]
            if ((num < q1 - (1.5*iqr)) | (num > (q3 + 1.5*iqr))):
                outlierIndices.append(i)
        
        print(outlierIndices)

        return outlierIndices
        

        
