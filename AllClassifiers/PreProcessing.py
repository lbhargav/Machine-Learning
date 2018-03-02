import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# Load dataset
dataset = pd.read_csv(sys.argv[1])
#writing outputfile
output_file=sys.argv[2]
print(dataset.head(5))

#drop rows with missing values
dataset.dropna(inplace=True)
#useful for testing print(dataset.isnull().any())
#split data into 2 matrixes for standardization & conversion
x = dataset.ix[:,2:31].values
y = dataset.ix[:,1].values

#Standardization
scaler = preprocessing.StandardScaler().fit(x)
scaler.mean_
scaler.scale_
X_scaled =  scaler.transform(x).round(4)

#convert category to numerical values
le=LabelEncoder()
le.fit(y)
class_scaled=le.transform(y).round(0)

#merge into single matrix
final_dataset = (np.c_[X_scaled, class_scaled])

#write as file output
pd.DataFrame(final_dataset).to_csv(sys.argv[2], index=False, header=False)