from sklearn.preprocessing import LabelEncoder
import pandas as pd



file_path = r'C:\Users\Mathias\Documents\Programmieren\GenBoost\ObesityDataSet_raw_and_data_sinthetic.csv'
df = pd.read_csv(file_path)

print(df.head())

#many features are encoded as strings
#so we turn them to numerics


encoder  =LabelEncoder()
for i in ['Gender','FAVC','SCC', 'SMOKE', 'family_history_with_overweight']:
    df[i] =encoder.fit_transform(df[i])


#some features have a clear hirarchy
#labelEncoder works alphabetically
# so we map it "by hand"
mapping_label = {
    "Insufficient_Weight": 0,
    "Normal_Weight": 1,
    "Overweight_Level_I": 2,
    "Overweight_Level_II": 3,
    "Obesity_Type_I": 4,
    "Obesity_Type_II": 5,
    "Obesity_Type_III": 6
}


df['NObeyesdad'] = df['NObeyesdad'].map(mapping_label)

mapping_calc = {
    "no": 0,
    "Sometimes": 1,
    "Frequently": 2,
    "Always": 3
}

df['CALC'] = df['CALC'].map(mapping_calc)

mapping_caec = {
    "no": 0,
    "Sometimes": 1,
    "Frequently": 2,
    "Always": 3
}

df['CAEC'] = df['CAEC'].map(mapping_caec)

mapping_MTRANS = {
    "Bike": 0,
    "Walking": 1,
    "Public_Transportation": 2,
    "Motorbike": 3,
    "Automobile": 4,
}

df['MTRANS'] = df['MTRANS'].map(mapping_MTRANS)

print(df.head(100))

#randomize the data
df = df.sample(frac=1).reset_index(drop=True)

x = df.to_csv('FinalTrain.csv', index=False)


#setting weight to zero

zero = df
zero['Weight']= 0

print(zero.head())

y = df.to_csv('ZeroFinalTrain.csv', index=False)