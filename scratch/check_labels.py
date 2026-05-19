import pandas as pd
df = pd.read_csv('c:/Users/90530/Desktop/barış/dataset/merged.csv', usecols=['Normal/Attack'])
print(df['Normal/Attack'].unique())
