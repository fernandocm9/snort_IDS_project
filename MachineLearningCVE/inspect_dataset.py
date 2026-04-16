import pandas as pd


df = pd.read_csv("Wednesday-workingHours.pcap_ISCX.csv")

#remove spaces in front of column labels
df.columns = df.columns.str.strip()


print(df.head())
print(df.columns.tolist())
print(df["Label"].value_counts())
