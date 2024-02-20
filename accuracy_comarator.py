import pandas as pd
import numpy as np

df1 = pd.read_csv("./cora_results/2024-02-04_21_27_40training_results.csv") # original performance
df2 = pd.read_csv("./cora_results/2024-02-20_15:33:17training_results.csv")

print(np.average(np.array(df1["Test AUC-PR"])))
print(np.average(np.array(df2["Test AUC-PR"])))