import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Description of your script.')

parser.add_argument('arg1', type=str, help='Description of arg1')
parser.add_argument('arg2', type=str, help='Description of arg2')
parser.add_argument('arg3', type=str, help='Description of arg3')

args = parser.parse_args()

# Define command-line arguments
file_path1 = "./cora_results/GAT/" + str(args.arg1)  # Replace 'your_file_path.csv' with the actual file path
file_path2 = "./cora_results/GIN/" + str(args.arg2)
file_path3 = "./cora_results/Original_results/" + str(args.arg3)



df1 = pd.read_csv(file_path1) # original performance
df2 = pd.read_csv(file_path2)
df3 = pd.read_csv(file_path3)


print(file_path1, np.average(np.array(df1["Test AUC-PR"])))
print(file_path2, np.average(np.array(df2["Test AUC-PR"])))
print(file_path3, np.average(np.array(df3["Test AUC-PR"])))