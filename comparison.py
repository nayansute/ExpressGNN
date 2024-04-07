import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Load the CSV file into a DataFrame

# Create an argument parser
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

# Plotting
plt.figure(figsize=(10, 6))

# Train Loss
# plt.plot(df['Epoch'].values, df['Train Loss'].values, label='Train Loss', marker='o')


plt.plot(df1['Epoch'].values, df1['Test AUC-PR'].values, label=str(args.arg1), marker='o')
# Test AUC-PR
plt.plot(df2['Epoch'].values, df2['Test AUC-PR'].values, label=str(args.arg2), marker='o')

plt.plot(df3['Epoch'].values, df3['Test AUC-PR'].values, label=str(args.arg3), marker='o')


# Test Log Prob
# plt.plot(df['Epoch'].values, df['Test Log Prob'].values, label='Test Log Prob', marker='o')

# Customize the plot
plt.title('Epoch vs AUC-PR')
plt.xlabel('Epoch')
plt.ylabel('AUC-PR')
plt.legend()
plt.grid(True)
plt.savefig("./cora_results/" + str(args.arg1)[:-4] + str(args.arg2)[:-4] + ".png", bbox_inches='tight')
# Show the plot
plt.show()
