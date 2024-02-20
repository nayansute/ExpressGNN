import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Load the CSV file into a DataFrame

# Create an argument parser
parser = argparse.ArgumentParser(description='Description of your script.')

parser.add_argument('arg1', type=str, help='Description of arg1')
parser.add_argument('arg2', type=str, help='Description of arg2')
args = parser.parse_args()

# Define command-line arguments
file_path1 = "./cora_results/" + str(args.arg1)  # Replace 'your_file_path.csv' with the actual file path
file_path2 = "./cora_results/" + str(args.arg2)

df1 = pd.read_csv(file_path1) # original performance
df2 = pd.read_csv(file_path2)

# Plotting
plt.figure(figsize=(10, 6))

# Train Loss
# plt.plot(df['Epoch'].values, df['Train Loss'].values, label='Train Loss', marker='o')

# Test AUC-ROC
plt.plot(df1['Epoch'].values, df1['Test AUC-PR'].values, label=str(args.arg1), marker='o')

# Test AUC-PR
plt.plot(df2['Epoch'].values, df2['Test AUC-PR'].values, label=str(args.arg2), marker='o')

# Test Log Prob
# plt.plot(df['Epoch'].values, df['Test Log Prob'].values, label='Test Log Prob', marker='o')

# Customize the plot
plt.title('Epoch vs Metrics')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig("./cora_results/" + str(args.arg1)[:-4] + str(args.arg2)[:-4] + ".png", bbox_inches='tight')
# Show the plot
plt.show()