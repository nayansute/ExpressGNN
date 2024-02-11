import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Load the CSV file into a DataFrame

# Create an argument parser
parser = argparse.ArgumentParser(description='Description of your script.')
parser.add_argument('arg1', type=str, help='Description of arg1')
args = parser.parse_args()
print(f"arg1: {args.arg1}")

# Define command-line arguments
file_path = "exp/kinship/" + str(args.arg1)  # Replace 'your_file_path.csv' with the actual file path
df = pd.read_csv(file_path)

# Plotting
plt.figure(figsize=(10, 6))

# Train Loss
# plt.plot(df['Epoch'].values, df['Train Loss'].values, label='Train Loss', marker='o')

# Test AUC-ROC
plt.plot(df['Epoch'].values, df['Test AUC-ROC'].values, label='Test AUC-ROC', marker='o')

# Test AUC-PR
plt.plot(df['Epoch'].values, df['Test AUC-PR'].values, label='Test AUC-PR', marker='o')

# Test Log Prob
# plt.plot(df['Epoch'].values, df['Test Log Prob'].values, label='Test Log Prob', marker='o')

# Customize the plot
plt.title('Epoch vs Metrics')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig("exp/kinship/" + str(args.arg1)[:-3] + ".png", bbox_inches='tight')
# Show the plot
plt.show()
