import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# Read data
def read_data(file_path):
    times = []
    rates = []
    with open(file_path, 'r') as f:
        for line in f:
            #print(line)
            parts = line.split(" - ")
            time_part = parts[0]
            time = datetime.strptime(time_part, '%Y-%m-%d %H:%M:%S,%f')
            rate_part = parts[-1]
            rate = float(rate_part)
            # print(time_part, rate_part)

            times.append(time)
            rates.append(rate)
    return times, rates

# Plot data
def plot_data(ax, times, rates, label):
    # Calculate mean and standard deviation
    average_rate = np.mean(rates)
    std_rate = np.std(rates)

    # Remove points with large deviations from the mean
    filtered_times = []
    filtered_rates = []
    for time, rate in zip(times, rates):
        if abs(rate - average_rate) <= 3 * std_rate:  # Adjust threshold or multiplier
            filtered_times.append(time)
            filtered_rates.append(rate)

    ax.plot(filtered_times, filtered_rates, label=label)

# File paths and labels
file_paths = ['logs_rate/model_1_rate.log', 'logs_rate/model_2_rate.log', 'logs_rate/model_3_rate.log', 'logs_rate/model_4_rate.log', 'logs_rate/model_5_rate.log']
labels = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5']

# Create subplots
fig, axs = plt.subplots(5, 1, figsize=(10, 30), sharex=True)

# Plot each subplot
for i, (ax, file_path, label) in enumerate(zip(axs, file_paths, labels)):
    times, rates = read_data(file_path)
    plot_data(ax, times, rates, label)
    ax.set_ylabel('Rate')
    ax.set_title(label)

# Set common x-axis label
axs[-1].set_xlabel('Time')

# Adjust layout
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
