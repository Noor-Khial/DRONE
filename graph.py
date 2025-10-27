import os
from matplotlib import pyplot as plt
import numpy as np

plt.rcParams["figure.dpi"] = 200
plt.style.use("ggplot")
plt.rcParams.update({'font.size': 18, 'font.weight': 'bold'})

def column(A, j):
    return [A[i][j] for i in range(len(A))]

def transpose(A):
    return [column(A, j) for j in range(len(A[0]))]

def process_file(filename):
    with open(filename, 'r') as infile:
        lines = infile.readlines()

    lines = [[eval(x.split(": ")[1]) for x in line.split('\t')] for line in lines]
    data = transpose(lines)
    regret = np.abs(np.array(data[0]))
    regret_bound = np.array(data[1]) / 100
    weights = np.array(transpose(data[2]))
    xs = np.array(list(range(len(data[0]))))

    return regret, regret_bound, weights, xs

def moving_average(xs, ys, window_size=1):
    """Calculates a moving average for the data."""
    ys_avg = np.convolve(ys, np.ones(window_size) / window_size, mode='valid')
    xs_avg = xs[:len(ys_avg)]  # Adjust xs to match the length of ys_avg
    return xs_avg, ys_avg

def plot_regret_bound_and_cumulative(filenames, output_dir):
    """Plots the regret bound and the cumulative regret on the same axes."""
    plt.figure(figsize=(7, 6))
    line_styles = ['-', '--', '-', '--']
    colors = ['b', 'g', 'r', 'c']

    for i, filename in enumerate(filenames):
        base_name = os.path.splitext(os.path.basename(filename))[0]
        regret, regret_bound, _, xs = process_file(filename)

        regret_bound = regret_bound / (xs+1)
        print(regret_bound)

        # Plot cumulative regret
        plt.plot(
            xs, regret,
            label=f"Average Regret",
            linewidth=3,
            linestyle=line_styles[i % len(line_styles)],
            color=colors[i % len(colors)]
        )

        # Plot regret bound
        plt.plot(
            xs, regret_bound,
            label=f"Regret Bound",
            linewidth=3,
            linestyle='--',
            color=colors[(i + 1) % len(colors)]
        )

    plt.xlabel("Time slot")
    plt.ylabel("Regret / Bound")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "r6.png"))
    plt.close()

def plot_action_probabilities(filenames, output_dir):
    plt.figure(figsize=(7, 6))
    line_styles = ['--', '--', '-.', '-.']
    colors = ['b', 'g', 'r', 'c']
    for i, filename in enumerate(filenames):
        base_name = os.path.splitext(os.path.basename(filename))[0]
        _, _, weights, xs = process_file(filename)
        xs_avg, weights_avg = moving_average(xs, weights[0])
        #plt.plot(xs_avg, weights_avg, label=f"{base_name}", linewidth=3, linestyle=line_styles[i % len(line_styles)], color=colors[i % len(colors)])
        plt.plot(xs, weights[0], label=f"Policy (env1)", linewidth=3, linestyle=line_styles[1 % len(line_styles)], color=colors[1 % len(colors)])
        plt.plot(xs, weights[1], label=f"Policy (env3)", linewidth=3, linestyle=line_styles[2 % len(line_styles)], color=colors[2 % len(colors)])
    plt.xlabel("Time slot")
    plt.ylabel("Best Policy Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "action_probability.png"))
    plt.close()

def plot_utility(filenames, output_dir):
    plt.figure(figsize=(7, 6))
    line_styles = ['-', '--', '-.', ':']
    colors = ['b', 'g', 'r', 'c']
    for i, filename in enumerate(filenames):
        base_name = os.path.splitext(os.path.basename(filename))[0]
        _, _, _, xs = process_file(filename)
        utility = np.random.random(len(xs))  # Replace with actual utility calculation
        xs_avg, utility_avg = moving_average(xs, utility)
        plt.plot(xs_avg, utility_avg, label=f"{base_name} Utility", linewidth=3, linestyle=line_styles[i % len(line_styles)], color=colors[i % len(colors)])
    plt.xlabel("Time slot ($x10^{4}$)")
    plt.ylabel("Utility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "utility.png"))
    plt.close()

def main():
    # Add your filenames here
    input_files = ["weights/exp3.txt"]
    output_dir = "FIGs"
    os.makedirs(output_dir, exist_ok=True)

    # New combined plot: regret bound vs cumulative regret
    plot_regret_bound_and_cumulative(input_files, output_dir)

    # Keep the other plots if you still want them
    plot_action_probabilities(input_files, output_dir)
    plot_utility(input_files, output_dir)

if __name__ == "__main__":
    main()
