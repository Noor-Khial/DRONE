import random
import matplotlib.pyplot as plt
from collections import Counter

# Set plot styles
plt.rcParams["figure.dpi"] = 200
plt.style.use("ggplot")
plt.rcParams.update({'font.size': 18, 'font.weight': 'bold'})

def sample_policies(n, t, n_i, other_probs, output_file):
    """
    Samples policies based on the given probabilities and saves the sampled indices to a file.
    
    Parameters:
    - n (int): Total number of policies.
    - t (int): Number of timesteps.
    - n_i (int): Index of the policy to sample with a higher probability.
    - other_probs (list): Probabilities for the other policies.
    - output_file (str): Filepath to save sampled indices.
    
    Returns:
    - list: List of sampled policy indices.
    """
    if n <= 1 or n_i < 0 or n_i >= n:
        raise ValueError("Invalid input: ensure n > 1 and 0 <= n_i < n.")
    if len(other_probs) != n - 1:
        raise ValueError("Invalid input: other_probs must have length n-1.")
    if not all(0 <= p <= 1 for p in other_probs):
        raise ValueError("Probabilities must be between 0 and 1.")
    
    # Assign probabilities
    probabilities = other_probs[:n_i] + [1 - sum(other_probs)] + other_probs[n_i:]
    if abs(sum(probabilities) - 1) > 1e-6:
        raise ValueError("Sum of probabilities must equal 1.")
    
    # Sample policies based on the probabilities
    sampled_policies = random.choices(range(n), weights=probabilities, k=t)
    
    # Save sampled indices to file
    with open(output_file, 'w') as f:
        for policy in sampled_policies:
            f.write(f"{policy}\n")
    
    print(f"Sampled policy indices saved to {output_file}")
    return sampled_policies

def plot_sampling_distribution(sampled_policies, n, output_file):
    """
    Plots the distribution of sampled policies and saves the figure.
    
    Parameters:
    - sampled_policies (list): List of sampled policy indices.
    - n (int): Total number of policies.
    - output_file (str): Filepath to save the figure.
    """
    # Count occurrences of each policy
    counts = Counter(sampled_policies)
    total_samples = len(sampled_policies)
    policies = range(n)
    frequencies = [counts.get(policy, 0) / total_samples for policy in policies]

    # Define custom labels for the x-axis
    x_labels = [f'env{i+1}' for i in range(n)]

    # Plot the distribution
    plt.figure(figsize=(8, 6))
    plt.bar(policies, frequencies, color='skyblue', edgecolor='black')
    plt.xticks(range(n), x_labels)  # Update x-axis labels
    plt.xlabel('Environments', fontsize=18)
    plt.ylabel('Sampling Frequency', fontsize=18)
    plt.grid(axis='y', linestyle='--', alpha=0.9)
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

def read_policy_at_t(file_path, t):
    """
    Reads the policy index at time t from the file.
    
    Parameters:
    - file_path (str): Path to the file containing sampled indices.
    - t (int): The time step (index) to read the policy for.

    Returns:
    - int: The policy index at time t.
    """
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i == t%100:
                    return int(line.strip())
        raise ValueError(f"Index {t} is out of range for the file {file_path}.")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

# Example usage
n = 3          # Total number of policies
t = 100       # Number of timesteps
n_i = 2       # Index of the policy to sample with a higher probability
other_probs = [0.25, 0.25]  # Distinct probabilities for other policies

output_file_txt = 'exp3/sampled_policies(3policies4).txt'  # File to save sampled indices
output_file_fig = 'exp3/sampling_distribution(3policies4).png'  # File to save the plot

# Sample policies, save indices, and plot distribution
sampled = sample_policies(n, t, n_i, other_probs, output_file_txt)
plot_sampling_distribution(sampled, n, output_file_fig)

# Test reading the saved file
# read_policy_at_t(output_file_txt, t= 10)
