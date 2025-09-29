# import seaborn as sns
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import re
from collections import defaultdict
import gc

def downsample(data, window_size):
    """Averages every `window_size` instances."""
    n = len(data) // window_size * window_size  # Trim to multiple of window_size
    return data[:n].reshape(-1, window_size).mean(axis=1)


def kappa_no_chan_probs(window_size, class_values):
    class_values = np.array(class_values)  # Ensure it's a NumPy array
    num_class = len(np.unique(class_values))

    no_change_probabilities = []

    for start in range(0, len(class_values) - window_size + 1, window_size):
        end = start + window_size
        window_class_values = class_values[start:end]

        transitions = 0.0
    
        for j in range(num_class):
            transition = np.sum((window_class_values[:-1] == j) & (window_class_values[1:] == j))
            transitions += transition

        probability = transitions / (len(window_class_values) - 1) if len(window_class_values) > 1 else 0
        no_change_probabilities.append(probability)
    
    return np.array(no_change_probabilities)

def kappa_chan_probs(window_size, class_values):
    num_class = len(class_values.unique())  # Number of unique classes
    total_instances = len(class_values)     # Total number of instances

    # Calculate marginal probabilities of each class
    marginal_probs = class_values.value_counts(normalize=True).values
    
    # Expected agreement (accuracy of random chance classifier)
    expected_agreement = np.sum(marginal_probs ** 2)
    
    # Compute chance accuracy per window
    no_change_probabilities = []
    # class_values = np.array(class_values)  # Convert to numpy array for slicing

    for start in range(0, len(class_values) - window_size + 1, window_size):
        window = class_values[start:start + window_size]
        marginal_probs = window.value_counts(normalize=True).values
        expected_agreement = np.sum(marginal_probs ** 2)
        
        # Compute frequency of each class in the window
        window_counts = np.bincount(window, minlength=num_class)


        window_probs = window_counts / window_size

        # Compute probability of a random classifier being correct in this window
        no_change_prob = np.sum(window_probs ** 2)
        no_change_probabilities.append(no_change_prob)

    return np.array(no_change_probabilities)