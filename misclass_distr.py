import json
import matplotlib.pyplot as plt
import numpy as np

colors = {
    "bert-source_code": "#1f77b4",  # Blue
    "bert-ast": "#c4a484",         # Light Brown/Orange
    "rf": "#2ca02c",               # Green
    "att-nn": "#c4a484"            # Light Brown/Orange (same as bert-ast)
}

NUM_BINS = 50
FILE = "misclass_lens.json"
JSON_FILE_LEN = "misclass_lens.json"
JSON_FILE_HAL = "misclass_halstead.json"


def normalize(value, min_val, max_val):
    """ Normalize a value to range [0,1] """
    return (value - min_val) / (max_val - min_val) if min_val < max_val else 0.0001  # Avoid division by zero

def compute_halstead(halstead_metrics, min_vals, max_vals, weights=(0.3, 0.4, 0.3)):
    """ Weighted sum"""
    V_norm = normalize(halstead_metrics['volume'], min_vals['volume'], max_vals['volume'])
    D_norm = normalize(halstead_metrics['difficulty'], min_vals['difficulty'], max_vals['difficulty'])
    E_norm = normalize(halstead_metrics['effort'], min_vals['effort'], max_vals['effort'])

    # Weighted sum
    halstead_score = (weights[0] * V_norm) + (weights[1] * D_norm) + (weights[2] * E_norm)
    return round(halstead_score, 4)

if FILE == JSON_FILE_HAL:
    with open(FILE) as json_f:
        root = json.load(json_f)
        
    volumes = [measure["volume"] for name, measures in root.items() for measure in measures]
    efforts = [measure["effort"] for name, measures in root.items() for measure in measures]
    diffs = [measure["difficulty"] for name, measures in root.items() for measure in measures]
    
    # Create the min and max dictionaries
    min_vals = {
        "volume": np.percentile(volumes, 10),
        "effort": np.percentile(min(efforts), 10),
        "difficulty": np.percentile(min(diffs), 10)
    }
    
    max_vals = {
        "volume": np.percentile(max(volumes),90),
        "effort": np.percentile(max(efforts),90),
        "difficulty": np.percentile(max(diffs), 90)
    }



    plt.figure(figsize=(10, 6))
 
    for i, (name, measures) in enumerate(root.items()):
        index = len(root["bert-source_code"]) / len(measures) if len(measures) > 0 else 0  # Compute scaling factor
        print(f"Scaling factor for {name}: {index}")
    
        if name != "att-nn":  # Skip "att-nn"
            values = [compute_halstead(measure, min_vals, max_vals) for measure in measures]
    
            # Compute histogram manually
            counts, bin_edges = np.histogram(values, bins=NUM_BINS)
            scaled_counts = counts * index  # Scale the frequencies
    
            # Plot scaled histogram without clearing previous ones
            plt.bar(bin_edges[:-1], scaled_counts, width=np.diff(bin_edges), alpha=0.5, color=colors[name], label=name)


    plt.xlabel("Halstead complexity")
    plt.ylabel("Frequency")
    plt.title("Halstead complexity distribution of misclassified functions")
    plt.legend()
    plt.grid(True)
    plt.savefig("misclass_halstead.pdf", format="pdf")

    
    
elif FILE == JSON_FILE_HAL: 
    with open(FILE) as json_f:
        root = json.load(json_f)
    
        plt.figure(figsize=(10, 6))
 
    for i, (name, measures) in enumerate(root.items()):
        index = len(root["att-nn"]) / len(measures) if len(measures) > 0 else 0  # Compute scaling factor
        print(f"Scaling factor for {name}: {index}")
    
        if name != "best-ast":  # Skip "att-nn"
            values = [compute_halstead(measure, min_vals, max_vals) for measure in measures]
    
            # Compute histogram manually
            counts, bin_edges = np.histogram(values, bins=NUM_BINS)
            scaled_counts = counts * index  # Scale the frequencies
    
            # Plot scaled histogram without clearing previous ones
            plt.bar(bin_edges[:-1], scaled_counts, width=np.diff(bin_edges), alpha=0.5, color=colors[name], label=name)


    plt.xlabel("Halstead complexity")
    plt.ylabel("Frequency")
    plt.title("Halstead complexity distribution of misclassified functions")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("misclass_lens.pdf", format="pdf")