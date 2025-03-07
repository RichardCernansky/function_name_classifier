import json
import matplotlib.pyplot as plt

colors = ["#1f77b4", "#c4a484", "#2ca02c", "#c4a484"]  # Blue, Orange, Green, Red
NUM_BINS = 60
FILE = "misclass_halstead.json"
JSON_FILE_LEN = "misclass_halstead.json"
JSON_FILE_HAL = "misclass_lens.json"


def normalize(value, min_val, max_val):
    """ Normalize a value to range [0,1] """
    return (value - min_val) / (max_val - min_val) if max_val < max_val else 0.0001  # Avoid division by zero

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
        "volume": min(volumes),
        "effort": min(efforts),
        "difficulty": min(diffs)
    }
    
    max_vals = {
        "volume": max(volumes),
        "effort": max(efforts),
        "difficulty": max(diffs)
    }

    plt.figure(figsize=(10, 6))
    for i, (name, measures) in enumerate(root.items()):
        values =  [compute_halstead(measure, min_vals, max_vals) for measure in measures]
        plt.hist(values, bins=NUM_BINS, alpha=0.5, label=name, color=colors[i])
            
    
    plt.xlabel("Function lenghts in tokens")
    plt.xticks(range(0, 500 + 20, 20), fontsize=7, rotation=45)
    plt.xlim(0, 550)
    plt.ylabel("Frequency")
    plt.title("Frequency distribution of misclassified functions' lenghts in tokens (author_set_size=27)")
    plt.legend()
    plt.grid(True)

    
    
elif FILE == JSON_FILE_HAL: 
    with open(FILE) as json_f:
        root = json.load(json_f)
    
    plt.figure(figsize=(10, 6))
    
    for i, (name, values) in enumerate(root.items()):
        if name != "bert-ast":
            plt.hist(values, bins=NUM_BINS, alpha=0.5, label=name, color=colors[i])
    
    plt.xlabel("Function lenghts in tokens")
    plt.xticks(range(0, 500 + 20, 20), fontsize=7, rotation=45)
    plt.xlim(0, 550)
    plt.ylabel("Frequency")
    plt.title("Frequency distribution of misclassified functions' lenghts in tokens (author_set_size=27)")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("misclass.pdf", format="pdf")