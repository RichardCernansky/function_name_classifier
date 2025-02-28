import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

NUM_FOLDS = 5

heatmap_pdf_file = "analysis/metrics_plots/average_class_metrics_heatmap_full.pdf"
prefix_bin_pdf_file = "analysis/metrics_plots/metrics_bins/"
agg_avg_metrics_file = "analysis/agg_average_metrics_plot.png"

# initialize accumulators for metrics
total_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
bin_accuracies = {}
all_labels = []
true_labels = []
predicted_labels = []

bin_accuracies_keys = [
    "num_tokens_50_bin_accuracies",
    "num_tokens_20_bin_accuracies",
    "ast_depth_5_bin_accuracies",
    "ast_depth_2_bin_accuracies",
    "num_nodes_50_bin_accuracies",
    "num_nodes_20_bin_accuracies"
]

# load all fold metrics
for fold_idx in range(NUM_FOLDS):
    with open(f"analysis/metrics_json/fold_{fold_idx+1}_metrics.json", "r") as f:
        fold_metrics = json.load(f)
        # aggregate accuracy
        total_metrics["accuracy"] += fold_metrics["accuracy"]
        total_metrics["precision"] += fold_metrics["precision"]
        total_metrics["recall"] += fold_metrics["recall"]
        total_metrics["f1"] += fold_metrics["f1"]

        all_labels += fold_metrics["all_labels"]
        true_labels += fold_metrics["true_lables"]
        predicted_labels += fold_metrics["predicted_labels"]

        key_bin_accuracies = {key: {} for key in bin_accuracies_keys}
        # aggregate bin accuracies
        for key in bin_accuracies_keys:
            if key not in fold_metrics:
                continue
            for bin_label, bin_data in fold_metrics[key].items():
                if bin_label not in key_bin_accuracies[key]:
                    key_bin_accuracies[key][bin_label] = {"correct": 0, "total": 0}
                key_bin_accuracies[key][bin_label]["correct"] += bin_data["correct"]
                key_bin_accuracies[key][bin_label]["total"] += bin_data["total"]

# compute average metrics
average_metrics_model = {metric: total_metrics[metric] / NUM_FOLDS for metric in total_metrics}

# PLOT AVERAGE METRICS
metrics = list(average_metrics_model.keys())
values = list(average_metrics_model.values())
plt.figure(figsize=(8, 5))
bars = plt.bar(metrics, values, color=['blue', 'orange', 'green', 'red'], alpha=0.7)
# add exact values on the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{value:.4f}",
             ha='center', va='bottom', fontsize=10, color='black')
plt.title("Average Metrics Across Folds", fontsize=14)
plt.ylabel("Score", fontsize=12)
plt.ylim(0, 1)
plt.xticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(agg_avg_metrics_file, dpi=300)

#-----BINS-----
#average bin accuracies
average_bin_accuracies_per_key = {}
for key, bins in key_bin_accuracies.items():
    average_bin_accuracies_per_key[key] = {}
    for bin_label, values in bins.items():
        avg_accuracy = values["correct"] / values["total"] if values["total"] > 0 else 0
        average_bin_accuracies_per_key[key][bin_label] = avg_accuracy


#----------------------PLOTTING-------------------------
# --- Plot 1: Average Accuracy and Bin Accuracies ---
for key, bins in average_bin_accuracies_per_key.items():
    plt.figure(figsize=(12, 8))

    bin_labels = list(bins.keys())
    bin_values = list(bins.values())

    bars = plt.bar(bin_labels, bin_values, color="skyblue", edgecolor="black")

    for bar, value in zip(bars, bin_values):
        plt.text(bar.get_x() + bar.get_width() / 2,  # X position (center of bar)
                 value + 0.01,  # Y position (slightly above bar)
                 f"{value:.4f}",  # Format value with 4 decimal places
                 ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.axhline(y=average_metrics_model["accuracy"], color='red', linestyle='--', label=f"Model Avg. Accuracy: {average_metrics_model['accuracy']:.4f}")

    plt.legend(loc="upper right", fontsize=12)

    key_split = key.split("_")
    
    plt.title(f"Bin Average Accuracies for {key_split[0]} {key_split[1]}. Bin step = {key_split[2]}", fontsize=18, fontweight="bold")
    plt.xlabel("Bins", fontsize=14, labelpad=10)
    plt.ylabel("Accuracy", fontsize=14, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{prefix_bin_pdf_file}{key}.pdf")


#------REPORT AND CONFUSION MATRIX-----
# put the measurements together
report = classification_report(
    true_labels,
    predicted_labels,
    labels=all_labels,  # Explicitly specify all classes
    target_names=target_names,
    output_dict=True
)
conf_matrix = confusion_matrix(true_labels, predicted_labels)
#---------------------------------------------------------------------
# --- Plot Confusion Matrix-Like Visualization ---
sorted_classes = sorted(
    report.keys(),
    key=lambda cls: (
        report[cls]["support"],      # Primary: Support
        report[cls]["precision"],   # Tertiary: Precision
        report[cls]["recall"]       # Quaternary: Recall
    ),
    reverse=True  # Sort in descending order for all metrics
)
#metrics
metrics = ["precision", "recall", "f1-score"]
heatmap_report = np.array([
    [report[cls][metric] for metric in metrics]
    for cls in sorted_classes
])
sorted_supports = [report[cls]["support"] for cls in sorted_classes]
yticklabels = [f"{cls} (Support: {support})" for cls, support in zip(sorted_classes, sorted_supports)]
figure_height = len(sorted_classes) * 0.6
plt.figure(figsize=(12, figure_height))
sns.heatmap(
    heatmap_report,
    annot=True,
    fmt=".2f",
    xticklabels=metrics,
    yticklabels=yticklabels,
    cmap="coolwarm",
    cbar_kws={'label': 'Metric Value'}
)
plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position('top')
plt.title(f"Average Metrics Per Class Sorted by Support ({len(sorted_classes)} Classes)", fontsize=16, pad=30)
plt.xlabel("")
plt.ylabel("Classes", fontsize=14)
plt.tight_layout()
plt.savefig(heatmap_pdf_file)

# plot the confusion matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="coolwarm", linewidths=0.5)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix Heatmap")
plt.show()


