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
classification_reports = []

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

        #aggregate classification_reports
        classification_reports.append(fold_metrics["classification_report"])

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

#average bin accuracies
average_bin_accuracies_per_key = {}
for key, bins in key_bin_accuracies.items():
    average_bin_accuracies_per_key[key] = {}
    for bin_label, values in bins.items():
        avg_accuracy = values["correct"] / values["total"] if values["total"] > 0 else 0
        average_bin_accuracies_per_key[key][bin_label] = avg_accuracy


# put the measurements together
combined_report = {}
for report in classification_reports:
    for label, metrics in report.items():
        # ensure metrics is a dictionary before processing
        if isinstance(metrics, dict):
            if label not in combined_report:
                combined_report[label] = {metric: [] for metric in metrics}

            for metric, value in metrics.items():
                combined_report[label][metric].append(value)
        else:
            print(f"Warning: Metrics for label '{label}' is not a dictionary: {metrics}")


# calculate the average of the measurements
average_report = {}
for label, metrics in combined_report.items():
    average_report[label] = {
        metric: np.mean(values) for metric, values in metrics.items()
    }

#---------------------------PLOTTING-------------------------
# --- Plot 1: Average Accuracy and Bin Accuracies ---
for key, bins in average_bin_accuracies_per_key.items():
    plt.figure(figsize=(12, 8))

    bin_labels = list(bins.keys())
    bin_values = list(bins.values())

    plt.bar(bin_labels, bin_values, color="skyblue", edgecolor="black")

    plt.axhline(y=average_metrics_model["accuracy"], color='red', linestyle='--', label=f"Model Avg. Accuracy: {average_metrics_model['accuracy']:.4f}")

    plt.legend(loc="upper right", fontsize=12)

    plt.text(0.5, 1.05, f"Average Model Accuracy for {key.replace('_', ' ').title()}=step_size : {average_metrics_model['accuracy']:.4f}",
             fontsize=14, ha="center", transform=plt.gca().transAxes, fontweight="bold")
    plt.title(f"Bin Average Accuracies for {key.replace('_', ' ').title()}", fontsize=18, fontweight="bold")
    plt.xlabel("Bins", fontsize=14, labelpad=10)
    plt.ylabel("Accuracy", fontsize=14, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{prefix_bin_pdf_file}{key}.pdf")
#---------------------------------------------------------------------
# --- Plot 2: Confusion Matrix-Like Visualization ---
# Assuming you have `average_report` and `confusion_matrix_like` already computed
classes = list(average_report.keys())
metrics = ["precision", "recall", "f1-score"]
confusion_matrix_like = np.array([[average_report[cls][metric] for metric in metrics] for cls in classes])

# Adjust figure height dynamically based on the number of classes
figure_height = len(classes) * 0.2
plt.figure(figsize=(20, figure_height))

# Create the heatmap
sns.heatmap(
    confusion_matrix_like,
    annot=False,
    fmt=".2f",
    xticklabels=metrics,
    yticklabels=classes,
    cmap="coolwarm",
    cbar_kws={'label': 'Metric Value'}
)

# Move the x-axis labels (metrics) to the top
plt.gca().xaxis.tick_top()  # Move the ticks to the top
plt.gca().xaxis.set_label_position('top')  # Move the axis label position to the top

# Add title and axis labels
plt.title(f"Average Metrics Per Class ({len(classes)} Classes)", fontsize=16, pad=30)
plt.xlabel("")  # Remove default x-axis label text (since metrics are already at the top)
plt.ylabel("Classes", fontsize=14)

# Adjust layout to fit everything nicely
plt.tight_layout()

# Save and show the plot
plt.savefig(heatmap_pdf_file)


