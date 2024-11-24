import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

NUM_FOLDS = 5

# initialize accumulators for metrics
total_accuracy = 0
bin_accuracies = {}
classification_reports = []

# load all fold metrics
for fold_idx in range(NUM_FOLDS):
    with open(f"analysis/fold_{fold_idx+1}_metrics.json", "r") as f:
        fold_metrics = json.load(f)
        # aggregate accuracy
        total_accuracy += fold_metrics["accuracy"]

        #aggregate classification_reports
        classification_reports.append(fold_metrics["classification_report"])

        # aggregate bin accuracies
        for bin_label, bin_data in fold_metrics["bin_accuracies"].items():
            if bin_label not in bin_accuracies:
                bin_accuracies[bin_label] = {"correct": 0, "total": 0}
            bin_accuracies[bin_label]["correct"] += bin_data["correct"]
            bin_accuracies[bin_label]["total"] += bin_data["total"]

# compute average accuracy
average_accuracy_model = total_accuracy / NUM_FOLDS

#average bin accuracies
average_bin_accuracies = {}
for bin_label, bin_data in bin_accuracies.items():
    if bin_data["total"] > 0:
        average_accuracy = bin_data["correct"] / bin_data["total"]
    else:
        average_accuracy = 0
    average_bin_accuracies[bin_label] = average_accuracy

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
plt.figure(figsize=(12, 8))

plt.text(0.5, 1.05, f"Average Model Accuracy: {average_accuracy_model:.4f}",
         fontsize=16, ha="center", transform=plt.gca().transAxes, fontweight="bold")

bin_labels = list(average_bin_accuracies.keys())
bin_values = list(average_bin_accuracies.values())
plt.bar(bin_labels, bin_values, color="skyblue")
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.title("Bin Average Accuracies", fontsize=18, fontweight="bold")
plt.xlabel("Bins", fontsize=14, labelpad=10)
plt.ylabel("Accuracy", fontsize=14, labelpad=10)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("analysis/average_accuracy_and_bins.pdf")

#---------------------------------------------------------------------
# --- Plot 2: Confusion Matrix-Like Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
plt.savefig("analysis/average_class_metrics_heatmap_full.pdf")
plt.show()


