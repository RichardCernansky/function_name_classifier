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
average_accuracy = total_accuracy / NUM_FOLDS

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
        # Ensure metrics is a dictionary before processing
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

# print the final metrics
# print(f"{NUM_FOLDS}-Fold Cross Validation Average Model Accuracy: {average_accuracy:.4f}")
# print("Average Bin Accuracies:")
# for bin_label, accuracy in average_bin_accuracies.items():
#     print(f"  {bin_label}: {accuracy:.4f}")
# print("Combined Average Classification Report:")
# for label, metrics in average_report.items():
#     print(f"Class {label}:")
#     for metric, value in metrics.items():
#         print(f"  {metric}: {value:.4f}")

#---------------------------PLOTTING-------------------------
# --- Plot 1: Average Accuracy and Bin Accuracies ---
plt.figure(figsize=(10, 6))

# add average accuracy text
plt.text(0.5, 0.9, f"Average Model Accuracy: {average_accuracy:.4f}",
         fontsize=14, ha="center", transform=plt.gca().transAxes)

# bar plot for bin accuracies
bin_labels = list(average_bin_accuracies.keys())
bin_values = list(average_bin_accuracies.values())
plt.bar(bin_labels, bin_values, color="skyblue")

plt.title("Bin Average Accuracies", fontsize=14)
plt.xlabel("Bins", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig("analysis/average_accuracy_and_bins.pdf")

#---------------------------------------------------------------------
# --- Plot 2: Confusion Matrix-Like Visualization ---
classes = list(average_report.keys())
metrics = ["precision", "recall", "f1-score"]
confusion_matrix_like = np.array([[average_report[cls][metric] for metric in metrics] for cls in classes])

plt.figure(figsize=(12, 8))

# create heatmap for class metrics
sns.heatmap(confusion_matrix_like, annot=True, fmt=".2f", xticklabels=metrics, yticklabels=classes, cmap="coolwarm")

plt.title("Average Metrics Per Class", fontsize=14)
plt.xlabel("Metrics", fontsize=12)
plt.ylabel("Classes", fontsize=12)
plt.tight_layout()

plt.savefig("analysis/average_class_metrics_heatmap.pdf")


# Plot heatmap for 800 classes
plt.figure(figsize=(20, 40))  # Adjust the size for better readability

sns.heatmap(
    confusion_matrix_like,
    annot=False,  # Remove annotations to improve clarity for a large number of classes
    fmt=".2f",
    xticklabels=metrics,
    yticklabels=classes,
    cmap="coolwarm",
    cbar_kws={'label': 'Metric Value'}  # Add a color bar label for context
)

plt.title("Average Metrics Per Class (800 Classes)", fontsize=16)
plt.xlabel("Metrics", fontsize=14)
plt.ylabel("Classes", fontsize=14)
plt.tight_layout()

# Save the heatmap as a PDF
plt.savefig("analysis/average_class_metrics_heatmap_full.pdf")

# Show the heatmap
plt.show()
