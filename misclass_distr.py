import json
import matplotlib.pyplot as plt

colors = ["#1f77b4", "#c4a484", "#2ca02c", "#c4a484"]  # Blue, Orange, Green, Red
NUM_BINS = 60

with open("misclass_halstead.json") as json_f:
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