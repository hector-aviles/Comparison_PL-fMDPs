import os
import re
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# CONFIGURATION
# ==============================

train_dirs = ["Train_01", "Train_50", "Train_100"]

models = {
    "CART": "testing_cart_frequency.txt",
    "LR": "testing_lr_frequency.txt",
    "MLP": "testing_mlp_frequency.txt",
    "NB": "testing_nb_frequency.txt",
    "PL-fMDP": "testing_numeralia_lookup_table_frequency.txt",
    "RF": "testing_rf_frequency.txt",
    "XGBoost": "testing_xgboost_frequency.txt"
}

freq_order = ["Low", "Medium", "Medium-High", "High"]

# Journal-friendly grayscale styles
line_styles = ["-", "--", "-.", ":", (0, (3,1,1,1)), (0,(5,1)), (0,(1,1))]
colors = ["black"] * len(models)


# ==============================
# PARSER FUNCTION
# ==============================

def parse_frequency_file(filepath):
    results = {}

    with open(filepath, "r") as f:
        lines = f.readlines()

    current_range = None

    for line in lines:

        # detect frequency range
        if "Low (0-4)" in line:
            current_range = "Low"
        elif "Medium (5-100)" in line:
            current_range = "Medium"
        elif "Medium-High (101-2000)" in line:
            current_range = "Medium-High"
        elif "High (>2000)" in line:
            current_range = "High"

        # extract F1
        if "F1-score:" in line and current_range is not None:
            match = re.search(r"F1-score:\s+([0-9.]+)\s+±\s+([0-9.]+)", line)
            if match:
                mean = float(match.group(1))
                std = float(match.group(2))
                results[current_range] = (mean, std)

    return results


# ==============================
# PLOTTING
# ==============================

for train in train_dirs:

    plt.figure(figsize=(7,5))

    for i, (model, filename) in enumerate(models.items()):

        filepath = os.path.join("..", train, "models", model, "Results", filename)

        if not os.path.exists(filepath):
            print(f"Missing file: {filepath}")
            continue

        data = parse_frequency_file(filepath)

        means = [data[r][0] for r in freq_order]
        stds  = [data[r][1] for r in freq_order]

        x = np.arange(len(freq_order))

        plt.plot(
            x, means,
            linestyle=line_styles[i],
            color=colors[i],
            linewidth=1.8,
            label=model
        )

        plt.fill_between(
            x,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            color="gray",
            alpha=0.08
        )

    plt.xticks(x, freq_order)
    plt.ylabel("F1-score")
    plt.xlabel("Frequency Range")
    plt.title(f"Test 1 Performance by Frequency Range ({train.replace('_',' ')})")

    plt.ylim(0,1)
    plt.grid(axis="y", linestyle=":", linewidth=0.6)

    plt.legend(frameon=False, fontsize=9)

    plt.tight_layout()

    plt.savefig(f"frequency_plot_{train}.pdf", dpi=300)
    plt.savefig(f"frequency_plot_{train}.png", dpi=300)

    plt.close()

print("Done.")
