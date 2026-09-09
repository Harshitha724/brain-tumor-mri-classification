import csv
import os


# --------------------------------------------------
# Final experiment results
# --------------------------------------------------

results = [
    [
        "Simple CNN",
        "Baseline",
        0.001,
        10,
        94.20,
        None
    ],
    [
        "ResNet34 Frozen",
        "Baseline",
        0.001,
        10,
        88.30,
        None
    ],
    [
        "ResNet34 Frozen",
        "Learning Rate Experiment",
        0.0001,
        10,
        84.82,
        None
    ],
    [
        "ResNet34 Fine-tuned",
        "Baseline",
        0.0001,
        10,
        98.57,
        None
    ],
    [
        "ResNet34 Fine-tuned",
        "Learning Rate Experiment",
        0.00005,
        10,
        99.29,
        94.94
    ],
    [
        "ResNet34 Fine-tuned",
        "Learning Rate Experiment",
        0.00001,
        10,
        98.39,
        None
    ],
    [
        "ResNet34 Fine-tuned",
        "Epoch Experiment",
        0.0001,
        15,
        98.75,
        None
    ]
]


# --------------------------------------------------
# Save CSV
# --------------------------------------------------

os.makedirs("results", exist_ok=True)

output_path = "results/model_comparison.csv"

with open(
    output_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Model",
        "Experiment",
        "Learning Rate",
        "Epochs",
        "Best Validation Accuracy (%)",
        "Final Test Accuracy (%)"
    ])

    writer.writerows(results)


print("Final results saved to:")
print(output_path)
