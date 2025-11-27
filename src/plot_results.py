import matplotlib.pyplot as plt
import os

def plot_predictions(y_test, y_pred_lr, y_pred_rf):
    os.makedirs("../results/figures", exist_ok=True)

    plt.figure(figsize=(10,6))
    plt.plot(y_test, label="Actual PM2.5", alpha=0.7)
    plt.plot(y_pred_lr, label="Linear Regression", alpha=0.7)
    plt.plot(y_pred_rf, label="Random Forest", alpha=0.7)
    plt.xlabel("Samples")
    plt.ylabel("PM2.5")
    plt.legend()
    plt.title("PM2.5 Predictions vs Actual")
    plt.tight_layout()
    plt.savefig("../results/figures/predictions_plot.png")
    plt.show()
