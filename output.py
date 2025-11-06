import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("qca_predictions_insample.csv")

plt.scatter(df["y_true"], df["y_pred"], alpha=0.7)
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("QCA Predicted vs True")
plt.plot([df.y_true.min(), df.y_true.max()],
         [df.y_true.min(), df.y_true.max()], 'r--')
plt.show()

