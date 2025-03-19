import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_feature_importance(IG_recurrent, feature_names, title="Feature Importance"):
    plt.figure(figsize=(10, 6))
    sns.heatmap(np.mean(IG_recurrent, axis=0), cmap="coolwarm", center=0, annot=False)
    plt.xlabel("Features")
    plt.ylabel("Time Steps")
    plt.title(title)
    plt.xticks(ticks=np.arange(len(feature_names)), labels=feature_names)
    plt.show()