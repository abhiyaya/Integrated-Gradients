import plotly.express as px
import numpy as np
import pandas as pd

def plot_feature_importance(IG_recurrent, feature_names, title="Integrated Gradients Attribution",save_path=None):
    # Average the Integrated Gradients over time steps (days)
    avg_IG = np.mean(IG_recurrent, axis=0)
    
    # Create a DataFrame for Plotly (for better handling of labels)
    df = pd.DataFrame(avg_IG, columns=feature_names)
    
    # Create a heatmap using Plotly
    fig = px.imshow(df.T, labels={'x': 'Features', 'y': 'Time Steps (Days)'}, color_continuous_scale="RdBu", title=title)
    
    # Adjust layout for better readability
    fig.update_layout(
        xaxis_title="Time Steps (Days)",
        yaxis_title="Features",
        width=800,
        height=600
    )
    if save_path:
        fig.write_html(save_path)
        print(f"Figure saved to {save_path}")
    
    # Show the plot
    fig.show()

