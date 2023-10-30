import sys
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Check if the script is inside the 'scripts' directory
if os.path.basename(current_dir) == "scripts":
    root_dir = os.path.dirname(current_dir)
else:
    root_dir = current_dir

sys.path.append(root_dir)
import numpy as np
from bokeh.plotting import show, figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.transform import factor_cmap
from dfclust.ogmc import OGMCGraph
from umap import UMAP
from tqdm import tqdm


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--cutoff", type=int, default=200)
    args = ap.parse_args()

    print(f'loading npz from disk...')
    with np.load(f"{root_dir}/data/test.npz") as f:
        samples = f["features"][: args.cutoff]

    # Use UMAP to reduce dimensionality
    umap_2d = UMAP(n_components=2, n_jobs=-1).fit_transform(samples)

    # Cluster using OGMCGraph
    graph = OGMCGraph()
    for sample in tqdm(samples, desc="Adding samples"):
        graph.add_sample(sample)

    # Get labels
    labels = graph._labels_with_noise
    label_str = labels.astype(str)

    # Create a linear color mapping based on the number of unique labels
    unique_labels = np.unique(label_str)
    cmap = factor_cmap('label', palette="Turbo256", factors=unique_labels.tolist())
    # colors = [cmap[label] for label in label_str]

    # Create a ColumnDataSource from the data
    source_data = ColumnDataSource(data=dict(x=umap_2d[:, 0], y=umap_2d[:, 1], label=labels.astype(str)))

    # Plot with Bokeh
    p = figure(
        title="2D UMAP Projection with OGMCGraph Clustering",
        x_axis_label="UMAP1",
        y_axis_label="UMAP2",
        sizing_mode='stretch_both'
    )

    # Create a scatter plot and capture the renderer to use for the hover tool
    renderer = p.circle(x="x", y="y", source=source_data, color=cmap)

    # Add hover tool
    hover = HoverTool(renderers=[renderer], tooltips=[("Label", "@label")])
    p.add_tools(hover)

    show(p)
