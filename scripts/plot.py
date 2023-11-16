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
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Turbo256
from dfclust.ogmc import OGMCGraph
from umap import UMAP
from tqdm import tqdm


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", default="data/test.npz")
    ap.add_argument("-c", "--cutoff", type=int, default=5000)
    args = ap.parse_args()

    print(f"loading npz from disk...")
    with np.load(f"{args.data}") as f:
        samples = f["features"]
        np.random.shuffle(samples)
        samples = samples[: args.cutoff]
        image_urls = f["urls"][: args.cutoff] if "urls" in f else None

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

    # Determine the step size for sampling from Turbo256 based on the number of unique labels minus one
    # (to account for the -1 label if it exists)
    has_negative_one = "-1" in unique_labels
    num_labels = len(unique_labels) - 1 if has_negative_one else len(unique_labels)
    step_size = max(1, len(Turbo256) // num_labels)
    if has_negative_one:
        unique_labels = [label for label in unique_labels if label != "-1"]

    # Create the sparse palette
    sparse_palette = [Turbo256[i] for i in range(0, len(Turbo256), step_size)][
        : len(unique_labels)
    ]

    # If -1 was in the original labels, append #000000 to the palette and '-1' back to the unique_labels
    if has_negative_one:
        sparse_palette.append("#000000")
        unique_labels.append("-1")
    mapper = CategoricalColorMapper(factors=unique_labels, palette=sparse_palette)

    # Create a ColumnDataSource from the data
    if image_urls is not None:
        source_data = ColumnDataSource(
            data=dict(
                x=umap_2d[:, 0], y=umap_2d[:, 1], label=label_str, img_url=image_urls
            )
        )
    else:
        source_data = ColumnDataSource(
            data=dict(x=umap_2d[:, 0], y=umap_2d[:, 1], label=label_str)
        )

    # Plot with Bokeh
    p = figure(
        title="2D UMAP Projection with OGMCGraph Clustering",
        x_axis_label="UMAP1",
        y_axis_label="UMAP2",
        sizing_mode="stretch_both",
    )

    # Create a scatter plot and capture the renderer to use for the hover tool
    renderer = p.circle(
        x="x", y="y", source=source_data, color={"field": "label", "transform": mapper}
    )

    # Add hover tool, adjust based on the presence of image URLs
    if image_urls is not None:
        hover = HoverTool(
            renderers=[renderer],
            tooltips="""
                <img
                    src="@img_url" height="84" width="84"
                    style="width: 80px; height: 80px; vertical-align: middle;"
                    border="2"
                >
                <span style="font-weight:bold"> cluster: </span>@label</p>
                </img>
        """,
        )
    else:
        hover = HoverTool(renderers=[renderer], tooltips=[("Label", "@label")])
    p.add_tools(hover)

    show(p)
