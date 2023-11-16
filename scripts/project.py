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
import umap
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Turbo256


def load_data(file_path):
    with np.load(file_path) as data:
        labels = data["labels"]
        features = data["features"]
        urls = data["urls"] if "urls" in data else None
    return labels, features, urls


def umap_projection(features):
    reducer = umap.UMAP(n_jobs=-1)
    embedding = reducer.fit_transform(features)
    return embedding


def color_map(labels):
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)

    # Using Turbo256 color map
    turbo_colors = Turbo256
    step = max(1, int(len(turbo_colors) / n_labels))
    color_dict = {
        label: turbo_colors[i * step % len(turbo_colors)]
        for i, label in enumerate(unique_labels)
    }
    color_dict[-1] = "#000000"  # Color for noise
    colors = [color_dict[label] for label in labels]
    return colors


def plot_with_bokeh(embedding, labels, colors, urls=None):
    data_dict = dict(x=embedding[:, 0], y=embedding[:, 1], label=labels, color=colors)
    if urls is not None:
        data_dict["img_url"] = urls

    source = ColumnDataSource(data=data_dict)

    plot = figure(
        title="UMAP Projection of Features",
        tools="wheel_zoom,reset",
        sizing_mode="stretch_both",
    )
    renderer = plot.scatter("x", "y", color="color", source=source)

    if urls is not None:
        hover = HoverTool(
            renderers=[renderer],
            tooltips="""
                <div>
                    <img
                        src="@img_url" height="84" width="84"
                        style="float: left; margin: 0px 15px 15px 0px;"
                        border="2"
                    ></img>
                    <span style="font-size: 17px; font-weight: bold;">Cluster: </span>
                    <span style="font-size: 15px;">@label</span>
                </div>
            """,
        )
        plot.add_tools(hover)
    else:
        hover = HoverTool(renderers=[renderer], tooltips=[("Label", "@label")])
        plot.add_tools(hover)

    output_file("scripts/umap_projection.html")
    show(plot)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("file_path")
    args = ap.parse_args()

    print("loading data...")
    labels, features, urls = load_data(args.file_path)

    # Perform UMAP Projection
    print(f"creating projection...")
    embedding = umap_projection(features)

    # Create a color map
    colors = color_map(labels)

    # Plot with Bokeh
    plot_with_bokeh(embedding, labels, colors, urls)
