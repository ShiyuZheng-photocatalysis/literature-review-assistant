"""Visualization helpers using Plotly and pyvis."""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def similarity_heatmap(sim_matrix: list[list[float]], labels: list[str],
                       title: str = "Paper Similarity") -> go.Figure:
    """Create a paper-paper similarity heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=sim_matrix,
        x=labels,
        y=labels,
        colorscale="YlOrRd",
        zmin=0,
        zmax=1,
        text=np.round(sim_matrix, 3) if sim_matrix else None,
        texttemplate="%{text}",
        textfont={"size": 8},
    ))
    fig.update_layout(
        title=title,
        xaxis_tickangle=-45,
        height=500,
        margin=dict(l=20, r=20, t=50, b=100),
    )
    return fig


def method_cluster_chart(clusters: list[dict]) -> go.Figure:
    """Create a bar chart showing method clusters by paper count."""
    if not clusters:
        return go.Figure()

    labels = [c["label"][:60] for c in clusters]
    counts = [c["paper_count"] for c in clusters]

    fig = go.Figure(data=go.Bar(
        x=counts,
        y=labels,
        orientation="h",
        text=counts,
        textposition="outside",
        marker_color="steelblue",
    ))
    fig.update_layout(
        title="Shared Methods Across Papers",
        xaxis_title="Number of Papers",
        yaxis=dict(autorange="reversed"),
        height=max(300, len(clusters) * 30 + 50),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def presence_heatmap(method_names: list[str], paper_labels: list[str],
                     matrix: list[list[int]], title: str = "Method-Paper Presence") -> go.Figure:
    """Create a presence/absence heatmap (method x paper)."""
    if not method_names or not paper_labels:
        return go.Figure()

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=paper_labels,
        y=method_names,
        colorscale=[[0, "white"], [1, "darkblue"]],
        showscale=False,
    ))
    fig.update_layout(
        title=title,
        xaxis_tickangle=-45,
        height=max(200, len(method_names) * 25 + 80),
        margin=dict(l=20, r=20, t=50, b=100),
    )
    return fig


def intro_network_graph(edges: list[dict], clusters: list[dict] = None) -> str:
    """Create an interactive network graph using pyvis.

    Returns HTML string.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        return "<p>pyvis not installed. Install with: pip install pyvis</p>"

    net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="#333")
    net.barnes_hut()

    # Build node set
    nodes = set()
    for e in edges:
        nodes.add(e["source"])
        nodes.add(e["target"])

    # Color nodes by cluster if provided
    node_cluster = {}
    if clusters:
        colors = px.colors.qualitative.Set2
        for ci, c in enumerate(clusters):
            color = colors[ci % len(colors)]
            for p in c.get("papers", []):
                node_cluster[p] = color

    for node in nodes:
        color = node_cluster.get(node, "#97c2fc")
        net.add_node(node, label=node[:40], color=color, size=20)

    for e in edges:
        net.add_edge(e["source"], e["target"], value=e.get("weight", 1),
                     title=str(e.get("weight", "")))

    return net.generate_html()


def problem_comparison_chart(per_paper: list[dict]) -> go.Figure:
    """Create a bar chart comparing problem counts per paper."""
    if not per_paper:
        return go.Figure()

    labels = [p["paper_label"][:40] for p in per_paper]
    counts = [p["problem_count"] for p in per_paper]

    fig = go.Figure(data=go.Bar(
        x=labels,
        y=counts,
        text=counts,
        textposition="outside",
        marker_color="coral",
    ))
    fig.update_layout(
        title="Identified Problems per Paper",
        xaxis_tickangle=-45,
        yaxis_title="Number of Problems",
        height=400,
        margin=dict(l=20, r=20, t=50, b=100),
    )
    return fig


def open_questions_chart(convergent: list[dict], unique: list[dict]) -> go.Figure:
    """Create a chart showing convergent vs unique open questions."""
    if not convergent and not unique:
        return go.Figure()

    categories = ["Convergent\n(>=2 papers)"] * len(convergent) + ["Unique\n(1 paper)"] * len(unique)
    values = [q["paper_count"] if categories[i].startswith("Convergent") else 1
              for i, q in enumerate(convergent + unique)]
    labels = [q["summary"][:80] for q in convergent + unique]

    fig = go.Figure(data=go.Bar(
        x=labels,
        y=values,
        text=values,
        textposition="outside",
        marker_color=["darkgreen"] * len(convergent) + ["gray"] * len(unique),
    ))
    fig.update_layout(
        title="Open Questions by Convergence",
        xaxis_tickangle=-45,
        yaxis_title="Papers Identifying Gap",
        height=500,
        margin=dict(l=20, r=20, t=50, b=120),
    )
    return fig


def figure_type_chart(figure_types: list[dict]) -> go.Figure:
    """Create a horizontal bar chart of common figure types."""
    if not figure_types:
        return go.Figure()

    labels = [f["figure_type"] for f in figure_types]
    counts = [f["paper_count"] for f in figure_types]

    fig = go.Figure(data=go.Bar(
        x=counts,
        y=labels,
        orientation="h",
        text=counts,
        textposition="outside",
        marker_color="mediumseagreen",
    ))
    fig.update_layout(
        title="Common Figure Types Across Papers",
        xaxis_title="Number of Papers",
        yaxis=dict(autorange="reversed"),
        height=max(300, len(figure_types) * 28 + 50),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def section_coverage_chart(papers: list) -> go.Figure:
    """Show which sections were successfully extracted for each paper."""
    sections = ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]
    labels = [_short_label(p) for p in papers]

    data = []
    for section in sections:
        row = [1 if p.get_section(section) else 0 for p in papers]
        data.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=labels,
        y=sections,
        colorscale=[[0, "lightgray"], [1, "steelblue"]],
        showscale=False,
    ))
    fig.update_layout(
        title="Section Extraction Coverage",
        xaxis_tickangle=-45,
        height=250,
        margin=dict(l=20, r=20, t=50, b=80),
    )
    return fig


def _short_label(paper) -> str:
    if hasattr(paper, 'title') and paper.title:
        words = paper.title.split()
        return " ".join(words[:4]) + ("..." if len(words) > 4 else "")
    return str(paper.source)[:30]
