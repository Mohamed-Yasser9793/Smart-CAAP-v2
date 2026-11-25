"""Shared Plotly figures for the research dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ACCENT = "#8EBCFF"
TEAL = "#5EEAD4"
GOLD = "#E7C36A"
CORAL = "#F0A08A"
MUTED = "#9FB3CE"
GRID = "rgba(160, 190, 230, 0.12)"


def _layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=16, color="#F2F7FF")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8, 22, 42, 0.35)",
        font=dict(family="Segoe UI, sans-serif", color="#D5E4F7", size=13),
        height=height,
        margin=dict(l=52, r=24, t=64, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#10243F", bordercolor="#8EBCFF", font=dict(color="#F4F8FF")),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(160,190,230,0.2)")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(160,190,230,0.2)")
    return fig


def _bar_colors(names: list[str]) -> list[str]:
    colors = []
    for name in names:
        if "CAAP" in name:
            colors.append(TEAL)
        elif "Earlier" in name:
            colors.append(GOLD)
        elif "CLAHE" in name:
            colors.append(CORAL)
        elif "Blind" in name:
            colors.append(GOLD)
        else:
            colors.append(ACCENT)
    return colors


def metric_bars(rows: list[dict], metric: str, title: str) -> go.Figure:
    names = [row["Pipeline"] for row in rows]
    values = [row[metric] for row in rows]
    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker=dict(color=_bar_colors(names), line=dict(width=0)),
            text=[f"{value:.4f}" if value < 2 else f"{value:.2f}" for value in values],
            textposition="outside",
            hovertemplate="%{x}<br>" + metric + ": %{y:.4f}<extra></extra>",
        )
    )
    top = max(values) if values else 1
    span = 1.18 if top <= 1.2 else top * 1.18
    fig.update_yaxes(range=[0, span])
    return _layout(fig, title, height=440)


def metric_bars_with_ci(
    df: pd.DataFrame,
    metric: str,
    ci_low_col: str | None,
    ci_high_col: str | None,
    title: str,
) -> go.Figure:
    """Bar chart with asymmetric 95% patient-cluster bootstrap CI error bars."""
    names = df["DisplayPipeline"].tolist()
    values = df[metric].astype(float).tolist()
    error_y = None
    if ci_low_col and ci_high_col and ci_low_col in df.columns and ci_high_col in df.columns:
        lows = df[ci_low_col].astype(float).tolist()
        highs = df[ci_high_col].astype(float).tolist()
        error_y = dict(
            type="data",
            symmetric=False,
            array=[max(0.0, h - v) for v, h in zip(values, highs)],
            arrayminus=[max(0.0, v - l) for v, l in zip(values, lows)],
            color="#E6EEF8",
            thickness=1.8,
            width=6,
        )
        top = max(highs) if highs else 1.0
    else:
        top = max(values) if values else 1.0

    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            error_y=error_y,
            marker=dict(color=_bar_colors(names), line=dict(width=0)),
            text=[f"{v:.4f}" if v < 2.0 else f"{v:.2f}" for v in values],
            textposition="outside",
            hovertemplate="%{x}<br>" + metric + ": %{y:.4f}<extra></extra>",
        )
    )
    span = 1.18 if top <= 1.05 else top * 1.22
    fig.update_yaxes(range=[0, span])
    return _layout(fig, title, height=440)


def confusion_matrix_heatmap(tp: int, tn: int, fp: int, fn: int, title: str) -> go.Figure:
    """Dark-themed 2x2 study-level confusion matrix with exact integer counts."""
    z = [[tn, fp], [fn, tp]]
    text = [[f"TN<br><b>{tn}</b>", f"FP<br><b>{fp}</b>"], [f"FN<br><b>{fn}</b>", f"TP<br><b>{tp}</b>"]]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["Predicted Negative", "Predicted Positive"],
            y=["Actual Negative", "Actual Positive"],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=15, color="#F7FBFF"),
            colorscale=[[0.0, "#0C213B"], [0.5, "#1A4A7A"], [1.0, "#2B8A8A"]],
            showscale=False,
            hovertemplate="%{y} / %{x}: %{z} studies<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, title, height=340)


def routing_donut(counts: dict[str, int], title: str) -> go.Figure:
    labels = list(counts.keys())
    values = list(counts.values())
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.62,
            sort=False,
            marker=dict(colors=[ACCENT, TEAL, GOLD], line=dict(color="#071422", width=2)),
            textinfo="percent",
            hovertemplate="%{label}<br>%{value} views<br>%{percent}<extra></extra>",
        )
    )
    fig.update_layout(
        annotations=[dict(text=f"{sum(values)} views", x=0.5, y=0.5, font=dict(size=14, color="#F4F8FF"), showarrow=False)]
    )
    return _layout(fig, title, height=420)


def training_figure(rows: list[dict]) -> go.Figure:
    epochs = [row["Epoch"] for row in rows]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=[row["Validation ROC-AUC"] for row in rows],
            mode="lines+markers",
            name="Validation ROC-AUC",
            line=dict(color=TEAL, width=3),
            marker=dict(size=10),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=[row["Train BCE loss"] for row in rows],
            mode="lines+markers",
            name="Train BCE loss",
            line=dict(color=GOLD, width=3),
            marker=dict(size=10),
            yaxis="y2",
        )
    )
    fig.update_layout(
        yaxis=dict(title="Validation ROC-AUC", range=[0.88, 0.96]),
        yaxis2=dict(title="Train BCE loss", overlaying="y", side="right", range=[0.15, 0.55]),
        xaxis=dict(title="Epoch", dtick=1),
    )
    return _layout(fig, "Recorded restoration run for best_resnet50_wrist.pth", height=440)


def tier_lines(
    frame: pd.DataFrame,
    metric: str,
    ci_low_col: str | None = None,
    ci_high_col: str | None = None,
    title: str | None = None,
) -> go.Figure:
    colors = {
        "Raw unfiltered": ACCENT,
        "CLAHE": CORAL,
        "Blind global stretch": GOLD,
        "CAAP": TEAL,
        "Smart CAAP v2.0": TEAL,
    }
    pipe_col = "DisplayPipeline" if "DisplayPipeline" in frame.columns else "Pipeline"
    fig = go.Figure()
    for pipeline, group in frame.groupby(pipe_col, sort=False):
        error_y = None
        if ci_low_col and ci_high_col and ci_low_col in group.columns and ci_high_col in group.columns:
            vals = group[metric].astype(float).tolist()
            lows = group[ci_low_col].astype(float).tolist()
            highs = group[ci_high_col].astype(float).tolist()
            error_y = dict(
                type="data",
                symmetric=False,
                array=[max(0.0, h - v) for v, h in zip(vals, highs)],
                arrayminus=[max(0.0, v - l) for v, l in zip(vals, lows)],
                thickness=1.4,
                width=4,
            )
        fig.add_trace(
            go.Scatter(
                x=group["Tier"],
                y=group[metric],
                error_y=error_y,
                mode="lines+markers",
                name=str(pipeline),
                line=dict(color=colors.get(str(pipeline), MUTED), width=3),
                marker=dict(size=9),
            )
        )
    return _layout(fig, title or f"Synthetic stress test · {metric}", height=460)


def probability_bars(pipelines: list[dict]) -> go.Figure:
    names = [item["name"] for item in pipelines]
    values = [item["probability"] for item in pipelines]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(color=[ACCENT, CORAL, GOLD, TEAL], line=dict(width=0)),
            text=[f"{value:.1%}" for value in values],
            textposition="outside",
            hovertemplate="%{y}<br>probability %{x:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, 1.15], title="Positive-class probability")
    return _layout(fig, "Frozen ResNet-50 scores for this view", height=360)


def intensity_histogram(model_input: np.ndarray, caap_image: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=model_input.ravel(),
            name="Model input",
            nbinsx=64,
            opacity=0.62,
            marker_color=ACCENT,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=caap_image.ravel(),
            name="CAAP output",
            nbinsx=64,
            opacity=0.62,
            marker_color=TEAL,
        )
    )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Pixel value", range=[0, 255])
    fig.update_yaxes(title="Count")
    return _layout(fig, "Intensity distribution", height=380)
