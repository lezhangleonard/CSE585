import json
import re
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 180
plt.rcParams["savefig.bbox"] = "tight"


# -------------------------
# Parsing
# -------------------------
def parse_workload(name: str):
    match = re.search(r"w_(\d+)_hot_([\d.]+)", name)
    if match:
        return int(match.group(1)), float(match.group(2))
    return None, None


def build_df(root_dir: str):
    data = []
    root = Path(root_dir)

    for stats_path in root.glob("**/stats.json"):
        parts = stats_path.relative_to(root).parts
        if len(parts) < 4:
            continue

        run_type, br_str, workload_str, mode = parts[0], parts[1], parts[2], parts[3]

        size, hot_ratio = parse_workload(workload_str)
        if size is None:
            continue

        try:
            s = json.loads(open(stats_path).read())

            data.append({
                "run_type": run_type.lower(),
                "br": int(br_str.replace("br_", "")),
                "N": size,
                "hot": hot_ratio,
                "mode": mode.lower(),
                "correctness": s.get("correctness", 0.0),
                "throughput": s.get("throughput_updates_per_sec", 0.0),
                "width": s.get("avg_parallel_width", 1.0),
                "latency": s.get("reasoner_avg_time_per_decision_sec", 0.0),
                "total_updates": s.get("total_updates", 0.0),
                "num_no_ops": s.get("num_no_ops", 0.0),
                "num_mutations": s.get("num_mutations", 0.0),
                "extractor_llm_time_sec": s.get("extractor_llm_time_sec", 0.0),
                "reasoner_llm_time_sec": s.get("reasoner_llm_time_sec", 0.0),
                "total_time_sec": s.get("total_time_sec", 0.0),
            })

        except Exception:
            continue

    df = pd.DataFrame(data)
    if df.empty:
        return df

    df["mutation_ratio"] = df["num_mutations"] / df["total_updates"].replace(0, pd.NA)
    df["noop_ratio"] = df["num_no_ops"] / df["total_updates"].replace(0, pd.NA)
    df["llm_time_sec"] = (
    df["extractor_llm_time_sec"] + df["reasoner_llm_time_sec"])
    df["non_llm_time_sec"] = (df["total_time_sec"] - df["llm_time_sec"])
    df["llm_fraction"] = (df["llm_time_sec"] / df["total_time_sec"].replace(0, pd.NA))
    df["extractor_fraction"] = (df["extractor_llm_time_sec"] / df["total_time_sec"].replace(0, pd.NA))
    df["reasoner_fraction"] = (df["reasoner_llm_time_sec"] / df["total_time_sec"].replace(0, pd.NA))

    return df


# -------------------------
# Speedup metrics
# -------------------------
def add_speedup_tables(df):
    wide = df.pivot_table(
        index=["run_type", "br", "N", "hot"],
        columns="mode",
        values=["throughput", "correctness", "width"],
        aggfunc="mean"
    )

    wide.columns = [f"{m}_{mode}" for m, mode in wide.columns]
    wide = wide.reset_index()

    if {"throughput_dag", "throughput_sequential"}.issubset(wide.columns):
        wide["dag_speedup_vs_seq"] = wide["throughput_dag"] / wide["throughput_sequential"]

    return wide


def save(out, name):
    plt.savefig(out / name)
    plt.close()


def make_poster_figure(df, out_dir="evaluation/plots"):
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dff = df[(df["run_type"] == "real") & (df["N"] == 500)].copy()

    mode_order = ["sequential", "dag", "batch"]
    palette = {
        "sequential": "black",
        "dag": "tab:blue",
        "batch": "tab:orange"
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    metrics = ["correctness", "throughput"]

    for i, col in enumerate(metrics):
        ax = axes[i]

        sns.lineplot(
            data=dff,
            x="hot",
            y=col,
            hue="mode",
            hue_order=mode_order,
            palette=palette,
            marker="o",
            linewidth=2.5,
            ax=ax
        )

        # Clean per-axis legend (INSIDE plot)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        # Subplot titles (smaller)
        ax.set_title(
            "Correctness under Contention" if col == "correctness"
            else "Throughput under Contention",
            fontsize=11
        )

        if col == "correctness":
            ax.axhline(
                y=1.0,
                color="black",
                linestyle="--",
                linewidth=2.5,
                alpha=0.9
            )

        ax.set_xlabel("Hot Ratio")

    axes[0].legend(
        title="Executor",
        loc="lower left",
        frameon=True,
        fontsize=9,
        title_fontsize=10 
    )
    axes[0].set_ylabel("Correctness (0–1)")
    axes[1].set_ylabel("Throughput (updates/sec)")

    # Strong hierarchy: BIG main title
    fig.suptitle(
        "DAG vs Sequential vs Batch (Real, N=500)",
        fontsize=14,
        fontweight="bold",
        y=1.05
    )

    # spacing tuned for suptitle + internal legends
    fig.subplots_adjust(top=0.80, wspace=0.25)

    out_path = out / "poster_real_N500.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

# -------------------------
# Plotting
# -------------------------
def generate_plots(df, out_dir="evaluation/plots"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    speed = add_speedup_tables(df)

    df.to_csv(out / "master_metrics.csv", index=False)

    # modes order
    mode_order = [m for m in ["sequential", "dag", "batch"] if m in set(df["mode"])]

    # =========================================================
    # 01 Correctness
    # =========================================================
    palette = {
        "dag": "tab:blue",
        "batch": "tab:orange"
    }

    plot_df = df[df["mode"].isin(["dag", "batch"])]

    g = sns.relplot(
        data=plot_df,
        x="hot",
        y="correctness",
        hue="mode",
        style="mode",
        palette=palette,
        col="N",
        row="run_type",
        kind="line",
        marker="o",
        linewidth=2.5,
        hue_order=["dag", "batch"]
    )

    for ax in g.axes.flat:
        ax.axhline(
            y=1.0,
            color="black",
            linestyle="--",
            linewidth=2.5,
            alpha=0.9
        )

    # Build custom legend handles
    handles = [
        Line2D([0], [0], color="black", lw=2.5, ls="--", label="sequential"),
        Line2D([0], [0], color="tab:blue", lw=2.5, marker="o", label="dag"),
        Line2D([0], [0], color="tab:orange", lw=2.5, ls="--", marker="o", label="batch"),
    ]

    # Remove seaborn's default legend and replace it
    if g._legend is not None:
        g._legend.remove()

    g.figure.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        title="mode",
        loc="center right",
        bbox_to_anchor=(1.03, 0.5)
    )

    g.set_axis_labels("Hot ratio", "Correctness (0–1)")
    g.figure.suptitle("Correctness under Contention", y=1.02)
    g.savefig(out / "01_correctness.png", bbox_inches="tight")
    plt.close(g.figure)

    # =========================================================
    # 02 Throughput
    # =========================================================
    g = sns.relplot(
        data=df,
        x="hot", y="throughput",
        hue="mode",
        col="N", row="run_type",
        kind="line", marker="o"
    )
    g.set_axis_labels("Hot ratio", "Throughput (updates/sec)")
    g.figure.suptitle("Throughput under Contention", y=1.02)
    g.savefig(out / "02_throughput.png")
    plt.close(g.figure)

    # =========================================================
    # 03 DAG speedup
    # =========================================================
    g = sns.relplot(
        data=speed,
        x="hot", y="dag_speedup_vs_seq",
        hue="N",
        col="br", row="run_type",
        kind="line", marker="o"
    )
    g.set_axis_labels("Hot ratio", "Speedup (DAG / Sequential)")
    g.figure.suptitle("DAG Speedup over Sequential", y=1.02)
    g.savefig(out / "03_speedup.png")
    plt.close(g.figure)

    # =========================================================
    # 04 DAG width
    # =========================================================
    dag = df[df["mode"] == "dag"]

    g = sns.relplot(
        data=dag,
        x="hot", y="width",
        hue="N",
        col="br", row="run_type",
        kind="line", marker="o"
    )
    g.set_axis_labels("Hot ratio", "Parallel width (active branches)")
    g.figure.suptitle("DAG Parallel Width", y=1.02)
    g.savefig(out / "04_width.png")
    plt.close(g.figure)

    # =========================================================
    # 05 Latency
    # =========================================================
    g = sns.relplot(
        data=df,
        x="hot", y="latency",
        hue="mode",
        col="N", row="run_type",
        kind="line", marker="o"
    )
    g.set_axis_labels("Hot ratio", "Latency per update (sec)")
    g.figure.suptitle("Reasoning Latency under Contention", y=1.02)
    g.savefig(out / "05_latency.png")
    plt.close(g.figure)

    # =========================================================
    # 06 Heatmap
    # =========================================================
    for rt in dag["run_type"].unique():
        for br in dag["br"].unique():
            sub = dag[(dag["run_type"] == rt) & (dag["br"] == br)]
            if sub.empty:
                continue

            plt.figure(figsize=(8, 5))
            heat = sub.pivot_table(index="N", columns="hot", values="width")
            sns.heatmap(heat, cmap="Blues")
            plt.title(f"DAG Parallel Width Heatmap | {rt} | BR={br}")
            save(out, f"06_heatmap_{rt}_br{br}.png")

    # =========================================================
    # 07 Tradeoff
    # =========================================================
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="throughput", y="correctness",
        hue="mode",
        alpha=0.6
    )
    plt.title("Correctness–Throughput Tradeoff")
    plt.xlabel("Throughput (updates/sec)")
    plt.ylabel("Correctness (0–1)")
    save(out, "07_tradeoff.png")

    # =========================================================
    # 08 Scaling throughput
    # =========================================================
    g = sns.relplot(
        data=df,
        x="N", y="throughput",
        hue="mode",
        col="hot", row="br",
        kind="line", marker="o"
    )
    g.set_axis_labels("Input size (N)", "Throughput (updates/sec)")
    g.figure.suptitle("Throughput Scaling", y=1.02)
    g.savefig(out / "08_scaling.png")
    plt.close(g.figure)

    # =========================================================
    # 09 Latency boxplot
    # =========================================================
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="run_type", y="latency", hue="mode")
    plt.title("Latency Distribution across Workloads")
    plt.ylabel("Latency per update (sec)")
    save(out, "09_latency_box.png")

    # =========================================================
    # 10 No-op ratio
    # =========================================================
    sns.lineplot(data=df, x="hot", y="noop_ratio", hue="mode")
    plt.title("No-op Ratio under Contention")
    save(out, "10_noop.png")

    # =========================================================
    # 11 Mutation ratio
    # =========================================================
    sns.lineplot(data=df, x="hot", y="mutation_ratio", hue="mode")
    plt.title("Mutation Ratio under Contention")
    save(out, "11_mutation.png")

    # =========================================================
    # 12 DAG throughput scaling
    # =========================================================
    g = sns.relplot(
        data=dag,
        x="hot", y="throughput",
        hue="N",
        col="br",
        kind="line", marker="o"
    )
    g.set_axis_labels("Hot ratio", "Throughput (updates/sec)")
    g.figure.suptitle("DAG Throughput Scaling", y=1.02)
    g.savefig(out / "12_dag_throughput.png")
    plt.close(g.figure)

    # =========================================================
    # 13 LLM TIME FRACTION
    # =========================================================
    g = sns.relplot(
        data=df,
        x="hot",
        y="llm_fraction",
        hue="mode",
        col="N",
        row="run_type",
        kind="line",
        marker="o"
    )

    g.set_axis_labels("Hot ratio", "LLM Time / Total Time")
    g.figure.suptitle("LLM Utilization under Contention", y=1.02)
    g.savefig(out / "13_llm_fraction.png")
    plt.close(g.figure)


if __name__ == "__main__":
    df = build_df("./evaluation/eval_runs")
    generate_plots(df)
    # make_poster_figure(df)
    print("Done!!")