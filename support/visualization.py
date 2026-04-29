# ── support/visualization.py ──────────────────────────────────────────────────
# Results visualisation and reporting:
#   plot_dashboard()    — 7-panel matplotlib results dashboard  (Cell 10)
#   print_summary()     — console + JSON experiment report      (Cell 11)
#   print_outputs()     — confirm Drive output locations        (Cell 12)
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

DRIVE_BASE = "/content/drive/MyDrive/ML_Tracker"


# ═══════════════════════════════════════════════════════════════════════════════
#  Cell 10 — Results Dashboard (7-panel)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_dashboard(
    train_results: list,
    val_results: list,
    test_metrics: list,
    avg: dict,
    best: dict,
    best_train_f1: float,
    drive_base: str = DRIVE_BASE,
) -> None:
    """
    Generate and save a 7-panel results dashboard covering Train → Val → Test.
    Requires train_results, val_results, test_metrics, avg, best, best_train_f1.
    """
    if not (test_metrics and val_results and train_results):
        print("⚠️  Run the full experiment pipeline first to generate results.")
        return

    plt.rcParams.update({
        "figure.facecolor": "#0f0f1a",
        "axes.facecolor":   "#1a1a2e",
        "axes.edgecolor":   "#444466",
        "axes.labelcolor":  "#ccccee",
        "xtick.color":      "#ccccee",
        "ytick.color":      "#ccccee",
        "text.color":       "#ccccee",
        "grid.color":       "#333355",
        "grid.alpha":       0.4,
    })

    fig = plt.figure(figsize=(22, 17))
    fig.suptitle(
        "Real-World Traditional ML Object Tracker\nTrain → Val → Test Results",
        fontsize=17, fontweight="bold", color="#e0e0ff", y=0.98,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

    C = {
        "green":  "#4CAF50", "red":    "#F44336", "blue":   "#2196F3",
        "orange": "#FF9800", "purple": "#9C27B0", "pink":   "#E91E63",
        "cyan":   "#00BCD4", "yellow": "#FFEB3B",
    }

    # ── Panel 1: Train F1 Distribution ───────────────────────────────────────
    ax1  = fig.add_subplot(gs[0, 0])
    f1s  = [r["train_f1"] for r in train_results]
    ax1.hist(f1s, bins=min(30, len(f1s)), color=C["green"],
             edgecolor="#0f0f1a", linewidth=0.4, alpha=0.9)
    ax1.axvline(best_train_f1, color=C["red"], lw=2,
                label=f"Best = {best_train_f1:.4f}")
    ax1.set_title("Train: F1 Distribution\n(all grid-search configs)", fontsize=10)
    ax1.set_xlabel("F1 Score")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=8)
    ax1.grid(True)

    # ── Panel 2: Validation F1 Comparison ────────────────────────────────────
    ax2      = fig.add_subplot(gs[0, 1])
    v_labels = [f"#{r['rank']}" for r in val_results]
    v_f1s    = [r["val_f1"]    for r in val_results]
    bar_cols = [C["red"] if r["rank"] == best["rank"] else C["blue"]
                for r in val_results]
    bars2 = ax2.bar(v_labels, v_f1s, color=bar_cols, edgecolor="#0f0f1a", width=0.55)
    ax2.set_ylim(0, 1.15)
    ax2.set_title("Val: Top Configs — F1 Score", fontsize=10)
    ax2.set_xlabel("Config")
    ax2.set_ylabel("F1")
    for bar, val in zip(bars2, v_f1s):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                 f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax2.legend(handles=[
        mpatches.Patch(color=C["red"],  label="Chosen"),
        mpatches.Patch(color=C["blue"], label="Others"),
    ], fontsize=8)
    ax2.grid(True, axis="y")

    # ── Panel 3: Validation MAE Comparison ───────────────────────────────────
    ax3    = fig.add_subplot(gs[0, 2])
    v_maes = [r["val_mae"] for r in val_results]
    ax3.bar(v_labels, v_maes, color=bar_cols, edgecolor="#0f0f1a", width=0.55)
    ax3.set_title("Val: Top Configs — MAE (px)\n(lower is better)", fontsize=10)
    ax3.set_xlabel("Config")
    ax3.set_ylabel("Pixels")
    ax3.grid(True, axis="y")

    # ── Panel 4: Test Average Metrics ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    if avg:
        m_names = ["Precision", "Recall", "F1", "MOTA"]
        m_vals  = [avg["precision"], avg["recall"], avg["f1"], avg["mota"]]
        m_cols  = [C["purple"], C["orange"], C["green"], C["red"]]
        b4 = ax4.barh(m_names, m_vals, color=m_cols, edgecolor="#0f0f1a", height=0.55)
        ax4.set_xlim(0, 1.18)
        ax4.set_title("Test: Average Metrics", fontsize=10)
        for bar, val in zip(b4, m_vals):
            ax4.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=10, fontweight="bold")
        ax4.grid(True, axis="x")

    # ── Panel 5: Per-Video Precision / Recall / F1 ───────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    if len(test_metrics) > 1:
        vnames_short = [m["video"][:14] for m in test_metrics]
        x5  = np.arange(len(vnames_short))
        w5  = 0.25
        ax5.bar(x5 - w5, [m["precision"] for m in test_metrics],
                w5, label="Precision", color=C["purple"], edgecolor="#0f0f1a")
        ax5.bar(x5,      [m["recall"]    for m in test_metrics],
                w5, label="Recall",    color=C["orange"], edgecolor="#0f0f1a")
        ax5.bar(x5 + w5, [m["f1"]       for m in test_metrics],
                w5, label="F1",        color=C["green"],  edgecolor="#0f0f1a")
        ax5.set_xticks(x5)
        ax5.set_xticklabels(vnames_short, fontsize=8, rotation=12)
        ax5.set_ylim(0, 1.15)
        ax5.legend(fontsize=8)
        ax5.set_title("Test: Per-Video Precision / Recall / F1", fontsize=10)
        ax5.grid(True, axis="y")
    else:
        ax5.text(0.5, 0.5, "Only 1 test video\n(no per-video comparison)",
                 ha="center", va="center", transform=ax5.transAxes, fontsize=11)
        ax5.set_title("Test: Per-Video", fontsize=10)

    # ── Panel 6: ID Switches ──────────────────────────────────────────────────
    ax6   = fig.add_subplot(gs[1, 2])
    vn6   = [m["video"][:14]  for m in test_metrics]
    idsw  = [m["id_switches"] for m in test_metrics]
    bars6 = ax6.bar(vn6, idsw, color=C["pink"], edgecolor="#0f0f1a", width=0.55)
    ax6.set_title("Test: ID Switches per Video\n(lower is better)", fontsize=10)
    ax6.set_ylabel("Count")
    for bar, v in zip(bars6, idsw):
        ax6.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                 str(v), ha="center", fontsize=11, fontweight="bold")
    ax6.grid(True, axis="y")

    # ── Panel 7: Generalisation Curve ────────────────────────────────────────
    ax7        = fig.add_subplot(gs[2, :])
    gen_x      = [1, 2, 3]
    gen_labels = ["Train (best)", "Val (chosen)", "Test (final)"]
    gen_f1s    = [best_train_f1, best.get("val_f1", 0) or 0, avg.get("f1", 0) or 0]
    gen_mota   = [None, best.get("val_mota", 0), avg.get("mota", 0) or 0]

    ax7.plot(gen_x, gen_f1s, "o-", color=C["green"],  lw=2.5, ms=11,
             label="F1 Score", zorder=3)
    ax7.plot([2, 3], [gen_mota[1], gen_mota[2]], "s--", color=C["orange"],
             lw=2.0, ms=9, label="MOTA", zorder=3)
    ax7.fill_between(gen_x, gen_f1s, alpha=0.10, color=C["green"])

    ax7.set_xticks(gen_x)
    ax7.set_xticklabels(gen_labels, fontsize=13, fontweight="bold")
    ax7.set_ylim(0, 1.10)
    ax7.set_ylabel("Score", fontsize=12)
    ax7.legend(fontsize=11)
    ax7.set_title("Generalisation: Train → Val → Test", fontsize=13)
    ax7.grid(True)

    for xi, yi in zip(gen_x, gen_f1s):
        ax7.annotate(
            f"{yi:.4f}", (xi, yi),
            textcoords="offset points", xytext=(0, 14),
            ha="center", fontsize=12, fontweight="bold", color=C["green"],
        )

    dashboard_path = f"{drive_base}/outputs/reports/dashboard.png"
    plt.savefig(dashboard_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"✅ Dashboard saved → {dashboard_path}")

    plt.rcdefaults()


# ═══════════════════════════════════════════════════════════════════════════════
#  Cell 11 — Summary Report
# ═══════════════════════════════════════════════════════════════════════════════
def print_summary(
    split_videos: dict,
    combos: list,
    best_train_f1: float,
    best: dict,
    best_params: dict,
    avg: dict,
    test_metrics: list,
    random_seed: int = 42,
    drive_base: str = DRIVE_BASE,
) -> None:
    """Generate a JSON report and print a formatted experiment summary."""
    report = {
        "experiment_info": {
            "random_seed":   random_seed,
            "warmup_frames": 40,
            "algorithms":    ["MOG2/GMG/YOLO", "LK Optical Flow",
                              "Kalman Filter", "Hungarian Algorithm", "HSV Re-ID"],
        },
        "dataset": {
            "train_videos": split_videos.get("train", []),
            "val_videos":   split_videos.get("val",   []),
            "test_videos":  split_videos.get("test",  []),
        },
        "grid_search": {
            "total_combinations": len(combos),
            "best_train_f1":      best_train_f1,
        },
        "best_params":  best_params,
        "validation": {
            "val_f1":          best.get("val_f1"),
            "val_mae_px":      best.get("val_mae"),
            "val_mota":        best.get("val_mota"),
            "val_id_switches": best.get("val_idswitch"),
        },
        "test_per_video": test_metrics,
        "test_average":   avg,
    }

    report_path = f"{drive_base}/outputs/reports/experiment_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    LINE = "=" * 67
    THIN = "─" * 67

    print(LINE)
    print(f"{'EXPERIMENT SUMMARY':^67}")
    print(LINE)
    print(f"  Training videos   : {', '.join(split_videos.get('train', [])) or '(none)'}")
    print(f"  Validation videos : {', '.join(split_videos.get('val',   [])) or '(none)'}")
    print(f"  Test videos       : {', '.join(split_videos.get('test',  [])) or '(none)'}")
    print()
    print(f"  Grid-search combos : {len(combos):,}")
    print(f"  Best train F1      : {best_train_f1}")
    print(f"  Chosen val F1      : {best.get('val_f1', 'N/A')}")
    print(f"  Chosen val MOTA    : {best.get('val_mota', 'N/A')}")
    print()
    print(THIN)
    print(f"{'  FINAL TEST RESULTS (held-out data)':^67}")
    print(THIN)
    if avg:
        print(f"  Precision      : {avg.get('precision', 'N/A')}")
        print(f"  Recall         : {avg.get('recall',    'N/A')}")
        print(f"  F1 Score       : {avg.get('f1',        'N/A')}")
        print(f"  MOTA           : {avg.get('mota',      'N/A')}")
        print(f"  MAE (px)       : {avg.get('mae_px',    'N/A')}")
        print(f"  ID Switches    : {avg.get('id_switches', 'N/A')}")
    else:
        print("  (No test results available)")
    print()
    print(THIN)
    print("  Outputs:")
    print("    Tracked videos  → outputs/tracked/")
    print("    Dashboard       → outputs/reports/dashboard.png")
    print(f"   Report JSON     → {report_path}")
    print(LINE)


# ═══════════════════════════════════════════════════════════════════════════════
#  Cell 12 — Confirm Drive Output Locations
# ═══════════════════════════════════════════════════════════════════════════════
def print_outputs(drive_base: str = DRIVE_BASE) -> None:
    """Print links to all output files saved on Google Drive."""
    print("✅ All outputs are saved to your Google Drive:")
    print(f"   📂 Tracked videos  → {drive_base}/outputs/tracked/")
    print(f"   📂 Dashboard       → {drive_base}/outputs/reports/dashboard.png")
    print(f"   📂 Report JSON     → {drive_base}/outputs/reports/experiment_report.json")
    print()
    print("   Open Google Drive in your browser to access these files.")
