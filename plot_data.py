import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from rl import MAX_DIST, MIN_DIST

# ------------------------------------------------------------------------
# Functions for plotting data
# ------------------------------------------------------------------------

def plot_batchsize_vs_metric(csv_path, out_name, metric="MAE", save_dir="images", figsize=(7, 4)):

    df = pd.read_csv(csv_path)
    os.makedirs(save_dir, exist_ok=True)

    algos = df["Algorithm"].unique()
    colors = plt.cm.tab10.colors

    plt.figure(figsize=figsize)
    for i, algo in enumerate(algos):
        sub = df[df["Algorithm"] == algo].copy()
        sub = sub.groupby("BatchSize", as_index=False)[metric].mean().sort_values("BatchSize")
        plt.plot(
            sub["BatchSize"], sub[metric],
            marker="o", linestyle="-", label=algo, color=colors[i % len(colors)]
        )

    # Set xticks explicitly to tested batch sizes
    batch_values = sorted(df["BatchSize"].unique())
    plt.xticks(batch_values, [str(int(v)) for v in batch_values])

    plt.title(f"{metric} vs Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel(metric)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{out_name}.png"), dpi=300)
    plt.close()

    print(f"[INFO] Saved plot to {save_dir}/ for metric '{metric}'")

def plot_episodelength_vs_metric(csv_path, out_name, metric="MAE", save_dir="images", figsize=(7, 4)):

    df = pd.read_csv(csv_path)
    os.makedirs(save_dir, exist_ok=True)

    algos = df["Algorithm"].unique()
    colors = plt.cm.tab10.colors

    plt.figure(figsize=figsize)
    for i, algo in enumerate(algos):
        sub = df[df["Algorithm"] == algo].copy()
        sub = sub.groupby("EpisodeLength", as_index=False)[metric].mean().sort_values("EpisodeLength")
        plt.plot(
            sub["EpisodeLength"], sub[metric],
            marker="o", linestyle="-", label=algo, color=colors[i % len(colors)]
        )

    # Set xticks explicitly to tested batch sizes
    batch_values = sorted(df["EpisodeLength"].unique())
    plt.xticks(batch_values, [str(int(v)) for v in batch_values])

    plt.title(f"{metric} vs Episode Length")
    plt.xlabel("Episode Length")
    plt.ylabel(metric)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{out_name}.png"), dpi=300)
    plt.close()

    print(f"[INFO] Saved plot to {save_dir}/ for metric '{metric}'")


def plot_rewardtype_vs_metric(csv_path, out_name, metric="MAE", save_dir="images", figsize=(10, 8), clip_outliers=True, percentile_clip=95, annotate=True):

    df = pd.read_csv(csv_path)
    os.makedirs(save_dir, exist_ok=True)

    if "RewardType" not in df.columns:
        raise ValueError("CSV file must contain a 'RewardType' column.")

    algos = sorted(df["Algorithm"].unique())
    reward_types = sorted(df["RewardType"].unique())
    colors = plt.cm.tab10.colors

    grouped = df.groupby(["RewardType", "Algorithm"], as_index=False)[metric].mean()

    # Clip outliers globally (helps if one extreme value causes blowout)
    if clip_outliers:
        clip_val = np.percentile(grouped[metric], percentile_clip)
        grouped[metric] = np.clip(grouped[metric], None, clip_val)
        print(f"[INFO] Clipped values above {clip_val:.2f} for {metric}")

    # Create subplot grid (2x2)
    n_subplots = len(reward_types)
    n_rows, n_cols = 2, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for idx, reward in enumerate(reward_types):
        ax = axes[idx]
        sub = grouped[grouped["RewardType"] == reward]

        x = np.arange(len(algos))
        bar_width = 0.6

        bars = ax.bar(
            x,
            sub[metric],
            color=[colors[i % len(colors)] for i in range(len(algos))],
            alpha=0.9,
            width=bar_width,
        )

        # Annotate values
        if annotate:
            for bar, val in zip(bars, sub[metric]):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + 0.02 * max(sub[metric]),
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8
                    )

        # Local scaling per subplot
        local_max = max(sub[metric])
        ax.set_ylim(0, local_max * 1.15)

        ax.set_title(f"{reward} Reward", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=45, fontsize=9)
        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Remove empty subplots (if < 4 reward types)
    for j in range(len(reward_types), len(axes)):
        fig.delaxes(axes[j])

    # Overall title
    fig.suptitle(f"{metric} Comparison Across Reward Functions", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save combined figure
    out_path = os.path.join(save_dir, f"{out_name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[INFO] Saved multi-subplot comparison to {out_path}")

def plot_entropy_vs_metric(csv_path, out_name, algo, metric="MAE", save_dir="images", figsize=(7, 4)):

    df = pd.read_csv(csv_path)
    os.makedirs(save_dir, exist_ok=True)

    # Filter by algorithm
    sub = df[df["Algorithm"] == algo].copy()

    # Handle missing or mixed types in EntCoef column
    sub["EntCoef"] = sub["EntCoef"].astype(str)
    sub = sub.groupby("EntCoef", as_index=False)[metric].mean().sort_values("EntCoef")

    # Plot
    plt.figure(figsize=figsize)
    bars = plt.bar(sub["EntCoef"], sub[metric], alpha=0.75, label=algo, color="C0")

    # Add horizontal text labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height * 1.01,  # a bit above the bar
            f"{height:.3f}",
            ha="center", va="bottom", fontsize=9, rotation=0
        )

    plt.title(f"{algo}: {metric} vs Entropy Coefficient")
    plt.xlabel("Entropy Coefficient")
    plt.ylabel(metric)
    plt.grid(alpha=0.3, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{out_name}.png"), dpi=300)
    plt.close()

    print(f"[INFO] Saved plot to {save_dir}/{out_name}.png for metric '{metric}'")

def plot_lead_vs_ego(csv_path, out_name, algo, save_dir="images"):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # --- Resolve columns (allow a couple of aliases just in case) ---
    def pick(name, *aliases):
        for c in (name, *aliases):
            if c in df.columns:
                return c
        return None
    
    # Get speed values
    lead_speed_col = pick("lead_speed", "lead_v", "speed_lead")
    ego_speed_col  = pick("ego_speed",  "ego_v",  "speed_ego")
    if lead_speed_col is None or ego_speed_col is None:
        raise ValueError("Missing 'lead_speed' and 'ego_speed' columns.")

    lead_speed = df[lead_speed_col].to_numpy(dtype=float)
    ego_speed  = df[ego_speed_col].to_numpy(dtype=float)

    # Get time values
    T = len(lead_speed)
    t = np.arange(T)

    # Get position values
    lead_pos_col = pick("lead_pos")
    ego_pos_col  = pick("ego_pos")
    if lead_pos_col is None or ego_pos_col is None:
        raise ValueError("Missing 'lead_speed' and 'ego_speed' columns.")

    lead_pos = df[lead_pos_col].to_numpy(dtype=float)
    ego_pos = df[ego_pos_col].to_numpy(dtype=float)

    # Get jerk values
    jerk_col = pick("jerk")
    if jerk_col is None:
        raise ValueError("Missing 'jerk' columns")

    jerk = df[jerk_col].to_numpy(dtype=float)

    # jerk length might match T; if not, pad/trim to T for plotting
    if len(jerk) != T:
        jerk = np.resize(jerk, T)




    # Plot speed
    plt.figure(figsize=(10, 4))
    plt.plot(t, lead_speed, label="Lead Speed", linestyle="--")
    plt.plot(t, ego_speed,  label="Ego Speed",  linestyle="-")
    plt.xlabel("Timestep")
    plt.ylabel("Speed (m/s)")
    plt.title(f"Lead vs Ego Speed - {algo}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_speed = os.path.join(save_dir, f"{out_name}_speeds.png")
    plt.savefig(out_speed, dpi=300)
    plt.close()

    # Plot position
    plt.figure(figsize=(10, 4))
    plt.plot(t, lead_pos, label="Lead Position", linestyle="--")
    plt.plot(t, ego_pos,  label="Ego Position",  linestyle="-")
    plt.xlabel("Timestep")
    plt.ylabel("Position (m)")
    plt.title(f"Lead Vehicle vs Ego Vehicle Position - {algo}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_pos = os.path.join(save_dir, f"{out_name}_positions.png")
    plt.savefig(out_pos, dpi=300)
    plt.close()

    # Plot ego vehicle jerk
    plt.figure(figsize=(10, 4))
    plt.plot(t, jerk, label="Ego Jerk", linewidth=1.2)
    plt.axhline(0.0, linestyle="--", linewidth=1, alpha=0.6)
    plt.xlabel("Timestep")
    plt.ylabel("Jerk (m/s³)")
    plt.title(f"Ego Vehicle Jerk - {algo}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_jerk = os.path.join(save_dir, f"{out_name}_jerks.png")
    plt.savefig(out_jerk, dpi=300)
    plt.close()

    print(f"[INFO] Saved lead vs ego position plot: {out_speed}")
    print(f"[INFO] Saved lead vs ego speed plot: {out_speed}")
    print(f"[INFO] Saved ego jerk plot: {out_jerk}")

def plot_position_difference(csv_path, out_name, algo=None, save_dir="images", dt=1.0):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    def pick(name, *aliases):
        for c in (name, *aliases):
            if c in df.columns:
                return c
        return None

    lead_pos_col = pick("lead_pos")
    ego_pos_col  = pick("ego_pos")
    if lead_pos_col is None or ego_pos_col is None:
        raise ValueError("Missing 'lead_speed' and 'ego_speed' columns.")
    
    lead_pos = df[lead_pos_col].to_numpy(float)
    ego_pos  = df[ego_pos_col].to_numpy(float)

    # Calculate position difference
    pos_difference = lead_pos - ego_pos
    t = np.arange(len(pos_difference))

    # Plot following distance over time
    plt.figure(figsize=(10, 4))
    plt.plot(t, pos_difference, label="Following Distance")
    # safe band shading + guide lines
    plt.fill_between(t, MIN_DIST, MAX_DIST, alpha=0.1)
    plt.axhline(MIN_DIST, linestyle="--", linewidth=1, alpha=0.6)
    plt.axhline(MAX_DIST, linestyle="--", linewidth=1, alpha=0.6)

    plt.xlabel("Timestep")
    plt.ylabel("Distance (m)")
    plt.title(f"Following Distance - {algo}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{out_name}_position_difference.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[INFO] Saved position difference plot: {out_path}")


def plot_speed_difference(csv_path, out_name, algo=None, save_dir="images", band=None):

    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    def pick(name, *aliases):
        for c in (name, *aliases):
            if c in df.columns:
                return c
        return None

    lead_speed_col = pick("lead_speed", "lead_v", "speed_lead")
    ego_speed_col  = pick("ego_speed",  "ego_v",  "speed_ego")
    if lead_speed_col is None or ego_speed_col is None:
        raise ValueError("CSV must contain 'lead_speed' and 'ego_speed' columns (or aliases).")

    lead_speed = df[lead_speed_col].to_numpy(float)
    ego_speed  = df[ego_speed_col].to_numpy(float)

    # Calculate speed difference
    delta_v = ego_speed - lead_speed
    t = np.arange(len(delta_v))

    # Stats
    # mae = float(np.mean(np.abs(delta_v)))
    # mean = float(np.mean(delta_v))
    # std = float(np.std(delta_v))

    plt.figure(figsize=(10, 4))
    plt.plot(t, delta_v, label="$\Delta$v", linewidth=1.5)

    pct_in_band = None
    if band is not None and band > 0:
        plt.fill_between(t, -band, band, alpha=0.1)
        plt.axhline(band,  linestyle=":", linewidth=1, alpha=0.6)
        plt.axhline(-band, linestyle=":", linewidth=1, alpha=0.6)
        pct_in_band = 100.0 * np.mean((delta_v >= -band) & (delta_v <= band))

    plt.xlabel("Timestep")
    plt.ylabel("Speed Difference $\Delta$v (m/s)")
    plt.title(f"Speed Difference - {algo}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{out_name}_speed_difference.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[INFO] Saved speed difference plot: {out_path}")

    # if pct_in_band is not None:
    #     print(f", % within ±{band:g} m/s = {pct_in_band:.1f}%")