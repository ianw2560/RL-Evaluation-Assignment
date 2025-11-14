import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from rl import SAFE_MAX_DIST, SAFE_MIN_DIST

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

def plot_lead_vs_ego(csv_path, out_name, algo=None, save_dir="images", dt=1.0):
    """
    Plot lead vs ego speed and position over time from a CSV with columns:
      lead_speed, ego_speed, lead_pos, ego_pos
    If positions are missing, they will be integrated from speeds (x[0]=0; x[i]=x[i-1]+v[i-1]*dt).
    """
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # --- Resolve columns (allow a couple of aliases just in case) ---
    def pick(name, *aliases):
        for c in (name, *aliases):
            if c in df.columns:
                return c
        return None

    lead_speed_col = pick("lead_speed", "lead_v", "speed_lead")
    ego_speed_col  = pick("ego_speed",  "ego_v",  "speed_ego")
    if lead_speed_col is None or ego_speed_col is None:
        raise ValueError("CSV must contain 'lead_speed' and 'ego_speed' columns.")

    lead_speed = df[lead_speed_col].to_numpy(dtype=float)
    ego_speed  = df[ego_speed_col].to_numpy(dtype=float)
    T = len(lead_speed)
    t = np.arange(T)

    # Positions (use if present; else integrate: x[0]=0, x[i]=x[i-1]+v[i-1]*dt)
    lead_pos_col = pick("lead_pos")
    ego_pos_col  = pick("ego_pos")
    if lead_pos_col is not None:
        lead_pos = df[lead_pos_col].to_numpy(dtype=float)
    else:
        lead_pos = np.cumsum(np.r_[0.0, lead_speed[:-1]]) * dt

    if ego_pos_col is not None:
        ego_pos = df[ego_pos_col].to_numpy(dtype=float)
    else:
        ego_pos = np.cumsum(np.r_[0.0, ego_speed[:-1]]) * dt

    title_suffix = f" — {algo}" if algo else ""

    # --- Speeds plot ---
    plt.figure(figsize=(10, 4))
    plt.plot(t, lead_speed, label="Lead Speed", linestyle="--")
    plt.plot(t, ego_speed,  label="Ego Speed",  linestyle="-")
    plt.xlabel("Timestep")
    plt.ylabel("Speed (m/s)")
    plt.title(f"Lead vs Ego Speed{title_suffix}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_speed = os.path.join(save_dir, f"{out_name}_speeds.png")
    plt.savefig(out_speed, dpi=300)
    plt.close()

    # --- Positions plot ---
    plt.figure(figsize=(10, 4))
    plt.plot(t, lead_pos, label="Lead Position", linestyle="--")
    plt.plot(t, ego_pos,  label="Ego Position",  linestyle="-")
    plt.xlabel("Timestep")
    plt.ylabel("Position (m)")
    plt.title(f"Lead vs Ego Position{title_suffix}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_pos = os.path.join(save_dir, f"{out_name}_positions.png")
    plt.savefig(out_pos, dpi=300)
    plt.close()

    print(f"[INFO] Saved plots:\n  {out_speed}\n  {out_pos}")

def plot_lead_vs_ego_position_difference(csv_path, out_name, algo=None, save_dir="images", dt=1.0,
                      safe_min=SAFE_MIN_DIST, safe_max=SAFE_MAX_DIST):
    """
    Plot following distance (lead_pos - ego_pos) over time.
    If positions aren't in the CSV, integrate speeds with x[0]=0; x[i]=x[i-1]+v[i-1]*dt.

    CSV columns expected:
      lead_pos, ego_pos  (preferred)
    or (fallback):
      lead_speed, ego_speed
    """
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
        # integrate from speeds if positions missing
        lead_speed_col = pick("lead_speed", "lead_v", "speed_lead")
        ego_speed_col  = pick("ego_speed",  "ego_v",  "speed_ego")
        if lead_speed_col is None or ego_speed_col is None:
            raise ValueError("CSV must have 'lead_pos' & 'ego_pos' or 'lead_speed' & 'ego_speed'.")
        lead_speed = df[lead_speed_col].to_numpy(float)
        ego_speed  = df[ego_speed_col].to_numpy(float)
        lead_pos = np.cumsum(np.r_[0.0, lead_speed[:-1]]) * dt
        ego_pos  = np.cumsum(np.r_[0.0, ego_speed[:-1]]) * dt
    else:
        lead_pos = df[lead_pos_col].to_numpy(float)
        ego_pos  = df[ego_pos_col].to_numpy(float)

    gap = lead_pos - ego_pos  # positive = ego is behind (good), negative = ego ahead
    t = np.arange(len(gap))

    # % time inside the safe band
    in_band = (gap >= safe_min) & (gap <= safe_max)
    pct_in_band = 100.0 * in_band.mean()

    title_suffix = f" — {algo}" if algo else ""

    plt.figure(figsize=(10, 4))
    plt.plot(t, gap, label="Following Distance (lead − ego)")
    # safe band shading + guide lines
    plt.fill_between(t, safe_min, safe_max, alpha=0.1, label=f"Safe band [{safe_min:.0f}, {safe_max:.0f}] m")
    plt.axhline(safe_min, linestyle="--", linewidth=1, alpha=0.6)
    plt.axhline(safe_max, linestyle="--", linewidth=1, alpha=0.6)
    plt.axhline(0.0,      linestyle=":",  linewidth=1, alpha=0.5, label="0 m (ego alongside/ ahead)")

    plt.xlabel("Timestep")
    plt.ylabel("Distance (m)")
    plt.title(f"Following Distance Over Time{title_suffix}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{out_name}_gap.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[INFO] Saved gap plot: {out_path} | % in-band: {pct_in_band:.1f}%")


def plot_speed_difference(csv_path, out_name, algo=None, save_dir="images",
                          band=None, smoothing=None):
    """
    Plot speed difference over time: Δv = ego_speed - lead_speed.

    CSV must have:
      - 'lead_speed' and 'ego_speed'  (aliases: lead_v/speed_lead, ego_v/speed_ego)

    Args:
        csv_path (str): Path to time-series CSV (written in test()).
        out_name (str): Base name for output file (no extension).
        algo (str|None): Optional label in the title (e.g., 'SAC').
        save_dir (str): Output directory for the image.
        band (float|None): If set (m/s), draws ±band region and reports % time within.
        smoothing (int|None): Rolling window (timesteps) for optional smoothing.
    """
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
    dv = ego_speed - lead_speed   # positive => ego faster than lead
    t = np.arange(len(dv))

    if smoothing and smoothing > 1:
        # simple rolling mean smoothing (pad with nan at start, drop for plotting)
        dv_sm = pd.Series(dv).rolling(window=smoothing, min_periods=1).mean().to_numpy()
    else:
        dv_sm = dv

    # Stats
    mae = float(np.mean(np.abs(dv)))
    mean = float(np.mean(dv))
    std = float(np.std(dv))

    title_suffix = f" — {algo}" if algo else ""

    plt.figure(figsize=(10, 4))
    plt.plot(t, dv_sm, label="Δv = Ego − Lead", linewidth=1.5)
    plt.axhline(0.0, linestyle="--", linewidth=1, alpha=0.7, label="0 m/s")

    pct_in_band = None
    if band is not None and band > 0:
        plt.fill_between(t, -band, band, alpha=0.1, label=f"±{band:g} m/s band")
        plt.axhline(band,  linestyle=":", linewidth=1, alpha=0.6)
        plt.axhline(-band, linestyle=":", linewidth=1, alpha=0.6)
        pct_in_band = 100.0 * np.mean((dv >= -band) & (dv <= band))

    plt.xlabel("Timestep")
    plt.ylabel("Speed Difference Δv (m/s)")
    plt.title(f"Speed Difference Over Time{title_suffix}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{out_name}_speed_diff.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    msg = f"[INFO] Saved speed-diff plot: {out_path} | MAE={mae:.3f} m/s, mean={mean:.3f}, std={std:.3f}"
    if pct_in_band is not None:
        msg += f", % within ±{band:g} m/s = {pct_in_band:.1f}%"
    print(msg)