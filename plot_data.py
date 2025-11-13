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

def plot_lead_vehicle(csv_path, out_name, algo, metric="MAE", save_dir="images"):

    # Plot the speed
    plt.figure(figsize=(10, 5))
    plt.plot(speed, label="Speed", linestyle="-")
    plt.xlabel("Timestep")
    plt.ylabel("Speed (m/s)")
    plt.title(f"Lead Vehicle Speed Graph")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/lead_vehicle_speed.png")

    # Plot the position
    plt.figure(figsize=(10, 5))
    plt.plot(position, label="Position", linestyle="-")
    plt.xlabel("Timestep")
    plt.ylabel("Position (m)")
    plt.title(f"Lead Vehicle Position Graph")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/lead_vehicle_position.png")

    print("Created lead vehicle speed/position with for 1200 steps.")
