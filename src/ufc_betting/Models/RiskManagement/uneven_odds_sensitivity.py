"""Sample unique fight-specific movements using precomputed fixed-model EVs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ufc_betting.config import settings


def sample_movements(n_fights, n_samples, rng, levels=101):
    """Mixture over movement centers and dispersion, rounded to the saved grid."""
    if n_fights < 1 or n_samples < levels or levels < 2:
        raise ValueError("Need fights and at least one sample per grid level")
    # Uniform reference vectors are included exactly once.
    vectors = np.repeat(np.arange(levels)[:, None], n_fights, axis=1)
    target = min(n_samples, levels ** n_fights)
    for _ in range(40):
        if len(vectors) >= target:
            break
        count = max(2 * (target - len(vectors)), 1000)
        mean = rng.uniform(.001, .999, (count, 1))
        concentration = rng.choice([2., 10., 50.], size=(count, 1))
        draws = rng.beta(mean * concentration, (1 - mean) * concentration,
                         size=(count, n_fights))
        draws = np.rint(draws * (levels - 1)).astype(int)
        candidates = np.unique(np.concatenate([vectors, draws]), axis=0)
        # Keep all reference vectors; randomly choose other unique vectors.
        nonuniform = candidates[~np.all(candidates == candidates[:, :1], axis=1)]
        take = min(target - levels, len(nonuniform))
        vectors = np.concatenate([np.repeat(np.arange(levels)[:, None], n_fights, axis=1),
                                  nonuniform[rng.choice(len(nonuniform), take, replace=False)]])
    return vectors


def scenario_totals(contributions, vectors):
    """contributions shape: fights x grid; vectors shape: scenarios x fights."""
    return contributions[np.arange(contributions.shape[0])[None, :], vectors].sum(axis=1)


def run(input_dir=None, output_dir=None, samples=5000, seed=42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    input_dir = Path(input_dir) if input_dir else settings.data_dir / "model_results" / "event_odds_sensitivity"
    output_dir = Path(output_dir) if output_dir else input_dir / "uneven_movements"
    details = pd.read_csv(input_dir / "fight_shift_details.csv", parse_dates=["date"])
    uniform = pd.read_csv(input_dir / "event_curves.csv", parse_dates=["date"])
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_bands, summaries, orders = [], [], {}
    plot_data = []
    # Stream one compressed CSV to avoid storing all event samples in memory.
    import gzip
    with gzip.open(output_dir / "sampled_scenarios.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        first = True
        for date, frame in details.groupby("date", sort=True):
            table = frame.pivot(index=["fighter_red", "fighter_blue"], columns="shift_pct",
                                values="expected_bankroll_return_pct").sort_index(axis=1)
            exposure = frame.pivot(index=["fighter_red", "fighter_blue"], columns="shift_pct",
                                   values="fstar").reindex(index=table.index, columns=table.columns)
            if table.isna().any().any() or exposure.isna().any().any():
                raise ValueError("Incomplete fight movement grid")
            grid = table.columns.to_numpy(float)
            if not np.allclose(grid, np.linspace(0, 100, len(grid))):
                raise ValueError("Expected evenly spaced 0-100% grid")
            reference = uniform.loc[uniform.date.eq(date)].sort_values("shift_pct")
            np.testing.assert_allclose(table.sum(axis=0), reference.expected_bankroll_return_pct, atol=1e-9)
            target = float(reference.open_target_pct.iloc[0])
            ratio_scale = 100 / target if target > 0 else np.nan
            vectors = sample_movements(len(table), samples, rng, len(grid))
            ev = scenario_totals(table.to_numpy(), vectors)
            stakes = scenario_totals(exposure.to_numpy(), vectors)
            movement = grid[vectors]
            result = pd.DataFrame({"date": date, "scenario_id": np.arange(len(vectors)),
                "mean_movement_pct": movement.mean(axis=1),
                "movement_std_pct": movement.std(axis=1),
                "expected_bankroll_return_pct": ev, "pct_opening_ev": ratio_scale * ev,
                "stake_fraction": stakes,
                "is_uniform": np.all(vectors == vectors[:, :1], axis=1),
                "fight_shift_grid_indices": ["|".join(map(str, row)) for row in vectors]})
            result.to_csv(handle, index=False, header=first)
            first = False
            orders[str(date.date())] = [list(pair) for pair in table.index]
            result["movement_bin"] = np.minimum((result.mean_movement_pct // 5).astype(int), 19) * 5
            rows = []
            for low, group in result.groupby("movement_bin"):
                values = group.pct_opening_ev if target > 0 else group.expected_bankroll_return_pct
                rows.append({"date": date, "bin_low_pct": low, "bin_high_pct": low + 5,
                    "units": "percent_of_opening_ev" if target > 0 else "expected_bankroll_return_pct",
                    "n_scenarios": len(group), "sample_min": values.min(), "p05": values.quantile(.05),
                    "median": values.median(), "p95": values.quantile(.95), "sample_max": values.max()})
            bands = pd.DataFrame(rows)
            all_bands.append(bands)
            summaries.append({"date": date, "n_fights": len(table), "unique_scenarios": len(vectors),
                "opening_expected_return_pct": target,
                "uniform_min_pct_opening_ev": ratio_scale * reference.expected_bankroll_return_pct.min(),
                "uniform_max_pct_opening_ev": ratio_scale * reference.expected_bankroll_return_pct.max(),
                "sample_min_pct_opening_ev": result.pct_opening_ev.min(),
                "sample_max_pct_opening_ev": result.pct_opening_ev.max(),
                "exact_grid_global_min_pct_opening_ev": ratio_scale * table.min(axis=1).sum(),
                "exact_grid_global_max_pct_opening_ev": ratio_scale * table.max(axis=1).sum(),
                "sample_min_expected_return_pct": ev.min(), "sample_max_expected_return_pct": ev.max(),
                "maximum_stake_fraction": stakes.max()})
            plot_data.append((date, len(table), bands, reference, target))
    pd.concat(all_bands, ignore_index=True).to_csv(output_dir / "scenario_bands.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "event_summary.csv", index=False)
    (output_dir / "fight_order.json").write_text(json.dumps(orders, indent=2), encoding="utf-8")

    def draw(ax, item):
        date, count, bands, reference, target = item
        x = bands.bin_low_pct.to_numpy() + 2.5
        ax.fill_between(x, bands.sample_min, bands.sample_max, color="#2366af", alpha=.13, label="Sampled min-max")
        ax.fill_between(x, bands.p05, bands.p95, color="#2366af", alpha=.3, label="Sampled 5th-95th percentiles")
        ax.plot(x, bands["median"], color="#2366af", lw=1.5, label="Sample median")
        ax.plot(reference.shift_pct, (100 / target if target > 0 else 1) * reference.expected_bankroll_return_pct,
                color="#c25916", lw=2, label="Uniform movement")
        ax.axhline(100 if target > 0 else target, color="black", ls="--", lw=1, label="Opening EV")
        ax.set(xlim=(0, 100), title=f"{date:%Y-%m-%d} | {count} fights",
               xlabel="Average fight movement toward opening (%)", ylabel="Event EV / opening EV (%)" if target > 0 else "Expected bankroll return (%) | opening EV = 0")
        ax.grid(alpha=.18)
        ax.legend(fontsize=7, loc="best")

    def page(items):
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        for ax, item in zip(axes.flat, items):
            draw(ax, item)
        for ax in list(axes.flat)[len(items):]:
            ax.set_visible(False)
        fig.suptitle("Uneven fight-level line movements | scenario sensitivity", fontsize=17)
        fig.supxlabel("Bands group scenarios into 5-point average-movement bins. Percentiles describe the sampling design, not forecast probabilities.\n"
                      "Fixed saved models; recomputed Kelly stakes. Existing uncapped exposure retained. Both sides of each fight move together.", fontsize=10)
        return fig
    with PdfPages(output_dir / "all_event_scenario_bands.pdf") as pdf:
        for start in range(0, len(plot_data), 6):
            fig = page(plot_data[start:start + 6])
            pdf.savefig(fig)
            plt.close(fig)
    fig = page(plot_data[-6:])
    fig.savefig(output_dir / "latest_event_scenario_bands.png", dpi=150)
    plt.close(fig)
    metadata = {"seed": seed, "samples_requested_per_event": samples, "events": len(summaries),
        "grid": "Inherited evenly spaced 0-100% movement grid",
        "sampling": "Beta mixtures: centers uniform in [0.001,0.999], concentrations 2,10,50; rounded to grid, duplicate vectors removed; all uniform reference vectors included.",
        "deduplication": "Exact ordered shift vector within event; swapping shifts across distinct fights is a distinct scenario.",
        "aggregation": "Sum cached per-fight expected bankroll contributions. No shared exposure constraint; verified uniform sums against prior event curves.",
        "bands": "5-point bins of mean movement; sampled percentiles and extrema, not confidence intervals or exhaustive bin-specific bounds.",
        "global_bounds": "Sum each fight's grid minimum/maximum; exact for the discrete grid with independent per-bet stakes, not conditional on average movement.",
        "scenario_vectors": "Indices into grid; see fight_order.json for the ordered fighters.",
        "scope": "Within-event sensitivity; no cross-event bankroll probability distribution inferred."}
    metadata["zero_opening_ev"] = "Ratios are undefined and saved as missing. Those event plots show expected bankroll return percentages instead."
    upstream = json.loads((input_dir / "methodology.json").read_text(encoding="utf-8"))
    metadata["risk_parameters"] = {key: upstream[key] for key in ("mdd_open", "mdd_close1", "N_open", "N_close1")}
    (output_dir / "methodology.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary = pd.DataFrame(summaries)
    print(summary.tail(6).to_string(index=False))
    print(f"Total unique scenarios: {summary.unique_scenarios.sum()}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    run(**vars(parser.parse_args()))
