import pandas as pd
from pathlib import Path

from ufc_betting.config import settings
from ufc_betting.BettingStrategy.MonteCarlo.ordered_sampling import generate_ordered_paths

def generate_path_run(
    validation_path, output_path, *, n_paths=100, sample_prices=False,
    days_out=(7, 6, 5), seed=42, sample_fights=True,
    strict_numbered_months=True,
):
    """Save shared fights and optional intermediate prices, without model inference."""
    fights = pd.read_csv(validation_path)
    fights["date"] = pd.to_datetime(fights["date"]).dt.normalize()
    print(f"Source: {len(fights):,} fights across {fights.date.nunique()} events")
    generation_options = {}
    if sample_prices:
        generation_options = dict(
            days_out=days_out,
            density_models={day: {
                stage: settings.saved_models_dir / f"conditional_price_density_{stage}_{day}_days.joblib"
                for stage in ("close1", "close2")
            } for day in days_out},
        )
    paths = generate_ordered_paths(
        fights, n_paths=n_paths, output_path=Path(output_path),
        odds_type=None, sample_prices=sample_prices,
        random_state=seed, progress_every=10, storage="compact",
        sample_fights=sample_fights, strict_numbered_months=strict_numbered_months,
        **generation_options,
    )
    print(f"Saved {len(paths)} paths to {output_path}")
    return paths

if __name__ == "__main__":
    # Edit these arguments, then run this file in your editor.
    # Use a NEW output directory for each saved run.
    # paths = generate_path_run(
    #     validation_path=settings.split_feats_dir / "val.csv",
    #     output_path=settings.data_dir / "bootstrapped_paths" / " sampled_events_prices_765_val",
    #     n_paths=10000,
    #     sample_fights=True,             # True: resample fights. False: shuffle intact cards.
    #     sample_prices=True,            # True: sample prices. False: keep recorded prices.
    #     days_out=(7, 6, 5),              # Used only when sample_prices=True.
    #     strict_numbered_months=False,   # False: allow extra numbered cards in some months.                                # True: require exactly one numbered card per month.
    #     seed=42,
    # )
    # paths.close()  # Release the saved arrays' file handles after generation.


    paths = generate_path_run(
        validation_path=settings.split_feats_dir / "val.csv",
        output_path=(
            settings.data_dir
            / "bootstrapped_paths"
            / "sampled_fights_recorded_prices_val"
        ),
        n_paths=3000,
        sample_fights=True,
        sample_prices=False,
        strict_numbered_months=False,
        seed=42,
    )
    paths.close()