from dataclasses import dataclass
from pathlib import Path
from typing import Self

import pandas as pd


class ParlayIntegrityError(ValueError):
    """Raised when a parlay DataFrame fails validation."""


@dataclass(slots=True)
class ParlayDataFrame:
    """
    Validate and manage generated and settled parlay data.

    Winner and result columns are nullable before an event is settled.
    Dates are stored as timezone-naive pandas ``datetime64[ns]`` values
    normalized to midnight.

    Settlement columns use the same suffixes as their source bets:
    ``open``, ``close1_stack``, and ``close2_stack``.
    """

    frame: pd.DataFrame

    bet_types: tuple[str, ...] = (
        "open",
        "close1",
        "close2",
        "close1_stack",
        "close2_stack",
    )

    settled_types: tuple[str, ...] = (
        "open",
        "close1_stack",
        "close2_stack",
    )

    def __post_init__(self) -> None:
        self.frame = self.frame.copy()
        self._remove_exported_index()
        self._ensure_result_columns()
        self._normalize_date()
        self.validate()

    @classmethod
    def from_generated(
        cls,
        frame: pd.DataFrame,
        event_date: str | pd.Timestamp,
    ) -> "ParlayDataFrame":
        """Create a parlay frame and attach its event date."""
        prepared = frame.copy()
        prepared["date"] = event_date
        return cls(prepared)

    @classmethod
    def from_history_file(
        cls,
        file_path: str | Path,
    ) -> Self | None:
        """Load validated parlay history, or return ``None`` if absent."""
        file_path = Path(file_path)
        if not file_path.is_file():
            return None
        return cls(pd.read_csv(file_path))

    @classmethod
    def concatenate(
        cls,
        *frames: Self | None,
    ) -> Self | None:
        """Combine available parlay frames and validate the result."""
        available_frames = [
            parlays.frame
            for parlays in frames
            if parlays is not None
        ]
        if not available_frames:
            return None

        return cls(
            pd.concat(
                available_frames,
                axis=0,
                ignore_index=True,
            )
        )

    def with_results(
        self,
        single_event: pd.DataFrame,
    ) -> "ParlayDataFrame":
        """
        Return a new parlay frame populated with available fight results.

        A betting type remains unsettled when any selected fight is
        missing a winner. Settlement values are repeated for each parlay
        leg, matching the row-oriented representation used by the
        generated parlay DataFrame.
        """
        settled = self.frame.copy()

        required_event_columns = {"winner", "winner_name"}
        missing_event_columns = (
            required_event_columns - set(single_event.columns)
        )

        if missing_event_columns:
            raise ParlayIntegrityError(
                "Single-event results are missing columns: "
                f"{sorted(missing_event_columns)}"
            )

        for bet_type in self.settled_types:
            choice_column = f"choice_fighter_bool_{bet_type}"
            index_column = f"fight_index_{bet_type}"
            fstar_column = f"parlay_fstar_{bet_type}"
            odds_column = f"parlay_odds_{bet_type}"

            required_columns = {
                choice_column,
                index_column,
                fstar_column,
                odds_column,
            }
            missing_columns = required_columns - set(settled.columns)

            if missing_columns:
                raise ParlayIntegrityError(
                    f"{bet_type} is missing settlement columns: "
                    f"{sorted(missing_columns)}"
                )

            choices = pd.to_numeric(
                settled[choice_column],
                errors="coerce",
            ).astype("Int64")

            # A model may not have generated this betting type.
            if choices.isna().all():
                continue

            indexes = pd.to_numeric(
                settled[index_column],
                errors="coerce",
            )

            if indexes.isna().any():
                raise ParlayIntegrityError(
                    f"{bet_type} has invalid fight indexes"
                )

            indexes = indexes.astype(int)

            try:
                event_rows = single_event.loc[indexes]
            except KeyError as exc:
                raise ParlayIntegrityError(
                    f"{bet_type} references a missing fight index"
                ) from exc

            winners = pd.to_numeric(
                event_rows["winner"],
                errors="coerce",
            ).astype("Int64")
            winner_names = event_rows["winner_name"].astype("string")

            settled[f"winner_bool_{bet_type}"] = pd.Series(
                winners.to_numpy(),
                index=settled.index,
                dtype="Int64",
            )
            settled[f"winner_name_{bet_type}"] = pd.Series(
                winner_names.to_numpy(),
                index=settled.index,
                dtype="string",
            )

            resolved = choices.notna() & winners.reset_index(drop=True).notna()

            if not resolved.all():
                continue

            parlay_won = bool(
                choices.reset_index(drop=True)
                .eq(winners.reset_index(drop=True))
                .all()
            )

            settled[f"win_parlay_{bet_type}"] = pd.Series(
                parlay_won,
                index=settled.index,
                dtype="boolean",
            )

            fstar = pd.to_numeric(
                settled[fstar_column],
                errors="coerce",
            )
            odds = pd.to_numeric(
                settled[odds_column],
                errors="coerce",
            )

            settled[f"net_stake_{bet_type}"] = pd.Series(
                fstar if parlay_won else -fstar,
                index=settled.index,
                dtype="Float64",
            )
            settled[f"net_odds_{bet_type}"] = pd.Series(
                odds if parlay_won else -1.0,
                index=settled.index,
                dtype="Float64",
            )

        return type(self)(settled)

    def validate(self) -> None:
        """Validate the generic date and nullable settlement contract."""
        if self.frame.empty:
            raise ParlayIntegrityError("Parlay DataFrame is empty")

        for bet_type in self.settled_types:
            winner_column = f"winner_bool_{bet_type}"
            winner_values = self.frame[winner_column].dropna()

            if not winner_values.isin([0, 1]).all():
                raise ParlayIntegrityError(
                    f"{winner_column} values must be 0, 1, or null"
                )

    def _remove_exported_index(self) -> None:
        self.frame = self.frame.loc[
            :,
            ~self.frame.columns.str.match(r"^Unnamed"),
        ].copy()

    def _ensure_result_columns(self) -> None:
        for bet_type in self.settled_types:
            self._ensure_column(
                f"winner_bool_{bet_type}",
                "Int64",
            )
            self._ensure_column(
                f"winner_name_{bet_type}",
                "string",
            )
            self._ensure_column(
                f"win_parlay_{bet_type}",
                "boolean",
            )
            self._ensure_column(
                f"net_stake_{bet_type}",
                "Float64",
            )
            self._ensure_column(
                f"net_odds_{bet_type}",
                "Float64",
            )

    def _ensure_column(self, column: str, dtype: str) -> None:
        if column not in self.frame:
            self.frame[column] = pd.Series(
                pd.NA,
                index=self.frame.index,
                dtype=dtype,
            )
            return

        try:
            self.frame[column] = self.frame[column].astype(dtype)
        except (TypeError, ValueError) as exc:
            raise ParlayIntegrityError(
                f"Unable to convert {column!r} to {dtype}"
            ) from exc

    def _normalize_date(self) -> None:
        if "date" not in self.frame:
            raise ParlayIntegrityError(
                "Parlay DataFrame requires a date column"
            )

        try:
            self.frame["date"] = pd.to_datetime(
                self.frame["date"],
                format="%Y-%m-%d",
                errors="raise",
            ).dt.normalize()
        except (TypeError, ValueError) as exc:
            raise ParlayIntegrityError(
                "Parlay dates must use YYYY-MM-DD"
            ) from exc
