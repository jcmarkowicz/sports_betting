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
        """Combine parlay frames, replacing older events by date."""
        available_frames = [
            parlays.frame
            for parlays in frames
            if parlays is not None
        ]
        if not available_frames:
            return None

        combined = available_frames[0].drop_duplicates(
            keep="last",
        ).copy()

        for incoming in available_frames[1:]:
            incoming = incoming.drop_duplicates(
                keep="last",
            ).copy()
            updated_dates = incoming["date"].dropna().unique()
            combined = combined.loc[
                ~combined["date"].isin(updated_dates)
            ]
            combined = pd.concat(
                [combined, incoming],
                axis=0,
                ignore_index=True,
            )

        return cls(combined)

    def with_results(
        self,
        single_event: pd.DataFrame,
    ) -> "ParlayDataFrame":
        """
        Return a new parlay frame populated with available fight results.

        A betting type is excluded from settlement when any selected fight
        is a draw, no-contest, null result, or other non-binary outcome.
        Other betting types may still settle when they reference unaffected
        fights. Settlement values are repeated for each parlay leg, matching
        the row-oriented representation used by the generated DataFrame.
        """
        settled = self.frame.copy()

        required_event_columns = {
            "fighter_red",
            "fighter_blue",
            "winner",
            "winner_name",
        }
        missing_event_columns = (
            required_event_columns - set(single_event.columns)
        )

        if missing_event_columns:
            raise ParlayIntegrityError(
                "Single-event results are missing columns: "
                f"{sorted(missing_event_columns)}"
            )

        event_rows = single_event.reset_index(drop=True).copy()
        for color in ("red", "blue"):
            event_rows[f"_fighter_{color}_key"] = (
                event_rows[f"fighter_{color}"]
                .astype("string")
                .str.strip()
                .str.casefold()
            )
        settled_any = False

        for bet_type in self.settled_types:
            name_column = f"choice_fighter_name_{bet_type}"
            choice_column = f"choice_fighter_bool_{bet_type}"
            fstar_column = f"parlay_fstar_{bet_type}"
            odds_column = f"parlay_odds_{bet_type}"

            required_columns = {
                name_column,
                choice_column,
                fstar_column,
                odds_column,
            }
            missing_columns = required_columns - set(settled.columns)

            if missing_columns:
                raise ParlayIntegrityError(
                    f"{bet_type} is missing settlement columns: "
                    f"{sorted(missing_columns)}"
                )

            raw_choices = settled[choice_column]
            choices = pd.to_numeric(raw_choices, errors="coerce")
            invalid_choices = (
                raw_choices.notna()
                & (choices.isna() | ~choices.isin([0, 1]))
            )
            if invalid_choices.any():
                raise ParlayIntegrityError(
                    f"{choice_column} values must be 0, 1, or null"
                )
            choices = choices.astype("Int64")

            # A model may not have generated this betting type.
            if choices.isna().all():
                continue

            if choices.isna().any():
                raise ParlayIntegrityError(
                    f"{bet_type} has incomplete parlay legs"
                )

            choice_names = (
                settled[name_column]
                .astype("string")
                .str.strip()
                .str.casefold()
            )
            if choice_names.isna().any():
                raise ParlayIntegrityError(
                    f"{bet_type} has missing fighter names"
                )

            selected_indexes: list[int] = []
            for choice, choice_name in zip(choices, choice_names):
                color = "red" if choice == 1 else "blue"
                matches = event_rows[f"_fighter_{color}_key"].eq(
                    choice_name
                )
                match_count = int(matches.sum())
                if match_count == 0:
                    raise ParlayIntegrityError(
                        f"{bet_type} fighter {choice_name!r} is missing "
                        "from single-event results"
                    )
                if match_count > 1:
                    raise ParlayIntegrityError(
                        f"{bet_type} fighter {choice_name!r} matches "
                        "multiple single-event fights"
                    )
                selected_indexes.append(int(matches.idxmax()))

            if len(set(selected_indexes)) != len(selected_indexes):
                raise ParlayIntegrityError(
                    f"{bet_type} contains duplicate parlay legs"
                )

            selected_results = event_rows.loc[selected_indexes]

            winners = pd.to_numeric(
                selected_results["winner"],
                errors="coerce",
            )
            nonnumeric_winners = (
                selected_results["winner"].notna() & winners.isna()
            )
            if nonnumeric_winners.any():
                invalid_values = (
                    selected_results.loc[nonnumeric_winners, "winner"]
                    .unique()
                    .tolist()
                )
                raise ParlayIntegrityError(
                    "Single-event results contain nonnumeric winner values: "
                    f"{invalid_values}"
                )

            # One tied, missing, or otherwise non-binary leg voids this
            # parlay type without affecting other independently selected types.
            if not winners.isin([0, 1]).all():
                self._clear_settlement(settled, bet_type)
                continue

            winners = winners.astype("Int64")
            winner_names = selected_results["winner_name"].astype("string")

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
            if fstar.isna().any() or fstar.lt(0).any():
                raise ParlayIntegrityError(
                    f"{fstar_column} must contain nonnegative numeric values"
                )
            if odds.isna().any() or odds.le(0).any():
                raise ParlayIntegrityError(
                    f"{odds_column} must contain positive numeric values"
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

            settled_any = True

        if not settled_any:
            return type(self)(settled.iloc[0:0].copy())

        return type(self)(settled)

    def validate(self) -> None:
        """Validate the generic date and nullable settlement contract."""
        if self.frame.empty:
            return

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
            winner_column = f"winner_bool_{bet_type}"
            invalid_winner = pd.Series(False, index=self.frame.index)

            if winner_column not in self.frame:
                self.frame[winner_column] = pd.Series(
                    pd.NA,
                    index=self.frame.index,
                    dtype="Int64",
                )
            else:
                raw_winners = self.frame[winner_column]
                numeric_winners = pd.to_numeric(
                    raw_winners,
                    errors="coerce",
                )
                invalid_winner = (
                    raw_winners.notna()
                    & (
                        numeric_winners.isna()
                        | ~numeric_winners.isin([0, 1])
                    )
                )
                self.frame[winner_column] = numeric_winners.where(
                    ~invalid_winner
                ).astype("Int64")

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

            if invalid_winner.any():
                affected_rows = invalid_winner
                if "date" in self.frame:
                    affected_dates = self.frame.loc[
                        invalid_winner,
                        "date",
                    ]
                    affected_rows = self.frame["date"].isin(affected_dates)
                self._clear_settlement(
                    self.frame,
                    bet_type,
                    affected_rows,
                )

    def _clear_settlement(
        self,
        frame: pd.DataFrame,
        bet_type: str,
        rows: pd.Series | None = None,
    ) -> None:
        """Clear every derived result when a parlay type is void."""
        if rows is None:
            rows = pd.Series(True, index=frame.index)

        for prefix in (
            "winner_bool",
            "winner_name",
            "win_parlay",
            "net_stake",
            "net_odds",
        ):
            frame.loc[rows, f"{prefix}_{bet_type}"] = pd.NA

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
