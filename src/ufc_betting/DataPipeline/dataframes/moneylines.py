from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import pandas as pd


class MoneylineIntegrityError(ValueError):
    """Raised when a moneyline DataFrame fails validation."""


@dataclass(slots=True)
class MoneylineDataFrame:
    """
    Validate and manage generated and settled moneyline data.

    Prediction values may be null when a model cannot make a prediction.
    Winner and result values may be null before an event is settled.

    Dates are stored as timezone-naive pandas ``datetime64[ns]`` values
    normalized to midnight. Settlement fields retain the canonical
    ``open``, ``close1_stack``, and ``close2_stack`` suffixes.

    Net returns use the Kelly fraction as ``net_stake``. ``chosen_odds``
    stores the selected American line, while ``net_odds`` stores the settled
    net decimal odds: decimal odds minus one for a win and -1 for a loss.
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

    odds_type_by_settled_type: dict[str, str] = field(
        default_factory=lambda: {
            "open": "open",
            "close1_stack": "close1",
            "close2_stack": "close2",
        }
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
    ) -> "MoneylineDataFrame":
        """Create a validated moneyline wrapper from generated bets."""
        return cls(frame.copy())

    @classmethod
    def from_history_file(
        cls,
        file_path: str | Path,
    ) -> Self | None:
        """Load validated moneyline history, or return ``None`` if absent."""
        file_path = Path(file_path)
        if not file_path.is_file():
            return None
        return cls(pd.read_csv(file_path))

    @classmethod
    def concatenate(
        cls,
        *frames: Self | None,
    ) -> Self | None:
        """Combine available moneyline frames and validate the result."""
        available_frames = [
            moneylines.frame
            for moneylines in frames
            if moneylines is not None
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
    ) -> "MoneylineDataFrame":
        """
        Return a new moneyline frame populated with available results.

        Rows remain unsettled when the actual winner or the corresponding
        model prediction is unavailable.
        """
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
            raise MoneylineIntegrityError(
                "Single-event results are missing columns: "
                f"{sorted(missing_event_columns)}"
            )

        if len(single_event) != len(self.frame):
            raise MoneylineIntegrityError(
                "Single-event results and moneyline bets have "
                "different row counts"
            )

        event_rows = single_event.reset_index(drop=True).copy()
        settled = self.frame.reset_index(drop=True).copy()

        self._validate_fighter_alignment(settled, event_rows)

        winners = pd.to_numeric(
            event_rows["winner"],
            errors="coerce",
        ).astype("Int64")
        winner_names = event_rows["winner_name"].astype("string")

        settled["winner_bool"] = winners
        settled["winner_name"] = winner_names

        for bet_type in self.settled_types:
            prediction_column = f"pred_winner_{bet_type}"
            odds_type = self.odds_type_by_settled_type[bet_type]

            predictions = pd.to_numeric(
                settled[prediction_column],
                errors="coerce",
            ).astype("Int64")

            red_odds = pd.to_numeric(
                settled[f"{odds_type}_red"],
                errors="coerce",
            ).astype("Float64")
            blue_odds = pd.to_numeric(
                settled[f"{odds_type}_blue"],
                errors="coerce",
            ).astype("Float64")

            chosen_odds = red_odds.where(
                predictions.eq(1),
                blue_odds.where(predictions.eq(0)),
            )
            fstar = pd.to_numeric(
                settled[f"fstar_{bet_type}"],
                errors="coerce",
            ).astype("Float64")

            resolved = predictions.notna() & winners.notna()
            win_bet = predictions.eq(winners).where(resolved)

            decimal_odds = (1 + chosen_odds / 100).where(
                chosen_odds > 0,
                1 + 100 / chosen_odds.abs(),
            )
            net_stake = fstar.where(win_bet, -fstar).where(resolved)
            net_odds = (decimal_odds - 1).where(
                win_bet,
                -1.0,
            ).where(resolved)

            settled[f"chosen_odds_{bet_type}"] = chosen_odds.astype(
                "Float64"
            )
            settled[f"win_bet_{bet_type}"] = win_bet.astype("boolean")
            settled[f"net_stake_{bet_type}"] = net_stake.astype(
                "Float64"
            )
            settled[f"net_odds_{bet_type}"] = net_odds.astype("Float64")

        return type(self)(settled)

    def validate(self) -> None:
        """Validate identity, betting, date, and nullable result fields."""
        if self.frame.empty:
            raise MoneylineIntegrityError("Moneyline DataFrame is empty")

        required_identity_columns = {
            "date",
            "fighter_red",
            "fighter_blue",
            "open_red",
            "open_blue",
            "close1_red",
            "close1_blue",
            "close2_red",
            "close2_blue",
        }
        missing_identity_columns = (
            required_identity_columns - set(self.frame.columns)
        )

        if missing_identity_columns:
            raise MoneylineIntegrityError(
                "Moneyline data is missing columns: "
                f"{sorted(missing_identity_columns)}"
            )

        null_fighters = self.frame[
            ["fighter_red", "fighter_blue"]
        ].isna().any(axis=1)

        if null_fighters.any():
            raise MoneylineIntegrityError(
                "Moneyline data contains null fighter names"
            )

        same_fighter = self.frame["fighter_red"].eq(
            self.frame["fighter_blue"]
        )

        if same_fighter.any():
            raise MoneylineIntegrityError(
                "Moneyline data contains identical red and blue fighters"
            )

        for odds_type in ("open", "close1", "close2"):
            for color in ("red", "blue"):
                column = f"{odds_type}_{color}"
                numeric = pd.to_numeric(
                    self.frame[column],
                    errors="coerce",
                )
                invalid = self.frame[column].notna() & numeric.isna()

                if invalid.any():
                    raise MoneylineIntegrityError(
                        f"{column} contains nonnumeric odds"
                    )

                if numeric.dropna().eq(0).any():
                    raise MoneylineIntegrityError(
                        f"{column} contains zero American odds"
                    )

                self.frame[column] = numeric.astype("Float64")

        for bet_type in self.bet_types:
            self._validate_bet_type(bet_type)

        winner_values = self.frame["winner_bool"].dropna()

        if not winner_values.isin([0, 1]).all():
            raise MoneylineIntegrityError(
                "winner_bool values must be 0, 1, or null"
            )

    def _validate_bet_type(self, bet_type: str) -> None:
        required_columns = {
            f"pred_name_{bet_type}",
            f"pred_winner_{bet_type}",
            f"choice_proba_{bet_type}",
            f"fstar_{bet_type}",
            f"stake_{bet_type}",
            f"edge_{bet_type}",
            f"ev_{bet_type}",
        }
        missing_columns = required_columns - set(self.frame.columns)

        if missing_columns:
            raise MoneylineIntegrityError(
                f"{bet_type} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        prediction_column = f"pred_winner_{bet_type}"
        predictions = pd.to_numeric(
            self.frame[prediction_column],
            errors="coerce",
        ).astype("Int64")

        if not predictions.dropna().isin([0, 1]).all():
            raise MoneylineIntegrityError(
                f"{prediction_column} values must be 0, 1, or null"
            )

        self.frame[prediction_column] = predictions

        probability_column = f"choice_proba_{bet_type}"
        probabilities = pd.to_numeric(
            self.frame[probability_column],
            errors="coerce",
        )

        if not probabilities.dropna().between(0, 1).all():
            raise MoneylineIntegrityError(
                f"{probability_column} must be between 0 and 1"
            )

        for prefix in ("fstar", "stake"):
            column = f"{prefix}_{bet_type}"
            values = pd.to_numeric(
                self.frame[column],
                errors="coerce",
            )

            if values.dropna().lt(0).any():
                raise MoneylineIntegrityError(
                    f"{column} cannot contain negative values"
                )

    def _validate_fighter_alignment(
        self,
        bets: pd.DataFrame,
        event: pd.DataFrame,
    ) -> None:
        for color in ("red", "blue"):
            column = f"fighter_{color}"
            bet_names = bets[column].astype("string").str.strip().str.lower()
            event_names = (
                event[column].astype("string").str.strip().str.lower()
            )

            if not bet_names.equals(event_names):
                raise MoneylineIntegrityError(
                    "Single-event results are not aligned with moneyline "
                    f"bets by {column}"
                )

    def _remove_exported_index(self) -> None:
        self.frame = self.frame.loc[
            :,
            ~self.frame.columns.str.match(r"^Unnamed"),
        ].copy()

    def _ensure_result_columns(self) -> None:
        self._ensure_column("winner_bool", "Int64")
        self._ensure_column("winner_name", "string")

        for bet_type in self.settled_types:
            self._ensure_column(
                f"win_bet_{bet_type}",
                "boolean",
            )
            self._ensure_column(
                f"chosen_odds_{bet_type}",
                "Float64",
            )
            self._ensure_column(
                f"net_odds_{bet_type}",
                "Float64",
            )
            self._ensure_column(
                f"net_stake_{bet_type}",
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
            raise MoneylineIntegrityError(
                f"Unable to convert {column!r} to {dtype}"
            ) from exc

    def _normalize_date(self) -> None:
        if "date" not in self.frame:
            raise MoneylineIntegrityError(
                "Moneyline DataFrame requires a date column"
            )

        try:
            self.frame["date"] = pd.to_datetime(
                self.frame["date"],
                format="%Y-%m-%d",
                errors="raise",
            ).dt.normalize()
        except (TypeError, ValueError) as exc:
            raise MoneylineIntegrityError(
                "Moneyline dates must use YYYY-MM-DD"
            ) from exc
