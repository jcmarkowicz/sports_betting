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
        """Combine moneyline frames, replacing older events by date."""
        available_frames = [
            moneylines.frame
            for moneylines in frames
            if moneylines is not None
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
    ) -> "MoneylineDataFrame | None":
        """
        Return a new moneyline frame populated with available results.

        Results are matched by normalized fighter names rather than row
        position. Draws, no-contests, null results, and other non-binary
        outcomes are excluded because they cannot settle as a moneyline win
        or loss. Rows remain unsettled when a model prediction is unavailable.
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

        event_rows = single_event.copy()
        settled = self.frame.copy()
        match_columns = ("fighter_red", "fighter_blue")
        match_keys: list[str] = []

        for column in match_columns:
            key = f"_{column}_key"
            match_keys.append(key)
            event_rows[key] = (
                event_rows[column]
                .astype("string")
                .str.strip() # remove whitespace at start and end of string
                .str.casefold() # language aware str.lower()
            )
            settled[key] = (
                settled[column]
                .astype("string")
                .str.strip()
                .str.casefold()
            )

        if event_rows.duplicated(match_keys).any():
            duplicates = event_rows.loc[
                event_rows.duplicated(match_keys, keep=False),
                list(match_columns),
            ]
            raise MoneylineIntegrityError(
                "Single-event results contain duplicate fights: "
                f"{duplicates.to_dict('records')}"
            )

        if settled.duplicated(match_keys).any():
            duplicates = settled.loc[
                settled.duplicated(match_keys, keep=False),
                list(match_columns),
            ]
            raise MoneylineIntegrityError(
                "Moneyline data contains duplicate fights: "
                f"{duplicates.to_dict('records')}"
            )

        result_rows = event_rows[
            [*match_keys, "winner", "winner_name"]
        ].rename(
            columns={
                "winner": "_event_winner",
                "winner_name": "_event_winner_name",
            }
        )

        # assert that every fight as a result 
        settled = settled.merge(
            result_rows,
            how="left",
            on=match_keys,
            sort=False,
            validate="one_to_one",
            indicator=True, # creates column that indicates if match found 
        )

        missing_matches = settled["_merge"].eq("left_only")
        if missing_matches.any():
            missing_fights = settled.loc[
                missing_matches,
                list(match_columns),
            ]
            raise MoneylineIntegrityError(
                "Moneyline fights are missing from single-event results: "
                f"{missing_fights.to_dict('records')}"
            )

        winners = pd.to_numeric(
            settled["_event_winner"],
            errors="coerce",
        )
        nonnumeric_winners = (
            settled["_event_winner"].notna() & winners.isna()
        )
        if nonnumeric_winners.any():
            invalid_values = (
                settled.loc[nonnumeric_winners, "_event_winner"]
                .unique()
                .tolist()
            )
            raise MoneylineIntegrityError(
                "Single-event results contain nonnumeric winner values: "
                f"{invalid_values}"
            )

        # Ties and other non-binary outcomes cannot settle a moneyline bet.
        binary_results = winners.isin([0, 1])
        settled = settled.loc[binary_results].copy()
        if settled.empty:
            return None

        winners = winners.loc[binary_results].astype("Int64")
        settled["winner_bool"] = winners
        settled["winner_name"] = settled[
            "_event_winner_name"
        ].astype("string")
        settled = settled.drop(
            columns=[
                *match_keys,
                "_event_winner",
                "_event_winner_name",
                "_merge",
            ]
        )

        for bet_type in self.settled_types:
            prediction_column = f"pred_winner_{bet_type}"
            odds_type = self.odds_type_by_settled_type[bet_type]

            # errors="coerce" forces any non numeric value to na 
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
            chosen_ev = pd.to_numeric(
                settled[f"ev_{bet_type}"],
                errors='coerce'
            ).astype("Float64")
            fstar = pd.to_numeric(
                settled[f"fstar_{bet_type}"],
                errors="coerce",
            ).astype("Float64")

            resolved = predictions.notna() & winners.notna()
            positive_ev = chosen_ev.gt(0).fillna(False)

            qualifying_bet = resolved & positive_ev
            win_bet = predictions.eq(winners).where(qualifying_bet)

            decimal_odds = (1 + chosen_odds / 100).where(
                chosen_odds > 0,
                1 + 100 / chosen_odds.abs(),
            )
            net_stake = fstar.where(win_bet, -fstar).where(qualifying_bet)
            net_odds = (decimal_odds - 1).where(
                win_bet,
                -1.0,
            ).where(qualifying_bet)

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
        invalid_winner = (
            self.frame["winner_bool"].notna()
            & ~self.frame["winner_bool"].isin([0, 1])
        )
        # Preserve unsettled nulls, but exclude ties and other outcomes that
        # cannot settle as a moneyline win or loss.
        self.frame = self.frame.loc[~invalid_winner].copy()

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

    def _remove_exported_index(self) -> None:
        self.frame = self.frame.loc[
            :,
            ~self.frame.columns.str.match(r"^Unnamed"),
        ].copy()

    def _ensure_result_columns(self) -> None:
        if "winner_bool" not in self.frame:
            self.frame["winner_bool"] = pd.Series(
                pd.NA,
                index=self.frame.index,
                dtype="Int64",
            )
        else:
            raw_winners = self.frame["winner_bool"]
            numeric_winners = pd.to_numeric(raw_winners, errors="coerce")
            keep_winner = raw_winners.isna() | numeric_winners.isin([0, 1])
            self.frame = self.frame.loc[keep_winner].copy()
            self.frame["winner_bool"] = numeric_winners.loc[
                keep_winner
            ].astype("Int64")

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
