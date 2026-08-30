from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class OddsIntegrityError(ValueError):
    """Raised when an odds DataFrame fails an integrity check."""


@dataclass(frozen=True, slots=True)
class OddsRepository:
    """
    Load, validate, and combine historical and non-merged UFC odds.

    Date contract:
        The odds scraper parses event dates using
        ``dateutil.parser.parse(...).date()`` and converts them to ISO
        ``YYYY-MM-DD`` strings.

        Public loading methods enforce that format and return
        ``event_date`` as a timezone-naive pandas ``datetime64[ns]``
        column normalized to midnight.

        Invalid dates raise ``OddsIntegrityError``.

    Overlap contract:
        ``event_date``, ``blue_fighter``, and ``red_fighter`` form the
        unique fight key. When historical and non-merged odds overlap,
        the non-merged row takes precedence.

    Consumers should use the public loading methods rather than calling
    ``_read_csv()`` directly.
    """

    odds_history_file: Path
    non_merged_odds_file: Path

    key_columns: tuple[str, ...] = (
        "event_date",
        "blue_fighter",
        "red_fighter",
        "open_blue",
        "close1_blue",
        "close2_blue",
        "open_red",
        "close1_red",
        "close2_red",
    )

    odds_columns: tuple[str, ...] = (
        "open_blue",
        "close1_blue",
        "close2_blue",
        "open_red",
        "close1_red",
        "close2_red",
    )

    required_columns: tuple[str, ...] = (
        "blue_fighter",
        "open_blue",
        "close1_blue",
        "close2_blue",
        "red_fighter",
        "open_red",
        "close1_red",
        "close2_red",
        "event_date",
        "og_blue_name",
        "og_red_fighter",
    )

    def load_history(self) -> pd.DataFrame:
        history = self._read_csv(self.odds_history_file)
        history = history.drop_duplicates(
                subset=list(self.key_columns),
                keep="last",
            )
        self._validate_frame(history, source="odds history")
        return history

    def load_non_merged(self) -> pd.DataFrame:
        if not self.non_merged_odds_file.exists():
            return pd.DataFrame()

        non_merged = self._read_csv(self.non_merged_odds_file)
        self._validate_frame(non_merged, source="non-merged odds")
        return non_merged

    def merge_missing(
        self,
        missing_odds: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate and merge newly scraped odds into non-merged odds.

        Newly scraped rows take precedence when their unique fight key
        overlaps an existing non-merged row. This method does not write
        the resulting DataFrame to disk.
        """
        new_odds = missing_odds.copy()
        self._validate_frame(
            new_odds,
            source="newly scraped odds",
        )

        history = self.load_history()
        self._validate_compatible_schemas(history, new_odds)

        existing = self.load_non_merged()

        if existing.empty:
            merged = new_odds
        else:
            self._validate_compatible_schemas(existing, new_odds)

            merged = pd.concat(
                [existing, new_odds],
                axis=0,
                ignore_index=True,
            )

            merged = merged.drop_duplicates(
                subset=list(self.key_columns),
                keep="last",
            )

        self._validate_no_duplicates(
            merged,
            source="merged non-merged odds",
        )

        return (
            merged
            .sort_values("event_date")
            .reset_index(drop=True)
        )

    def combine_with_history(
        self,
        non_merged_odds: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate and combine odds history with non-merged odds.

        Non-merged rows take precedence when their unique fight key
        overlaps a historical row.
        """
        history = self.load_history()

        if non_merged_odds.empty:
            return history

        staged_odds = non_merged_odds.copy()

        self._validate_frame(
            staged_odds,
            source="non-merged odds input",
        )
        self._validate_compatible_schemas(
            history,
            staged_odds,
        )

        combined = pd.concat(
            [history, staged_odds],
            axis=0,
            ignore_index=True,
        )

        # Non-merged odds take precedence because they are concatenated
        # after historical odds.
        combined = combined.drop_duplicates(
            subset=list(self.key_columns),
            keep="last",
        )

        self._validate_no_duplicates(
            combined,
            source="combined odds",
        )

        return (
            combined
            .sort_values("event_date")
            .reset_index(drop=True)
        )

    def load_combined(self) -> pd.DataFrame:
        non_merged = self.load_non_merged()
        return self.combine_with_history(non_merged)

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        if not path.is_file():
            raise OddsIntegrityError(
                f"Odds file does not exist: {path}"
            )

        frame = pd.read_csv(path)

        # Remove indexes written by older CSV exports.
        return frame.loc[
            :,
            ~frame.columns.str.match(r"^Unnamed"),
        ].copy()

    def _validate_frame(
        self,
        frame: pd.DataFrame,
        *,
        source: str,
    ) -> None:
        if frame.empty:
            raise OddsIntegrityError(f"{source} is empty")

        missing_columns = set(self.required_columns) - set(frame.columns)

        if missing_columns:
            raise OddsIntegrityError(
                f"{source} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        # null_counts = frame[list(self.key_columns)].isna().sum()
        # columns_with_nulls = null_counts[null_counts > 0]

        # if not columns_with_nulls.empty:
        #     raise OddsIntegrityError(
        #         f"{source} has null key values: "
        #         f"{columns_with_nulls.to_dict()}"
        #     )

        try:
            frame["event_date"] = pd.to_datetime(
                frame["event_date"],
                format="%Y-%m-%d",
                errors="raise",
            ).dt.normalize()
        except (TypeError, ValueError) as exc:
            raise OddsIntegrityError(
                f"{source} contains an invalid event date; "
                "expected YYYY-MM-DD"
            ) from exc

        for column in self.odds_columns:
            numeric_values = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

            invalid_values = numeric_values.isna()

            if invalid_values.any():
                raise OddsIntegrityError(
                    f"{source} contains "
                    f"{int(invalid_values.sum())} invalid values "
                    f"in {column!r}"
                )

            if numeric_values.eq(0).any():
                raise OddsIntegrityError(
                    f"{source} contains zero American odds "
                    f"in {column!r}"
                )

            frame[column] = numeric_values

        same_fighter = frame["red_fighter"].eq(
            frame["blue_fighter"]
        )

        if same_fighter.any():
            raise OddsIntegrityError(
                f"{source} contains fights with the same red "
                "and blue fighter"
            )

        self._validate_no_duplicates(frame, source=source)

    @staticmethod
    def _validate_compatible_schemas(
        history: pd.DataFrame,
        non_merged: pd.DataFrame,
    ) -> None:
        history_columns = list(history.columns)
        non_merged_columns = list(non_merged.columns)

        if history_columns == non_merged_columns:
            return

        only_in_history = sorted(
            set(history_columns) - set(non_merged_columns)
        )
        only_in_non_merged = sorted(
            set(non_merged_columns) - set(history_columns)
        )

        raise OddsIntegrityError(
            "Odds schemas do not match. "
            f"Only in history: {only_in_history}. "
            f"Only in non-merged: {only_in_non_merged}."
        )

    def _validate_no_duplicates(
        self,
        frame: pd.DataFrame,
        *,
        source: str,
    ) -> None:
        duplicate_mask = frame.duplicated(
            subset=list(self.key_columns),
            keep=False,
        )

        if duplicate_mask.any():
            duplicate_rows = frame.loc[
                duplicate_mask,
                list(self.key_columns),
            ]

            raise OddsIntegrityError(
                f"{source} contains duplicate fights:\n"
                f"{duplicate_rows.head(10).to_string(index=False)}"
            )
