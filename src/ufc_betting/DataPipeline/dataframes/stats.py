from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class StatsIntegrityError(ValueError):
    """Raised when a stats DataFrame fails an integrity check."""


@dataclass(frozen=True, slots=True)
class StatsRepository:
    """
    Load, validate, and combine historical and non-merged UFC stats.

    Date contract:
        The stats scrapers initially produce ``event_date`` values using
        ``dateutil.parser.parse(...).date()``. Once stored in CSV, event
        dates must use the ISO ``YYYY-MM-DD`` format.

        The public loading methods enforce that format and return
        ``event_date`` as a timezone-naive pandas ``datetime64[ns]``
        column. Every value is normalized to midnight, for example
        ``Timestamp("2026-08-29 00:00:00")``.

        Invalid date strings raise ``StatsIntegrityError``.

    Public loading methods:
        - ``load_history()``
        - ``load_non_merged()``
        - ``load_combined()``

    Consumers should use the public loading methods rather than calling
    ``_read_csv()`` directly, because raw CSV values are converted and
    validated after reading.
    """

    stats_history_file: Path
    non_merged_stats_file: Path

    key_columns: tuple[str, ...] = (
        "fighter_red",
        "fighter_blue",
        "event_date",
        "event_name",
    )

    required_columns: tuple[str, ...] = (
        "event_name",
        "event_date",
        "fight_url",
        "fighter_red",
        "fighter_blue",
        "winner",
    )

    def load_history(self) -> pd.DataFrame:
        history = self._read_csv(self.stats_history_file)
        self._validate_frame(history, source="stats history")
        return history

    def load_non_merged(self) -> pd.DataFrame:
        if not self.non_merged_stats_file.exists():
            return pd.DataFrame()

        non_merged = self._read_csv(self.non_merged_stats_file)
        self._validate_frame(non_merged, source="non-merged stats")
        return non_merged

    def merge_missing(
        self,
        missing_stats: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate and merge newly scraped stats into non-merged stats.

        Newly scraped rows take precedence when their unique fight key
        overlaps an existing non-merged row. This method does not write
        the resulting DataFrame to disk.
        """
        new_stats = missing_stats.copy()
        self._validate_frame(
            new_stats,
            source="newly scraped stats",
        )

        history = self.load_history()
        self._validate_compatible_schemas(history, new_stats)

        existing = self.load_non_merged()

        if existing.empty:
            merged = new_stats
        else:
            self._validate_compatible_schemas(existing, new_stats)

            merged = pd.concat(
                [existing, new_stats],
                axis=0,
                ignore_index=True,
            )

            merged = merged.drop_duplicates(
                subset=list(self.key_columns),
                keep="last",
            )

        self._validate_no_duplicates(
            merged,
            source="merged non-merged stats",
        )

        return (
            merged
            .sort_values("event_date")
            .reset_index(drop=True)
        )

    def combine_with_history(
        self,
        non_merged_stats: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate and combine stats history with non-merged stats.

        Non-merged rows take precedence when their unique fight key
        overlaps a historical row.
        """
        history = self.load_history()

        if non_merged_stats.empty:
            return history

        staged_stats = non_merged_stats.copy()

        self._validate_frame(
            staged_stats,
            source="non-merged stats input",
        )
        self._validate_compatible_schemas(
            history,
            staged_stats,
        )

        combined = pd.concat(
            [history, staged_stats],
            axis=0,
            ignore_index=True,
        )

        # The two sources can overlap by date. The key columns uniquely
        # identify a fight, and non-merged data takes precedence because
        # it is concatenated after the historical data.
        combined = combined.drop_duplicates(
            subset=list(self.key_columns),
            keep="last",
        ).reset_index(drop=True)

        self._validate_no_duplicates(combined, source="combined stats")
        return combined.sort_values("event_date").reset_index(drop=True)

    def load_combined(self) -> pd.DataFrame:
        non_merged = self.load_non_merged()
        return self.combine_with_history(non_merged)

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        if not path.is_file():
            raise StatsIntegrityError(f"Stats file does not exist: {path}")

        frame = pd.read_csv(path)

        # The historical CSV currently contains a saved DataFrame index.
        frame = frame.loc[
            :,
            ~frame.columns.str.match(r"^Unnamed")
        ].copy()

        return frame

    def _validate_frame(
        self,
        frame: pd.DataFrame,
        *,
        source: str,
    ) -> None:
        if frame.empty:
            raise StatsIntegrityError(f"{source} is empty")

        missing_columns = set(self.required_columns) - set(frame.columns)

        if missing_columns:
            raise StatsIntegrityError(
                f"{source} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        null_counts = frame[list(self.key_columns)].isna().sum()
        columns_with_nulls = null_counts[null_counts > 0]

        if not columns_with_nulls.empty:
            raise StatsIntegrityError(
                f"{source} has null key values: "
                f"{columns_with_nulls.to_dict()}"
            )

        try:
            frame["event_date"] = pd.to_datetime(
                frame["event_date"],
                format="%Y-%m-%d",
                errors="raise",
            ).dt.normalize()
        except (TypeError, ValueError) as exc:
            raise StatsIntegrityError(
                f"{source} contains an invalid event date; "
                "expected YYYY-MM-DD"
            ) from exc

        invalid_winners = ~(
            frame["winner"].eq(frame["fighter_red"])
            | frame["winner"].eq(frame["fighter_blue"])
        )

        if invalid_winners.any():
            bad_rows = frame.loc[
                invalid_winners,
                ["event_name", "fighter_red", "fighter_blue", "winner"],
            ]

            raise StatsIntegrityError(
                f"{source} has winners who are not one of the fighters:\n"
                f"{bad_rows.head().to_string(index=False)}"
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

        raise StatsIntegrityError(
            "Stats schemas do not match. "
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

            raise StatsIntegrityError(
                f"{source} contains duplicate fights:\n"
                f"{duplicate_rows.head(10).to_string(index=False)}"
            )
