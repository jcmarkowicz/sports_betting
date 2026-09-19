from pathlib import Path

import optuna


class OptunaSQLiteManager:
    """Create reproducible Optuna studies backed by a SQLite database."""

    def __init__(self, database_path, seed=42):
        self.database_path = Path(database_path).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.seed = seed

    @property
    def storage_url(self):
        return f"sqlite:///{self.database_path.as_posix()}"

    def create_study(
        self,
        study_name,
        direction='maximize',
        load_if_exists=True,
        overwrite=False,
    ):
        if overwrite:
            try:
                optuna.delete_study(
                    study_name=study_name,
                    storage=self.storage_url,
                )
            except KeyError:
                pass

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        return optuna.create_study(
            study_name=study_name,
            storage=self.storage_url,
            sampler=sampler,
            direction=direction,
            load_if_exists=load_if_exists and not overwrite,
        )
