import pandas as pd 
import numpy as np 
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from ufc_betting.config import settings
from ufc_betting.Models.LogisticRegression.odds_input import replace_opening_odds

class TrainTestBuilder: 

    def __init__(
            self, df, feats, target_col, date_col, odds_type, year, month, day,
            required_features_only=False,
    ):

        self.selected_feats = feats
        self.target_col = target_col
        self.date_col = date_col
        self.odds_type = odds_type
        self.start_date = pd.Timestamp(year=year, month=month, day=day)

        cat_cols = ['math_red', 'math_blue', 'elo_pred', 'womens_fight']
        self.cat_cols = cat_cols
        odds_cols = [
            f'dec_fair_{odds_type}_red', f'dec_fair_{odds_type}_blue',
            f'dec_{odds_type}_red', f'dec_{odds_type}_blue'
        ]
        # cols that should not effect filtering of the dataframe
        other_cols = [
            'fighter_red', 'fighter_blue', target_col, date_col,
            'open_red', 'open_blue', 'close1_red', 'close1_blue', 
            'close2_red', 'close2_blue', 
            'num_wins_red', 'num_wins_blue', 'num_losses_red', 'num_losses_blue',
            'dec_fair_open_red', 'dec_fair_open_blue', 'dec_fair_close1_red', 'dec_fair_close1_blue',
            'dec_fair_close2_red', 'dec_fair_close2_blue', 'dec_open_red', 'dec_open_blue', 'dec_close1_red', 'dec_close1_blue',
            'dec_close2_red', 'dec_close2_blue', 
            'proba_fair_open_diff', 'proba_fair_close1_diff', 'proba_fair_close2_diff'
        ]
        valid_cols = feats + odds_cols + other_cols
        valid_cols = list(np.unique(valid_cols))
        self.valid_cols = valid_cols
        
        df = df.copy()
        df[date_col] = pd.to_datetime(
            df[date_col], format='%Y-%m-%d', errors='coerce'
        ).dt.normalize()
        df = df.sort_values(by=date_col, ascending=True, kind='stable')

        if required_features_only:
            # Preserve optional price columns without filtering opening history
            # according to whether later market snapshots were recorded.
            required = list(dict.fromkeys(feats + [target_col, date_col]))
            missing = set(required) - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {sorted(missing)}")
            for col in set(cat_cols) & set(df.columns):
                df[col] = df[col].astype('category')
            self.all_processed_df = df.loc[df[target_col].isin([0, 1])].dropna(
                subset=required
            ).reset_index(drop=True)
        else:
            self.all_processed_df = self.process_feats(
                df, valid_cols, target_col, cat_cols
            )
        self.df = self.filter_by_date(
            self.all_processed_df.copy(),
            year=year, 
            month=month, 
            day=day,
            date_col=date_col
        )
        print('PREPARE SHAPE:', self.df.shape)

    @staticmethod
    def process_feats(df, valid_cols, target_col, cat_cols):
        
        df[cat_cols] = df[cat_cols].astype('category')
        metadata_cols = [col for col in ('bout_position', 'event_name', 'event_type')
                         if col in df.columns and col not in valid_cols]
        df = df[valid_cols + metadata_cols].copy()
        # Optional sampling metadata must not change the training cohort.
        df = df[df[target_col] < 2].dropna(subset=valid_cols).reset_index(drop=True)
        return df

    @staticmethod
    def filter_by_date(df, year, month=1, day=1, date_col='date'):
        """Keep all rows after a given year/month/day."""
        
        # Ensure datetime
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        # Create cutoff date
        cutoff = pd.Timestamp(year=year, month=month, day=day)

        # Filter
        df = df[df[date_col] > cutoff]

        # Update internal dataframe
        df = df.reset_index(drop=True)

        print(f"Filtered: kept {len(df)} rows from {cutoff.date()} onward.")
        return df 

    @staticmethod
    def scale_encode(X_train, X_test, cat_cols, num_cols):
        # Fit only on training rows; both splits use the same fitted transforms.
        scaler = StandardScaler().fit(X_train[num_cols]) if len(num_cols) else None
        encoder = (OneHotEncoder(
            drop="if_binary", handle_unknown="ignore", sparse_output=False, dtype=int,
        ).fit(X_train[cat_cols]) if len(cat_cols) else None)
        return {
            'X_train': TrainTestBuilder.prepare_inference_data(X_train, scaler, encoder),
            'X_test': TrainTestBuilder.prepare_inference_data(X_test, scaler, encoder),
            'encoded_cols': (encoder.get_feature_names_out(cat_cols)
                             if encoder is not None else np.array([])),
            'scaler': scaler,
            'encoder': encoder,
        }

    @staticmethod
    def encode_categorical(features, encoder, categorical_columns):
        """Preserve fitted one-hot column names and the input row index."""
        columns = list(categorical_columns)
        values = encoder.transform(features[columns])
        if hasattr(values, 'toarray'):
            values = values.toarray()
        return pd.DataFrame(values, columns=encoder.get_feature_names_out(columns),
                            index=features.index)


    @staticmethod
    def transform_features(features, scaler, encoder):
        """Apply training-fitted transforms without refitting or inferring dtypes."""
        parts = []
        if scaler is not None:
            columns = list(scaler.feature_names_in_)
            parts.append(pd.DataFrame(scaler.transform(features[columns]),
                                      columns=columns, index=features.index))
        if encoder is not None:
            parts.append(TrainTestBuilder.encode_categorical(features, encoder, encoder.feature_names_in_))
        if not parts:
            return features.copy()
        return pd.concat(parts, axis=1)

    @staticmethod
    def prepare_inference_data(
        features, scaler=None, encoder=None, *, switch_open_odds=False, odds_type=None,
    ):
        """Prepare raw rows using fitted transforms, without fitting again.

        Set switch_open_odds=True and odds_type='close1' or 'close2' to copy
        recorded closing values into the opening feature names before scaling.
        With switch_open_odds=False, odds values are left unchanged even when
        odds_type is supplied. The input frame and row index are preserved.
        """
        if switch_open_odds:
            if odds_type not in ('close1', 'close2'):
                raise ValueError("switch_open_odds=True requires odds_type='close1' or 'close2'")
            names = []
            for transformer in (scaler, encoder):
                if transformer is not None:
                    names.extend(transformer.feature_names_in_)
            if not names:
                raise ValueError('Odds substitution requires fitted preprocessing')
            # Only select model inputs and their market counterparts: preserve the
            # recorded fair features instead of recomputing them from decimal odds.
            sources = [name.replace('open', odds_type) for name in names]
            columns = list(dict.fromkeys(names + sources))
            features = replace_opening_odds(
                features.loc[:, [name for name in columns if name in features]],
                odds_type, features=names,
            )
        return TrainTestBuilder.transform_features(features, scaler, encoder)

    @staticmethod
    def close_odds_to_open(feats_open, feats_close, odds_type):
        # Compatibility for existing notebooks; substitution lives in odds_input.
        inputs = replace_opening_odds(feats_close, odds_type)
        feats_open = feats_open.copy()
        feats_open['proba_fair_open_diff'] = inputs['proba_fair_open_diff']
        return feats_open

    def prepare_train_test(
            self, train_size=0.8, scale=True, group_dates=True, *,
            save_split=False, output_dir=None,
            train_split_path=None, test_split_path=None,
            val_start_date=None, test_start_date=None, val_split_path=None,
    ):
        """Split chronologically using dates, or the legacy two-way ratio split.

        Supply both val_start_date and test_start_date for three disjoint sets:
        train < val_start_date; val_start_date <= val < test_start_date;
        test >= test_start_date. Date boundaries override train_size/group_dates.
        Transforms fit on training rows only. Optional card metadata is retained
        in each full-row dataframe and saved split, not added to model inputs.

        With save_split=True, save unscaled full rows as train_<odds_type>.csv
        and val_<odds_type>.csv under Data/train_val_test/split, or output_dir.
        Explicit train_split_path/test_split_path save the corresponding unscaled
        full rows even without save_split=True, overriding the default filenames.
        Whole event dates stay together by default, so proportions may vary.
        Existing test-named return keys refer to the validation portion.
        Categorical features use one-hot encoding when scale=True.
        """
        if val_start_date is not None or test_start_date is not None:
            if val_start_date is None or test_start_date is None:
                raise ValueError('Supply both val_start_date and test_start_date')
            return self._prepare_dated_splits(
                val_start_date, test_start_date, scale=scale, save_split=save_split,
                output_dir=output_dir, train_split_path=train_split_path,
                val_split_path=val_split_path, test_split_path=test_split_path,
            )
        if val_split_path is not None:
            raise ValueError('val_split_path requires val_start_date and test_start_date')
        y = self.df[self.target_col]
        X = self.df[self.selected_feats]
        dates = self.df[self.date_col]

        num_cols = X.select_dtypes(include='number').columns
        cat_cols = X.select_dtypes(include='category').columns

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, shuffle=False
        )
        if group_dates:
            # Move the entire boundary date to test; the requested row target is
            # approximate when an event would otherwise straddle the split.
            boundary = dates.loc[X_test.index].min()
            train_mask = dates < boundary
            X_train, X_test = X.loc[train_mask], X.loc[~train_mask]
            y_train, y_test = y.loc[train_mask], y.loc[~train_mask]
            if X_train.empty or X_test.empty:
                raise ValueError("Need events on both sides of the train/test split")
        scaler = None
        encoder = None
        if scale: 
            scale_pkt = TrainTestBuilder.scale_encode(
                X_train, X_test, cat_cols, num_cols
            )
            scaler, encoder = scale_pkt['scaler'], scale_pkt['encoder']
            X_train, X_test = scale_pkt["X_train"], scale_pkt["X_test"]

        dates_train = dates.loc[X_train.index]
        dates_test = dates.loc[X_test.index]

        pkt = {
            'X_train':X_train, 'y_train':y_train,
            'X_test':X_test, 'y_test':y_test, 
            'dates_train':dates_train, 'dates_test':dates_test,
            'scaler':scaler, 'encoder':encoder,
            'filtered_df':self.df.copy(), 
            'train_df':self.df.loc[X_train.index].copy(),
            'test_df':self.df.loc[X_test.index].copy(),
            'num_cols':num_cols,
            'cat_cols':cat_cols,
        }
        if save_split or train_split_path is not None or test_split_path is not None:
            directory = (Path(output_dir) if output_dir is not None else
                         settings.data_dir / 'train_val_test' / 'split')
            paths = {}
            for name, key, explicit_path in (
                ('train', 'train_df', train_split_path),
                ('val', 'test_df', test_split_path),
            ):
                if explicit_path is not None or save_split:
                    paths[name] = (Path(explicit_path) if explicit_path is not None
                                   else directory / f'{name}_{self.odds_type}.csv')
            if len(paths) == 2 and paths['train'].resolve() == paths['val'].resolve():
                raise ValueError('Train and test split paths must be different')
            for name, key in (('train', 'train_df'), ('val', 'test_df')):
                if name in paths:
                    paths[name].parent.mkdir(parents=True, exist_ok=True)
                    pkt[key].to_csv(paths[name], index=False)
            pkt['split_paths'] = paths
        return pkt

    def _prepare_dated_splits(
        self, val_start_date, test_start_date, *, scale, save_split, output_dir,
        train_split_path, val_split_path, test_split_path,
    ):
        val_start = pd.Timestamp(val_start_date).normalize()
        test_start = pd.Timestamp(test_start_date).normalize()
        if pd.isna(val_start) or pd.isna(test_start) or val_start >= test_start:
            raise ValueError('Dates must be valid and val_start_date must precede test_start_date')
        dates = self.df[self.date_col]
        masks = {'train': dates < val_start,
                 'val': (dates >= val_start) & (dates < test_start),
                 'test': dates >= test_start}
        for name, mask in masks.items():
            if not mask.any():
                raise ValueError(f'{name} split has no eligible rows; check cutoff dates and feature data')
        frames = {name: self.df.loc[mask].copy() for name, mask in masks.items()}
        inputs = {name: frame[self.selected_feats].copy() for name, frame in frames.items()}
        num_cols = inputs['train'].select_dtypes(include='number').columns
        cat_cols = inputs['train'].select_dtypes(include='category').columns
        scaler = encoder = None
        if scale:
            fitted = self.scale_encode(inputs['train'], inputs['val'], cat_cols, num_cols)
            scaler, encoder = fitted['scaler'], fitted['encoder']
            inputs['train'], inputs['val'] = fitted['X_train'], fitted['X_test']
            inputs['test'] = self.prepare_inference_data(inputs['test'], scaler, encoder)
        pkt = {'scaler': scaler, 'encoder': encoder, 'num_cols': num_cols,
               'cat_cols': cat_cols, 'filtered_df': self.df.copy(),
               'val_start_date': val_start, 'test_start_date': test_start}
        summaries = []
        for name, frame in frames.items():
            pkt.update({f'X_{name}': inputs[name], f'y_{name}': frame[self.target_col],
                        f'dates_{name}': frame[self.date_col], f'{name}_df': frame})
            count = len(frame)
            percentage = 100 * count / len(self.df)
            first, last = frame[self.date_col].min(), frame[self.date_col].max()
            summaries.append(dict(split=name, rows=count, percentage=percentage,
                                  first_date=first, last_date=last))
            print(f'{name}: {count:,} rows ({percentage:.2f}%) | {first.date()} to {last.date()}')
        pkt['split_summary'] = pd.DataFrame(summaries)
        directory = Path(output_dir) if output_dir is not None else settings.data_dir / 'train_val_test' / 'split'
        paths = {}
        for name, explicit in [('train', train_split_path), ('val', val_split_path),
                               ('test', test_split_path)]:
            if explicit is not None or save_split:
                paths[name] = Path(explicit) if explicit is not None else directory / f'{name}_{self.odds_type}.csv'
        if len({path.resolve() for path in paths.values()}) != len(paths):
            raise ValueError('Train, validation and test split paths must be different')
        for name, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            frames[name].to_csv(path, index=False)
        if paths:
            pkt['split_paths'] = paths
        return pkt

    @staticmethod
    def prepare_stacked_data(
        train_sets, test_sets, df_train_open, df_test_open
    ):
        X_train = pd.concat(
            [
                (
                    dat['proba_red'] - dat['proba_blue']
                ).rename(f'proba_diff_{i}')
                for i, dat in enumerate(train_sets)
            ],
            axis=1
        )
        # X_train['total_fights'] = df_total_fights_train['total_fights']
        y_train = df_train_open['winner']

        X_test = pd.concat(
            [
                (
                    dat['proba_red'] - dat['proba_blue']
                ).rename(f'proba_diff_{i}')
                for i, dat in enumerate(test_sets)
            ],
            axis=1
        )
        y_test = df_test_open['winner']
        
        return X_train, y_train, X_test, y_test
