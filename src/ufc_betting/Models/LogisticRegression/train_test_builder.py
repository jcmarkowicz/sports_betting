import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class TrainTestBuilder: 

    def __init__(
            self, df, feats, target_col, date_col, odds_type, year, month, day
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
        ]
        valid_cols = feats + odds_cols + other_cols
        self.valid_cols = valid_cols
        
        df = df.copy()
        df[date_col] = pd.to_datetime(
            df[date_col], format='%Y-%m-%d', errors='coerce'
        ).dt.normalize()
        df = df.sort_values(by=date_col, ascending=True)

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
        df = df[valid_cols].copy()
        df = df[df[target_col] < 2].dropna().reset_index(drop=True)
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
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(X_train[num_cols])
        X_test_num = scaler.transform(X_test[num_cols])

        encoder = OneHotEncoder(
            drop="if_binary",
            handle_unknown="ignore", # cat not present in train gets all 0s in test 
            sparse_output=False,
            dtype=int,
        )
        X_train_cat = encoder.fit_transform(X_train[cat_cols])
        X_test_cat = encoder.transform(X_test[cat_cols])
        encoded_cols = encoder.get_feature_names_out(cat_cols) # only columns fitted in X_train

        X_train_cat = pd.DataFrame(
            X_train_cat,
            columns=encoded_cols,
            index=X_train.index,
        )
        X_train_num = pd.DataFrame(
            X_train_num, 
            columns=num_cols,
            index=X_train.index
        )
        X_test_cat = pd.DataFrame(
            X_test_cat,
            columns=encoded_cols,
            index=X_test.index,
        )
        X_test_num = pd.DataFrame(
            X_test_num, 
            columns=num_cols,
            index=X_test.index
        )
        X_train = pd.concat(
            [X_train_num, X_train_cat], axis=1
        )
        X_test = pd.concat(
            [X_test_num, X_test_cat], axis=1
        )
        pkt = {
            'X_train': X_train, 
            'X_test': X_test, 
            'encoded_cols':encoded_cols, 
            'scaler':scaler,
            'encoder':encoder
        }
        return pkt

    @staticmethod
    def encode_categorical(features, encoder, categorical_columns):
        """Encode categorical columns using an already-fitted encoder."""
        categorical_columns = list(categorical_columns)
        encoded = encoder.transform(features[categorical_columns])

        return pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(categorical_columns),
            index=features.index,
        )

    @staticmethod
    def close_odds_to_open(feats_open, feats_close, odds_type):
        feats_open['proba_fair_open_diff'] = feats_close[f'proba_fair_{odds_type}_diff']
        return feats_open

    def prepare_train_test(
            self, train_size, scale=True,
    ):
        """ curently handles all categorical columns as non ordinal, uses onehotencoder"""
        y = self.df[self.target_col]
        X = self.df[self.selected_feats]
        dates = self.df[self.date_col]

        num_cols = X.select_dtypes(include='number').columns
        cat_cols = X.select_dtypes(include='category').columns

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, shuffle=False
        )
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
