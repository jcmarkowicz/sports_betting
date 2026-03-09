
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import numpy as np 
import pandas as pd 


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

class TrainTestBuilder: 
    def __init__(self, df, target_col, non_features=[], train_size=0.8, random_state=42):
        self.df = df
        self.target_col = target_col
        self.non_features = non_features
        self.train_size = train_size
        self.random_state = random_state

    def fit_gmm_cluster(self, X_train, X_test):
        blue_feats = ['td_defense_pct_blue','td_accuracy_pct_blue', 'control_pm_blue',
        'sig_str_defense_pct_blue', 'sig_str_accuracy_pct_blue', 'sig_str_landed_pm_blue','sig_str_absorbed_pm_blue',
        'kd_pm_blue', 'total_fight_time_blue', 'age_blue', 'reach_blue', 'total_bonus_blue', 'elo_blue',
        'win_pct_blue', 'ko_losses_blue']

        red_feats = ['td_defense_pct_red', 'td_accuracy_pct_red', 'control_pm_red',
        'sig_str_defense_pct_red', 'sig_str_accuracy_pct_red', 'sig_str_landed_pm_red', 'sig_str_absorbed_pm_red',
        'kd_pm_red',
        'total_fight_time_red', 'age_red', 'reach_red', 'total_bonus_red', 
        'elo_red', 'win_pct_red', 'ko_losses_red']

        df_mens = X_train[X_train['womens_fight']==0]
        df_mens_red  = df_mens[red_feats].rename(columns=lambda x: x.replace("_red", ""))
        df_mens_blue = df_mens[blue_feats].rename(columns=lambda x: x.replace("_blue", ""))

        df_womens = X_train[X_train['womens_fight']==1]
        df_womens_red  = df_womens[red_feats].rename(columns=lambda x: x.replace("_red", ""))
        df_womens_blue = df_womens[blue_feats].rename(columns=lambda x: x.replace("_blue", ""))

        df_all = pd.concat([df_mens_red, df_mens_blue, df_womens_red, df_womens_blue], axis=0, join="inner")

        scaler = StandardScaler()
        X_proc = scaler.fit_transform(df_all)

        gmm = GaussianMixture(
            n_components=3,
            covariance_type='full',
            random_state=0
        )
        gmm.fit(X_proc)
        labels = gmm.predict(X_proc)

        # compute silhouette
        sil = silhouette_score(X_proc, labels)
        print('SIL: ', sil)

        X_train['red_gmm_cluster'] = gmm.predict(scaler.transform(X_train[red_feats].rename(columns=lambda x: x.replace("_red", ""))))
        X_train['blue_gmm_cluster'] = gmm.predict(scaler.transform(X_train[blue_feats].rename(columns=lambda x: x.replace("_blue", ""))))

        X_test['red_gmm_cluster'] = gmm.predict(scaler.transform(X_test[red_feats].rename(columns=lambda x: x.replace("_red", ""))))
        X_test['blue_gmm_cluster'] = gmm.predict(scaler.transform(X_test[blue_feats].rename(columns=lambda x: x.replace("_blue", ""))))

        # Convert to categorical
        X_train['red_gmm_cluster'] = X_train['red_gmm_cluster'].astype('category')
        X_train['blue_gmm_cluster'] = X_train['blue_gmm_cluster'].astype('category')

        X_test['red_gmm_cluster'] = X_test['red_gmm_cluster'].astype('category')
        X_test['blue_gmm_cluster'] = X_test['blue_gmm_cluster'].astype('category')

        return X_train, X_test
    
    
    def encode_fighter_ids(self, df, red_col='fighter_red', blue_col='fighter_blue'):

        # Get all unique fighters from both columns
        all_fighters = pd.concat([df[red_col], df[blue_col]]).unique()
        
        # Map each fighter to a unique integer
        fighter_id_map = {fighter: idx for idx, fighter in enumerate(all_fighters)}
        
        # Apply mapping to create new ID columns
        df['red_id'] = df[red_col].map(fighter_id_map)
        df['blue_id'] = df[blue_col].map(fighter_id_map)
        
        return df, fighter_id_map

    def prepare_train_test(
        self,
        selected_features,
        outlier_dict=None,
        target_col='winner',
        clustering=False,
        one_hot_encode=False,
        run_pca=False,
        pca_variance_threshold=0.95
    ):
        # Clean the dataframe
        df = self.df.dropna().copy()
        print('PREPARE SHAPE:', df.shape)
        df = df[df['winner'] != 2]
        y = df[target_col]

        if outlier_dict is not None:
            for col, (lower, upper) in outlier_dict.items():
                if col in df.columns:
                    df = df[(df[col] >= lower) & (df[col] <= upper)]
                    y = y[df.index]  # align target with filtered features
                    
        df = df.reset_index(drop=True)
        y = y.reset_index(drop=True)

        df_model = df.copy()
        print('MODEL SHAPE:', df_model.shape)

        # Encode fighter IDs if present
        if 'fighter_id' in selected_features: 
            df_model, _ = self.encode_fighter_ids(df_model)
            df_model['red_id'] = df_model['red_id'].astype('category')
            df_model['blue_id'] = df_model['blue_id'].astype('category')
            selected_features += ['red_id', 'blue_id']
            selected_features.remove('fighter_id')

        # Drop non-feature columns and target from feature set
        feature_cols = [c for c in selected_features if c not in self.non_features + [self.target_col]]
        X = df_model.copy()

        # Split into train/test
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, train_size=self.train_size, random_state=self.random_state, shuffle=False
        )

        # Optional clustering
        if clustering:
            X_train_raw, X_test_raw = self.fit_gmm_cluster(X_train_raw, X_test_raw)

        # Convert boolean / categorical columns
        if 'womens_fight' in X_train_raw.columns:
            X_train_raw['womens_fight'] = X_train_raw['womens_fight'].astype('category')
            X_test_raw['womens_fight'] = X_test_raw['womens_fight'].astype('category')

        # Keep only selected features
        X_train_raw = X_train_raw[feature_cols]
        X_test_raw = X_test_raw[feature_cols]

        # Identify numerical and categorical columns
        num_cols = X_train_raw.select_dtypes(include='number').columns.tolist()
        cat_cols = X_train_raw.select_dtypes(exclude='number').columns.tolist()
        print("Categorical columns:", cat_cols)
        print("Numerical columns:", num_cols)

        # Standardize numerical features
        scaler = StandardScaler()
        if num_cols:
            X_train_num = scaler.fit_transform(X_train_raw[num_cols])
            X_test_num  = scaler.transform(X_test_raw[num_cols])
        else:
            X_train_num, X_test_num = np.array([]), np.array([])

        # Optionally run PCA
        if run_pca and num_cols:
            pca = PCA(n_components=pca_variance_threshold)
            X_train_num = pca.fit_transform(X_train_num)
            X_test_num = pca.transform(X_test_num)
            pca_cols = [f"PCA_{i+1}" for i in range(X_train_num.shape[1])]
        else:
            pca_cols = num_cols

        # Handle categorical features
        if cat_cols:
            if one_hot_encode:
                ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
                X_train_cat = ohe.fit_transform(X_train_raw[cat_cols])
                X_test_cat  = ohe.transform(X_test_raw[cat_cols])
                cat_cols_ohe = ohe.get_feature_names_out(cat_cols)
            else:
                X_train_cat = X_train_raw[cat_cols].to_numpy()
                X_test_cat  = X_test_raw[cat_cols].to_numpy()
                # Ensure 2D even if only one categorical column
                if X_train_cat.ndim == 1:
                    X_train_cat = X_train_cat.reshape(-1,1)
                    X_test_cat  = X_test_cat.reshape(-1,1)
                cat_cols_ohe = cat_cols
        else:
            X_train_cat, X_test_cat = np.array([]).reshape(len(X_train_num),0), np.array([]).reshape(len(X_test_num),0)
            cat_cols_ohe = []

        # Combine numerical (or PCA) + categorical
        X_train_array = np.hstack([X_train_num, X_train_cat])
        X_test_array = np.hstack([X_test_num, X_test_cat])
        all_features = list(pca_cols) + list(cat_cols_ohe)

        # Reconstruct DataFrames with original index
        X_train = pd.DataFrame(X_train_array, columns=all_features, index=X_train_raw.index)
        X_test = pd.DataFrame(X_test_array, columns=all_features, index=X_test_raw.index)

        # Original data for reference
        df_train = df.loc[X_train.index].reset_index(drop=True)
        df_test = df.loc[X_test.index].reset_index(drop=True)

        return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True), df_train.reset_index(drop=True), df_test.reset_index(drop=True), scaler

    def filter_by_date(self, year, month=1, day=1, date_col="date"):
        """Remove all rows after a given year/month/day."""
        
        df = self.df.copy()

        # Ensure datetime
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        # Create cutoff date
        cutoff = pd.Timestamp(year=year, month=month, day=day)

        # Filter
        df = df[df[date_col] > cutoff]

        # Update internal dataframe
        self.df = df.reset_index(drop=True)

        print(f"Filtered: kept {len(self.df)} rows from {cutoff.date()} onward.")

    def categorize_line_movement(self):
        columns = ['red_ud_to_fav_close1','red_ud_to_fav_close2','blue_ud_to_fav_close1','blue_ud_to_fav_close2',
                'red_stayed_fav_close1','red_stayed_fav_close2','blue_stayed_fav_close1','blue_stayed_fav_close2',
                'red_fav_to_ud_close1','red_fav_to_ud_close2','blue_fav_to_ud_close1','blue_fav_to_ud_close2',
                'red_stayed_dog_close1','red_stayed_dog_close2','blue_stayed_dog_close1','blue_stayed_dog_close2']

        df = self.df.copy()
        
        # Replace True with category string, keep False as-is
        for c in columns: 
            c_split = c.split('_')
            cat = "_".join(c_split[1:-1])
            df[c] = df[c].apply(lambda x: cat if x else x)  # keep False values

        # Group columns
        red_cols_line1 = [c for c in columns if c.startswith('red') and c.endswith('close1')]
        red_cols_line2 = [c for c in columns if c.startswith('red') and c.endswith('close2')]
        blue_cols_line1 = [c for c in columns if c.startswith('blue') and c.endswith('close1')]
        blue_cols_line2 = [c for c in columns if c.startswith('blue') and c.endswith('close2')]

        # Pick the first "truthy" value (string) per row; False will be ignored
        df['red_movement_close1_cat'] = df[red_cols_line1].replace(False, np.nan).bfill(axis=1).iloc[:, 0]
        df['red_movement_close2_cat'] = df[red_cols_line2].replace(False, np.nan).bfill(axis=1).iloc[:, 0]
        df['blue_movement_close1_cat'] = df[blue_cols_line1].replace(False, np.nan).bfill(axis=1).iloc[:, 0]
        df['blue_movement_close2_cat'] = df[blue_cols_line2].replace(False, np.nan).bfill(axis=1).iloc[:, 0]

        self.df = df.drop(columns=columns).reset_index(drop=True)
        print('CAT SHAPE', self.df.shape)


    def split_by_gender(self, df_, selected_features, gmm_men, gmm_women, scaler_men, scaler_women):

        # --- CLEAN BASE DATAFRAME ---
        df = df_.dropna().copy()
        df = df[df['winner'] != 2]

        # Drop Catch Weight
        df = df[~df['weight_class'].str.contains('Catch Weight', case=False, na=False)]

        # Identify Men/Women
        df['weight_group'] = df['weight_class'].apply(
            lambda x: 'Women' if 'Women' in str(x) else 'Men'
        )

        # --- Feature definitions ---
        blue_feats = [
            'td_defense_pct_blue','td_accuracy_pct_blue','control_pm_blue',
            'sig_str_defense_pct_blue','sig_str_accuracy_pct_blue',
            'sig_str_landed_pm_blue','sig_str_absorbed_pm_blue',
            'kd_pm_blue','total_fight_time_blue','age_blue','reach_blue',
            'total_bonus_blue','elo_blue','win_pct_blue','ko_losses_blue'
        ]

        red_feats = [
            'td_defense_pct_red','td_accuracy_pct_red','control_pm_red',
            'sig_str_defense_pct_red','sig_str_accuracy_pct_red',
            'sig_str_landed_pm_red','sig_str_absorbed_pm_red','kd_pm_red',
            'total_fight_time_red','age_red','reach_red','total_bonus_red',
            'elo_red','win_pct_red','ko_losses_red'
        ]

        # --- SPLIT MEN + WOMEN ---
        df_men = df[df['weight_group'] == 'Men'].reset_index(drop=True)
        df_women = df[df['weight_group'] == 'Women'].reset_index(drop=True)

        # ---------------------------------------------------------
        # Helper function to add GMM cluster labels for one gender
        # ---------------------------------------------------------
        def add_cluster_labels(df_group, gmm, scaler):

            red_labels = gmm.predict(scaler.transform(df_group[red_feats].values))
            blue_labels = gmm.predict(scaler.transform(df_group[blue_feats].values))

            df_group['red_gmm_cluster'] = pd.Series(red_labels).astype('category')
            df_group['blue_gmm_cluster'] = pd.Series(blue_labels).astype('category')
            print(f"Added GMM clusters: {df_group['red_gmm_cluster'].nunique()} red clusters, "
                  f"{df_group['blue_gmm_cluster'].nunique()} blue clusters.")

            return df_group

        # Add clusters for each gender using their own model
        df_men = add_cluster_labels(df_men, gmm_men, scaler_men)
        df_women = add_cluster_labels(df_women, gmm_women, scaler_women)

        # ---------------------------------------------------------
        # Process each gender separately (train/test + one-hot)
        # ---------------------------------------------------------
        def process_group(df_group):

            y = df_group['winner']
            df_model = df_group[selected_features].copy()

            feature_cols = [
                c for c in df_model.columns
                if c not in self.non_features + [self.target_col]
            ]
            X = df_model[feature_cols]

            # Identify feature types
            num_cols = X.select_dtypes(include='number').columns.tolist()
            cat_cols = X.select_dtypes(exclude='number').columns.tolist()

            # Train/test split
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y,
                train_size=self.train_size,
                random_state=self.random_state,
                shuffle=False
            )

            # Preprocessing
            transformers = []
            if num_cols:
                transformers.append(("num", StandardScaler(), num_cols))
            if cat_cols:
                transformers.append(
                    ("cat", OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False),
                    cat_cols)
                )

            print('Categorical columns:', cat_cols )
            preprocessor = ColumnTransformer(transformers=transformers)

            X_train_arr = preprocessor.fit_transform(X_train_raw)
            X_test_arr  = preprocessor.transform(X_test_raw)

            # Build final feature names
            all_features = list(num_cols)
            if cat_cols:
                cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
                all_features += list(cat_features)

            # Build final DataFrames
            X_train = pd.DataFrame(X_train_arr, columns=all_features, index=X_train_raw.index)
            X_test  = pd.DataFrame(X_test_arr,  columns=all_features, index=X_test_raw.index)

            # Full DataFrames (ALL columns)
            df_train_full = df_group.loc[X_train.index].reset_index(drop=True)
            df_test_full  = df_group.loc[X_test.index].reset_index(drop=True)

            return X_train, X_test, y_train, y_test, df_train_full, df_test_full

        # Process men and women
        men_results   = process_group(df_men)
        women_results = process_group(df_women)

        # Return results as dictionary
        return {
            "men": {
                "X_train": men_results[0],
                "X_test":  men_results[1],
                "y_train": men_results[2],
                "y_test":  men_results[3],
                "df_train_full": men_results[4],
                "df_test_full":  men_results[5]
            },
            "women": {
                "X_train": women_results[0],
                "X_test":  women_results[1],
                "y_train": women_results[2],
                "y_test":  women_results[3],
                "df_train_full": women_results[4],
                "df_test_full":  women_results[5]
            }
        }