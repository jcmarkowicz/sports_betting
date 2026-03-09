import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np 
import pandas as pd 
from DataPipeline.FeatureEngineering.BuildFeatures.final_feats_df import non_rolling_stats
from DataPipeline.FeatureEngineering.BuildFeatures.fight_time_feats import single_event_features, upcoming_event_features
from DataPipeline.FeatureEngineering.BuildFeatures.rolling_stats import apply_rolling_stats
from DataPipeline.FeatureEngineering.BuildFeatures.odds_features import build_odds_features
from DataPipeline.FeatureEngineering.BuildFeatures.feature_functions import count_fav_dog

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]

class FeatureEngineering: 
    """Requires df with all stats and odds merged, computes ai model features"""

    def __init__(self):
        pass

    def standardize_features(self, df):
        single_features = single_event_features(df)
         #rolling features currently relies on these column names 
        rolling_features = apply_rolling_stats(single_features)
        all_features = non_rolling_stats(rolling_features)
        return all_features
    
    def build_all_stats(self, stats_df, upcoming_stats, odds_df, upcoming_odds):

        # combine odds history and upcoming odds, build odds features
        total_odds = pd.concat([odds_df, upcoming_odds]).reset_index(drop=True) # combine upcoming odds and odds history
        total_odds = build_odds_features(total_odds)
        print(BASE_DIR)
        total_odds.to_csv(BASE_DIR / 'data/features_test_files/all_odds_features.csv', index=False)

        # compute features for past events
        past_event_stats = single_event_features(stats_df.copy())
        past_event_stats = past_event_stats.loc[:, ~past_event_stats.columns.str.contains('^Unnamed')]
        past_event_stats.to_csv(BASE_DIR / 'data/features_test_files/ufc_single_event_features.csv', index=False)
        print(f'Fight time feats shape" {past_event_stats.shape}')

        # features for upcoming event
        upcoming_single_event = upcoming_event_features(upcoming_stats)
        print('Upcoming single event features shape:', upcoming_single_event.shape)
        
        # Create empty rows/columns for pre fight stats 
        exclude_cols = ['fighter_red', 'fighter_blue']
        stats_cols = [col for col in past_event_stats.columns if col not in exclude_cols]
        upcoming_features_NA = pd.DataFrame(index=range(upcoming_single_event.shape[0]), columns=stats_cols) 

        # Combine fighter names with NaN stats
        empty_df = pd.concat([upcoming_single_event[exclude_cols], upcoming_features_NA], axis=1) 
        empty_df.columns = past_event_stats.columns # Assumes same order and length

        # Overwrite the scraped columns with actual values
        scraped_columns = [
            'fighter_red', 'fighter_blue', 'weight_class', 'event_date',
            'reach_red', 'reach_blue', 'height_red', 'height_blue',
            'age_red', 'age_blue', 'event_location', 'title_fight'
        ]
        
        for col in scraped_columns:
            if col in empty_df.columns and col in upcoming_single_event.columns:
                empty_df[col] = upcoming_single_event[col]

        # Make sure event_date is datetime
        empty_df['date'] = pd.to_datetime(empty_df['event_date'], format="%Y-%m-%d")
        combined_df = pd.concat([empty_df, past_event_stats], axis=0).reset_index(drop=True) # Combine with past event stats

        # compute rolling stats
        rolling_fp = BASE_DIR / 'data\features_test_files/ufc_new_rolling.csv'
        rolling_df = apply_rolling_stats(combined_df) #sort here
        rolling_df.to_csv(rolling_fp, index=False)
        print('Rolling features shape:', rolling_df.shape)

        # compute non rolling stats
        total_df = non_rolling_stats(rolling_df)
        total_df.to_csv(BASE_DIR / 'data/features_test_files/new_combined.csv', index=False)
        total_df = total_df.copy()
        print('Total features shape:', total_df.shape)

        # merge odds and features
        merged_df = self.standardized_merge(total_df, total_odds)
        print('Merged df shape:', merged_df.shape)
        merged_df.to_csv(BASE_DIR / 'data/features_test_files/stats_odds_merged.csv', index=False)

        # counts of fav and dog 
        cols = ['fav_counts_red','dog_counts_red','fav_counts_blue','dog_counts_blue']
        r = count_fav_dog(merged_df)
        for i, col in enumerate(cols):
            merged_df[col] = r[:, i] # work around for bug assigning np arrays directly

        merged_df['fav_counts_diff'] = merged_df['fav_counts_red'] - merged_df['fav_counts_blue']
        merged_df['dog_counts_diff'] = merged_df['dog_counts_red'] - merged_df['dog_counts_blue']

        # seperate the upcoming fight stats/odds from the history
        odds_stats_history = merged_df.iloc[:-upcoming_stats.shape[0], :]
        upcoming_df = merged_df.iloc[-upcoming_stats.shape[0]:, :]

        return odds_stats_history, upcoming_df

    def standardize_dates(self, stats, odds):
        # dates in odds_df/stats_df are +- 1 day apart, need these to be equal 

        # iterate through stats_df rows, get target date fropm stats
        for i, row in stats.iterrows():

            #find all matchups between red and blue fighter in odds
            mask = (odds['red_clean'] == row['red_clean']) & (odds['blue_clean'] == row['blue_clean']) 

            # if odds contains the stats matchup 
            if mask.any():
                for i in odds[mask].index:

                    # iterate and find dates of each red/blue matchup 
                    target_date = odds.at[i, 'date']

                    # check if current stats date matches with +- odds date 
                    if (row['date'] - pd.Timedelta(days=1) == target_date) or (row['date'] + pd.Timedelta(days=1) == target_date):
                        # if found, replace odds_date with stats date for standardization 
                        odds.at[i, 'date'] = row['date'] # set the date in odds to the date in stats 
        return odds 

    def clean_col(self, col):
        col = col.str.lower() \
            .str.replace('-', ' ', regex=False) \
            .str.replace('.', '', regex=False) \
            .str.replace("'", '', regex=False) \
            .str.replace(r'\bsaint\b', 'st', case=False, regex=True) # add space here for fixxing other fighter names 
        return col 
    
    def red_fighter_matching(self):
        red_map = {
            'A.J. Dobson': 'AJ Dobson',
            'A.J. Fletcher': 'AJ Fletcher',
            'Alvaro Herrera': 'Alvaro Herrera Mendoza',
            'Antonio Rogerio Nogueira': 'Rogerio Nogueira',
            'Asu Almabaev': 'Asu Almabayev',
            'B.J. Penn': 'BJ Penn',
            'Benoit Saint-Denis': 'Benoit Saint Denis',
            'C.B. Dollaway': 'CB Dollaway',
            'C.J. Vergara': 'CJ Vergara',
            'Carls John de Tomas': 'Carls John De Tomas',
            'Chris de La Rocha': 'Chris de la Rocha',
            'Christian Quinonez': 'Cristian Quinonez',
            'Cj Vergara': 'CJ Vergara',
            'Cm Punk': 'CM Punk',
            "Da'mon Blackshear": "Da'Mon Blackshear",
            'Damarques Johnson': 'DaMarques Johnson',
            'Dan Arguetta': 'Dan Argueta',
            'Danaa Batgerel': 'Batgerel Danaa',
            'Darmon Blackshear': "Da'Mon Blackshear",
            'Deanna Bennett': 'DeAnna Bennett',
            "Don'tale Mayes": "Don'Tale Mayes",
            'Elizeu Zaleski': 'Elizeu Zaleski dos Santos',
            'Elizeu Zaleski Dos Santos': 'Elizeu Zaleski dos Santos',
            'Felipe Dos Santos': 'Felipe dos Santos',
            'J.J. Aldrich': 'JJ Aldrich',
            'J.P. Buys': 'JP Buys',
            'Jake O Brien': "Jake O'Brien",
            'Jason Macdonald': 'Jason MacDonald',
            'Jeongyeong Lee': 'JeongYeong Lee',
            'Jessica Rakozczy': 'Jessica Rakoczy',
            'Joanne Calderwood': 'Joanne Wood',
            'Joe Gigliotti': 'Joe Giannetti',
            'Joshua van': 'Joshua Van',
            'Juan Puig': 'Juan Manuel Puig',
            'Jun Yong Park': 'JunYong Park',
            'K.J. Noons': 'KJ Noons',
            'Kai Kamaka III': 'Kai Kamaka',
            'Luiz Dutra Jr.': 'Luiz Dutra',
            'Magomed Bibulatov': 'Bibulatov Magomed',
            'Manny Gamburyan': 'Manvel Gamburyan',
            'Marcio Alexandre Jr.': 'Marcio Alexandre Junior',
            'Marcus Levesseur': 'Marcus LeVesseur',
            'Mark de La Rosa': 'Mark De La Rosa',
            'Matt van Buren': 'Matt Van Buren',
            'Mike de La Torre': 'Mike de la Torre',
            'Mizuki Inoue': 'Mizuki',
            'Montana de La Rosa': 'Montana De La Rosa',
            'Montserrat Rendon': 'Montse Rendon',
            'Ning Guangyou': 'Guangyou Ning',
            'Ovince St. Preux': 'Ovince Saint Preux',
            'Paige Vanzant': 'Paige VanZant',
            'Park Hyun-Sung': 'HyunSung Park',
            'Phil de Fries': 'Philip De Fries',
            'Philip de Fries': 'Philip De Fries',
            'Rameau Sokoudjou': 'Rameau Thierry Sokoudjou',
            'Roldan Sangcha-An': "Roldan Sangcha'an",
            'Rory Macdonald': 'Rory MacDonald',
            'Ryan Laflare': 'Ryan LaFlare',
            'Ryan Macdonald': 'Ryan MacDonald',
            'Seung Woo Choi': 'SeungWoo Choi',
            'Seungwoo Choi': 'SeungWoo Choi',
            'Silvana Gomez Juarez.': 'Silvana Gomez Juarez',
            'Su Young You': 'SuYoung You',
            'Sumudaerji Sumudaerji': 'Sumudaerji',
            'T.J. Dillashaw': 'TJ Dillashaw',
            'T.J. Grant': 'TJ Grant',
            'T.J. Waldburger': 'TJ Waldburger',
            'Thiago de Oliveira Perpetuo': 'Tiago dos Santos e Silva',
            'Tiequan Zhang': 'Zhang Tiequan',
            'Vernon Ramos': 'Vernon Ramos Ho',
            'Victoria Dudakova': 'Viktoriia Dudakova',
            'Waldo Cortes-Acosta': 'Waldo Cortes Acosta',
            'Wang Anying': 'Anying Wang',
            'Wang Sai': 'Sai Wang',
            'Yadier Delvalle': 'Yadier del Valle',
            'Yadong Song': 'Song Yadong',
            'Yanan Wu': 'Wu Yanan',
            'Yorgan de Castro': 'Yorgan De Castro'
            }
        return red_map
    
    def blue_fighter_matching(self):
        blue_map = {
            "A.J. Dobson": "AJ Dobson",
            "A.J. Fletcher": "AJ Fletcher",
            "Aj Cunningham": "AJ Cunningham",
            "Aj Dobson": "AJ Dobson",
            "Aj Fletcher": "AJ Fletcher",
            "Alatengheili Alateng": "Alatengheili",
            "Alex Munoz": "Alexander Munoz",
            "Alex da Silva": "Alex Da Silva",
            "Alexander Torres": "Alex Torres",
            "Ali Al Qaisi": "Ali AlQaisi",
            "Alvaro Herrera": "Alvaro Herrera Mendoza",
            "Antonio Dos Santos Jr.": "Antonio Dos Santos",
            "Antonio Rogerio Nogueira": "Rogerio Nogueira",
            "Aori Qileng": "Qileng Aori", 
            "B.J. Penn": "BJ Penn",
            "Benoit Saint-Denis": "Benoit Saint Denis",
            "Benoit St.Denis": "Benoit Saint Denis",
            "Bharat Khandare": "Bharat Kandare",
            "C.B. Dollaway": "CB Dollaway",
            "C.J. Keith": "CJ Keith",
            "C.J. Vergara": "CJ Vergara",
            "Cameron Vancamp": "Cameron VanCamp",
            "Carlos Leal Miranda": "Carlos Leal",
            "Carls John de Tomas": "Carls John De Tomas",
            "Chang Ho Lee": "ChangHo Lee",
            "Chris de La Rocha": "Chris de la Rocha",
            "Cj Vergara": "CJ Vergara",
            "Damon Blackshear": "Da'Mon Blackshear",
            "Dan Spohn": "Daniel Spohn",
            "Danaa Batgerel": "Batgerel Danaa",
            "Danaa Batgerel.": "Batgerel Danaa",
            "Daniel Argueta.": "Dan Argueta",
            "Da'mon Blackshear": "Da'Mon Blackshear",
            "Damarques Johnson": "DaMarques Johnson",
            "David Galera": "Dave Galera",
            "Dennis Buzukia": "Dennis Buzukja",
            "Dmitriy Sosnovskiy": "Dmitry Sosnovskiy",
            "Don'tale Mayes": "Don'Tale Mayes",
            "Dong Hoon Choi": "DongHun Choi",
            "Elizeu Zaleski": "Elizeu Zaleski dos Santos",
            "Elizeu Zaleski Dos Santos": "Elizeu Zaleski dos Santos",
            "Felipe Dos Santos": "Felipe dos Santos",
            "Humberto Brown": "Humberto Brown Morrison",
            "Hyun Sung Park": "HyunSung Park",
            "Hyunsung Park": "HyunSung Park",
            "Heili Alateng": "Alatengheili",
            "J.C. Cottrell": "JC Cottrell",
            "J.J. Aldrich": "JJ Aldrich",
            "J.P. Buys": "JP Buys",
            "Jake O Brien": "Jake O'Brien",
            "Jason Macdonald": "Jason MacDonald",
            "Jeong Yeong Lee": "JeongYeong Lee",
            "Jessica Rakozczy": "Jessica Rakoczy",
            "Jj Aldrich": "JJ Aldrich",
            "Joanne Calderwood": "Joanne Wood",
            "Joo Sang Yoo": "JooSang Yoo",
            "Joosang Yoo": "JooSang Yoo",
            "Jorge Oliveira": "Jorge de Oliveira",
            "Jose Medina": "Jose Daniel Medina",
            "Joseph Duffey": "Joe Duffy",
            "Joseph Duffy": "Joe Duffy",
            "Josh Burkman": "Joshua Burkman",
            "Joshua van": "Joshua Van",
            "Jp Buys.": "JP Buys",
            "Jun Yong Park": "JunYong Park",
            "Junyong Park": "JunYong Park",
            "K.B. Bhullar": "KB Bhullar",
            "K.J. Noons": "KJ Noons",
            "Kai Kamaka III": "Kai Kamaka",
            "Kai Kamaka Iii.": "Kai Kamaka",
            "Kb Bhullar": "KB Bhullar",
            "Larissa Moreira Pacheco": "Larissa Pacheco",
            "Lipeng Zhang": "Zhang Lipeng",
            "Luiz Dutra Jr.": "Luiz Dutra",
            "Magomed Bibulatov": "Bibulatov Magomed",
            "Manny Gamburyan": "Manvel Gamburyan",
            "Marcio Alexandre Jr.": "Marcio Alexandre Junior",
            "Marcus Levesseur": "Marcus LeVesseur",
            "Mark de La Rosa": "Mark De La Rosa",
            "Marquel Mederos": "MarQuel Mederos",
            "Martin Sano Jr.": "Martin Sano",
            "Matt van Buren": "Matt Van Buren",
            "Michael Aswell": "Michael Aswell Jr.",
            "Mike de La Torre": "Mike de la Torre",
            "Mizuki Inoue": "Mizuki",
            "Montana de La Rosa": "Montana De La Rosa",
            "Montserrat Rendon": "Montse Rendon",
            "Montserrat Ruiz": "Montserrat Conejo Ruiz",
            "Ning Guangyou": "Guangyou Ning",
            "Ovince St. Preux": "Ovince Saint Preux",
            "Paige Vanzant": "Paige VanZant",
            "Phil de Fries": "Philip De Fries",
            "Philip de Fries": "Philip De Fries",
            "Qileng Aori": "Aori Qileng",
            "Raffael Cerqueira": "Rafael Cerqueira",
            "Rameau Sokoudjou": "Rameau Thierry Sokoudjou",
            "Roberto Sanchez": "Robert Sanchez",
            "Rodrigo Goiana de Lima": "Rodrigo de Lima",
            "Rodrigo Lima": "Rodrigo de Lima",
            "Roldan Sangcha-An": "Roldan Sangcha'an",
            "Ronnys Torres": "Ronys Torres",
            "Rory Macdonald": "Rory MacDonald",
            "Seung Woo Choi": "SeungWoo Choi",
            "Seungwoo Choi": "SeungWoo Choi",
            "Shane Del Rosario": "Shane del Rosario",
            "Steven Kennedy": "Steve Kennedy",
            "Su Mudaerji": "Sumudaerji",
            "Su Young You": "SuYoung You",
            "Sumudaerji Sumudaerji": "Sumudaerji",
            "Suyoung You": "SuYoung You",
            "Suyoung Yu": "SuYoung You",
            "T.J. Brown": "TJ Brown",
            "T.J. Dillashaw": "TJ Dillashaw",
            "T.J. Grant": "TJ Grant",
            "T.J. Laramie": "TJ Laramie",
            "T.J. Obrien": "TJ O'Brien",
            "T.J. Waldburger": "TJ Waldburger",
            "Thomas Egan": "Tom Egan",
            "Timothy Cuamba": "Timmy Cuamba",
            "Tj Brown": "TJ Brown",
            "Tj Dillashaw": "TJ Dillashaw",
            "Tj Laramie": "TJ Laramie",
            "Tom Deblass": "Tom DeBlass",
            "Tony Desouza": "Tony DeSouza",
            "Treston Vines": "Tre'ston Vines",
            "Victoria Dudakova": "Viktoriia Dudakova",
            "Vincent Cachero": "Vince Cachero",
            "Vinicius Kappke de Quieroz": "Vinicius Queiroz",
            "Waldo Cortes-Acosta": "Waldo Cortes Acosta",
            "Wuliji Buren": "Wulijiburen",
            "Yadier Del Valle": "Yadier del Valle",
            "Yanan Wu": "Wu Yanan",
            "Yang Jianping": "Jianping Yang",
            'Yadong Song': 'Song Yadong',
            "Yorgan de Castro": "Yorgan De Castro",
            "Zach Scroggin": "Zachary Scroggin",
            "Zhuikui Yao": "Yao Zhikui"
        }
        return blue_map
    
    def align_fighter_names(self, odds_df, red_mapping, blue_mapping):
        """
        Aligns fighter names in odds_df to match stats_df using mapping dictionaries.
        
        Args:
            odds_df (pd.DataFrame): DataFrame with columns 'red_fighter' and 'blue_fighter'.
            stats_df (pd.DataFrame): DataFrame with columns 'fighter_red' and 'fighter_blue'.
            red_mapping (dict): Mapping for red fighters: odds_name -> stats_name (or None).
            blue_mapping (dict): Mapping for blue fighters: odds_name -> stats_name (or None).
        
        Returns:
            pd.DataFrame: Copy of odds_df with fighter names aligned to stats_df.
        """
        df = odds_df.copy()
        df['red_fighter'] = df['red_fighter'].map(red_mapping).fillna(df['red_fighter'])
        df['blue_fighter'] = df['blue_fighter'].map(blue_mapping).fillna(df['blue_fighter'])
        
        return df
                
    def standardized_merge(self, stats_df, odds_df):
        odds_df = odds_df.loc[:, ~odds_df.columns.str.contains('^Unnamed')] # filter out columns that contain 'Unamed'

        stats = stats_df.reset_index(drop=True).copy()
        odds = odds_df.reset_index(drop=True).copy()

        odds['date'] = pd.to_datetime(odds['event_date'])
        stats['date'] = pd.to_datetime(stats['date'])

        odds = odds.dropna(subset=['date']) # remove rows based on rows in 'date' that are NA 
        odds = odds.drop_duplicates(subset=['red_fighter', 'blue_fighter', 'date'], keep='first').reset_index(drop=True)

        # clean names in stats col 
        stats['red_clean'] = self.clean_col(stats['fighter_red']) # clean names so that the names in odds and stats match 
        stats['blue_clean'] = self.clean_col(stats['fighter_blue'])

        stats['red_fighter_stats'] = stats['fighter_red'] # fighter names coming from stats df 
        stats['blue_fighter_stats'] = stats['fighter_blue']

        # map fighter names from odds to stats names
        red_map = self.red_fighter_matching()
        blue_map = self.blue_fighter_matching()
        odds = self.align_fighter_names(odds, red_map, blue_map)

        # clean names in odds col 
        odds['red_clean'] = self.clean_col(odds['red_fighter']) # in odds its red_fighter
        odds['blue_clean'] = self.clean_col(odds['blue_fighter'])

        odds['red_fighter_odds'] = odds['red_fighter']
        odds['blue_fighter_odds'] = odds['blue_fighter']

        odds = self.standardize_dates(stats, odds)
        odds = odds.drop_duplicates(subset=['red_clean', 'blue_clean', 'date'], keep='first').reset_index(drop=True)
        odds.to_csv(BASE_DIR / 'data\features_test_files\look_at_odds.csv')

        # drop duplicate columns before merge
        columns_to_drop = ['fighter_red', 'fighter_blue']  # list of columns
        stats = stats.drop(columns=columns_to_drop)
        columns_to_drop = ['red_fighter', 'blue_fighter']  # list of columns
        odds = odds.drop(columns=columns_to_drop)

        # merge on clean fighter names and date 
        new_df = pd.merge(stats, odds, on=['red_clean', 'blue_clean', 'date'], how='left')
        new_df = new_df.loc[:, ~new_df.columns.str.contains('^Unnamed')]
        new_df = new_df.rename(columns={'red_clean': 'fighter_red', 'blue_clean':'fighter_blue'})
        new_df = new_df.drop_duplicates() # duplicat rows because of undstandardized fighter name columns 
        new_df = new_df.sort_values(by='date', ascending=True).reset_index(drop=True)

        return new_df



        

   