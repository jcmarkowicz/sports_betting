import numpy as np 
import pandas as pd 
from collections import defaultdict

def elo_rating(df,k):
    
    elo_dic = defaultdict(list)
    red_elo = []
    blue_elo = []
    for _, row in df.iterrows(): 
        red_name = row['fighter_red']
        blue_name = row['fighter_blue']

        if red_name not in elo_dic:
            elo_dic[red_name] = [1500]
        
        if blue_name not in elo_dic:
            elo_dic[blue_name] = [1500]

        prev_blue = elo_dic[blue_name][-1] #elo pre fight
        prev_red = elo_dic[red_name][-1]
        red_elo.append(prev_red)
        blue_elo.append(prev_blue)
        
        if row['winner'] == 1 and pd.notna(row['winner']):

            d = prev_red - prev_blue
            mu = 1 / (1 + 10**(-d/400))
            red_new = prev_red + k * (1-mu) #red wins

            d = prev_blue - prev_red
            mu = 1 / (1 + 10**(-d/400))
            blue_new = prev_blue + k * (0-mu) #blue loses
            
            elo_dic[red_name].append(red_new)
            elo_dic[blue_name].append(blue_new)

        if row['winner'] == 0 and pd.notna(row['winner']):

            d = prev_blue - prev_red
            mu = 1 / (1 + 10**(-d/400))
            blue_new = prev_blue + k * (1-mu) #blue wins

            d = prev_red - prev_blue
            mu = 1 / (1 + 10**(-d/400))
            red_new = prev_red + k * (0-mu) #red loses
            
            elo_dic[red_name].append(red_new)
            elo_dic[blue_name].append(blue_new)

    return np.column_stack([red_elo, blue_elo])