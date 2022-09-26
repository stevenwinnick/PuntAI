"""
Creates a new CSV file containing relevant data from NFL Play by Play 2009-2018 dataset and new columns "winner" and
"posteam_won", containing the winning team, and a boolean representing whether the team with possession won,
respectively. The original dataset can be found here:
https://www.kaggle.com/datasets/maxhorowitz/nflplaybyplay2009to2016
"""

import pandas as pd
import numpy as np

# Load data into dataframe
# Column descriptions found here: https://github.com/ryurko/nflscrapR-data/blob/master/legacy_data/README.md
df = pd.read_csv('NFL Play by Play 2009-2018 (v5).csv', index_col=False, usecols=[
                 'game_id', 'posteam', 'posteam_type', 'defteam', 'side_of_field', 'yardline_100',
                 'quarter_seconds_remaining', 'half_seconds_remaining', 'game_seconds_remaining', 
                 'down', 'goal_to_go', 'ydstogo', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining', 
                 'posteam_score', 'defteam_score', 'score_differential_post'
                 ])

# Determine game winner using last play for each game_id
last_play_each_game = df[~df['posteam'].isnull()].groupby('game_id').last().reset_index()
winners = last_play_each_game[['game_id', 'posteam', 'defteam', 'score_differential_post']]


def find_winner(score_diff, possessor, defense):
    if score_diff > 0:
        return possessor
    elif score_diff < 0:
        return defense
    else:
        return np.nan


winners['winner'] = winners.apply(lambda row: find_winner(row['score_differential_post'], row['posteam'],
                                                          row['defteam']), axis=1)

# Add winner column to original dataframe
df = df.merge(winners[['game_id', 'winner']], how='left', on='game_id')

# Add posteam_won column to original dataframe
df['posteam_won'] = np.where(df['posteam'] == df['winner'], 1, 0)

# Set posteam type to 1 if home, 0 if away
df['posteam_type'] = np.where(df['posteam_type'] == 'home', 1, 0)

# Drop plays missing data - includes kickoffs (where no down is listed)
df.dropna()

# Normalize values between 0 and 1
df['yardline_100'] = df['yardline_100'] / 100
df['quarter_seconds_remaining'] = df['quarter_seconds_remaining'] / 900
df['half_seconds_remaining'] = df['half_seconds_remaining'] / 1800
df['game_seconds_remaining'] = df['game_seconds_remaining'] / 3600
df['down'] = df['down'] / 4
df['ydstogo'] = df['ydstogo'] / 100
df['posteam_timeouts_remaining'] = df['posteam_timeouts_remaining'] / 3
df['defteam_timeouts_remaining'] = df['defteam_timeouts_remaining'] / 3
df['posteam_score'] = df['posteam_score'] / 100
df['defteam_score'] = df['defteam_score'] / 100

# Save dataframe to new csv file
df.to_csv('NormalizedPlayData.csv', index=False, columns=[
          'posteam_type', 'yardline_100', 'quarter_seconds_remaining', 'half_seconds_remaining', 
          'game_seconds_remaining', 'down', 'goal_to_go', 'ydstogo', 'posteam_timeouts_remaining', 
          'defteam_timeouts_remaining', 'posteam_score', 'defteam_score', 'posteam_won'
          ])
