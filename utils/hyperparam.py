import pandas as pd
from utils.plotting import *


def pick_params(results_csv):
    cv_df = pd.read_csv(results_csv)
    cv_df = cv_df[cv_df['epsilon'] <= 1]
    sorted_df = cv_df.sort_values(by=['cv_score_avg'], ascending=False)
    total_look = len(sorted_df)

    top_score = sorted_df.iloc[1]['cv_score_avg']
    top_score_sd = sorted_df.iloc[1]['cv_score_std']
    score_cutoff = top_score - top_score_sd

    above_cutoff = sorted_df['cv_score_avg'] > score_cutoff
    good_count = above_cutoff.value_counts()[True]
    good_percent = (good_count / total_look) * 100
    
 #   print('Score Cutoff is: ' + str(score_cutoff) + ' and ' +
 #         str(good_count) + ' / ' + str(total_look) + ' = '
 #         + str(good_percent) + '% of settings meet it:')

    print ( f'Score Cutoff is: {score_cutoff:.3f} and {good_count} / {total_look} = {good_percent:.3f} % of settings meet it.' )
    print ( f'Next, we display parameter values sorted by how often they appear in the admissable set.' )    
    
    good_params = sorted_df[above_cutoff]

    density_vote = good_params['density'].value_counts()
    density_vote = density_vote.sort_values(ascending=False)
    
    n_est_vote =  good_params['n_estimators'].value_counts()
    n_est_vote = n_est_vote.sort_values(ascending=False)

    learn_rate_vote = good_params['learning_rate'].value_counts()
    learn_rate_vote = learn_rate_vote.sort_values(ascending=False)
    
    print('\n Density by popularity:')
    print(density_vote.to_string())

    print('\n Learning Rate by popularity:')
    print(learn_rate_vote.to_string())
        
    print('\n Number of Rounds by popularity:')
    print(n_est_vote.to_string())

