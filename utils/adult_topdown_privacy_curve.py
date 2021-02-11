import pandas as pd 
from utils.plotting import *

adult_baseline = {'cv_score_avg': 0.7819804624823031, 
                  'cv_score_std': 0.00792869693158956}

cv_df = pd.read_csv("results/adult_TopDown_results.csv")

privacy_curves = cv_df[  (cv_df['n_estimators'] == 20) &
                        (cv_df['learning_rate'] == 0.5) &
                        (cv_df['density'] == 0.25 ) &
                        (cv_df['delta'].isin(['0','sLIN']))
                       ]

plt.ion()

plot_gs_df(privacy_curves, 'epsilon', 'delta', 'n_est',
           'Epsilon', adult_baseline)

# plt.savefig("adult_onerule_privacy.pdf")
