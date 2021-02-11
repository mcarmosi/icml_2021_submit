import pandas as pd 
from utils.plotting import *

def plot_gs_double_baseline(cv_df, grid_param_x, grid_param_color,
                            name_x, name_color, wk_baseline,
                            bst_baseline, error_bars=True):

    _, ax = plt.subplots(1,1)
    colors = cv_df.groupby(grid_param_color)
    marked_colors = zip(colors, good_symbols)
    for (name, color), symbol in marked_colors:
        x_pts = color[grid_param_x]
        if error_bars:
            ax.errorbar(color[grid_param_x], color['cv_score_avg'],
                        yerr=color['cv_score_std'],
                        marker=symbol, label = name)
        else:
            ax.plot(color[grid_param_x], color['cv_score_avg'], '-o', label = name,
                    marker=symbol)

    ax.set_xlabel(name_color, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)

    ax.fill_between(x_pts, bst_baseline['cv_score_avg'] +
                    bst_baseline['cv_score_std'],
                    bst_baseline['cv_score_avg'] -
                    bst_baseline['cv_score_std'], alpha=0.2)

    ax.fill_between(x_pts, wk_baseline['cv_score_avg'] +
                    wk_baseline['cv_score_std'],
                    wk_baseline['cv_score_avg'] -
                    wk_baseline['cv_score_std'], alpha=0.2)

# OneRule, 50-fold crossvalidated error.
adult_wk_baseline = {'cv_score_avg': 0.7819804624823031, 
                     'cv_score_std': 0.00792869693158956}

# NonAdaBoost, OneRule, #rounds = 100, learning_rate = 0.25
# Obtained via cross-validated grid search.
adult_bst_baseline = {'cv_score_avg': 0.8421732513049880,
                      'cv_score_std': 0.0018732021699287800
                      }


cv_df = pd.read_csv("results/adult_OneRule_results.csv")

privacy_curves = cv_df[  (cv_df['n_estimators'] == 100) &
                        (cv_df['learning_rate'] == 0.25) &
                        (cv_df['density'] == 0.25 ) &
                        (cv_df['delta'].isin(['0','sLIN']))
                       ]

plt.ion()
plt.rcParams['figure.figsize'] = [15, 10]

plot_gs_double_baseline(privacy_curves, 'epsilon', 'delta', 'n_est',
                        'Epsilon', adult_wk_baseline, adult_bst_baseline)
#plt.savefig("adult_onerule_privacy.pdf")
