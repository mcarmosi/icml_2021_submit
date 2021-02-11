import pandas as pd

from utils.plotting import *

min_DP_lvl = ['(1, 0)-DP', '(1, sLIN)-DP']

boost_oneRule_curve = {'weakl' : 'oneRule',
                       'adult':
                       {'param_spec':
                        {'density': 0.25,
                         'learning_rate': 0.25,
                         'max_rounds': 100},
                         'privacy_list': min_DP_lvl + ['(0.5, sLIN)-DP'],
                         'np_boost_baseline': 0.25,
                         'np_weak_baseline':
                        {'cv_score_avg': 0.7819804624823031,
                         'cv_score_std': 0.00792869693158956}},
                       
                       'codrna':
                       {'param_spec':
                        {'density': 0.2,
                         'learning_rate': 0.75,
                         'max_rounds': 100},
                         'privacy_list': min_DP_lvl + ['(0.25, 0)-DP'],
                         'np_boost_baseline': 0.25,
                         'np_weak_baseline':
                         {'cv_score_avg': 0.7366929953189331,
                          'cv_score_std': 0.011261421572243647} 
                        }
}

boost_DPTopDown_curve = {'weakl': 'DPTopDown', 
                         'adult':
                        {'param_spec':
                         {'density': 0.25, 
                         'learning_rate': 0.25, 
                         'max_depth': 3, 
                         'max_splits': 3, 
                         'max_rounds': 20},
                         'privacy_list': min_DP_lvl + ['(0.1, 0)-DP'],
                         'np_boost_baseline': 0.25,
                         'np_weak_baseline':
                         {'cv_score_avg': 0.7819804624823031,
                          'cv_score_std': 0.00792869693158956}
                        },

                        'codrna':
                        {'param_spec': 
                         {'density': 0.2, 
                         'learning_rate': 0.5, 
                         'max_depth':10,
                         'max_split':10,
                         'max_rounds': 20},
                         'privacy_list': min_DP_lvl + ['(0.05, sLIN)-DP'],
                         'np_boost_baseline': 0.75,
                         'np_weak_baseline':
                        {'cv_score_avg': 0.7366929953189331,
                         'cv_score_std': 0.011261421572243647} 
                        }
}

boost_RandomForest_curve = {'weakl': 'RandomForest', 
                            'adult':
                            {'param_spec':
                             {'density': 0.25, 
                              'learning_rate': 0.25, 
                              'max_random_trees': 5, 
                              'labeling_budget_ratio': 0.5,
                              'extra_random': False, 
                              'max_splits': 30, 
                              'max_depth': 13, 
                              'max_rounds': 20},
                             'privacy_list': min_DP_lvl + ['(0.5, sLIN)-DP'],
                             'np_boost_baseline': 0.25,
                             'np_weak_baseline':
                             {'cv_score_avg': 0.7819804624823031,
                              'cv_score_std': 0.00792869693158956}
                            },

                             'codrna':
                             {'param_spec':
                              {'density': 0.25, 
                              'learning_rate': 0.25, 
                              'max_random_trees': 5, 
                              'labeling_budget_ratio': 0.5,
                               'extra_random': False, 
                               'max_splits': 30, 
                               'max_depth': 14, 
                               'max_rounds': 20},
                              'privacy_list': min_DP_lvl + ['(0.75, 0)-DP'],
                              'np_boost_baseline': 0.75,
                              'np_weak_baseline':
                              {'cv_score_avg':  0.7366929953189331,
                              'cv_score_std':  0.011261421572243647}
                            }
}

# Take a dataset results and plot up n_rounds vs cv_score under
# varying privacy colors:

def boosting_fig(results_csv, baseline_csv, dataset_name, weak_l_curve, title=None):
    curve = weak_l_curve[dataset_name]
    
    cv_df = pd.read_csv(results_csv)

    chosen_params = curve['param_spec']
    chosen_privacy = curve['privacy_list']
    np_weak_baseline = curve['np_weak_baseline']

    if weak_l_curve['weakl'] == 'oneRule':
        cv_to_plot = lock_oneRule_params(cv_df, chosen_params, chosen_privacy)
        
    if weak_l_curve['weakl'] == 'DPTopDown':
        cv_to_plot = lock_DPTopDown_params(cv_df, chosen_params, chosen_privacy)

    if weak_l_curve['weakl'] == 'RandomForest':
        cv_to_plot = lock_RandomForest_params(cv_df, chosen_params, chosen_privacy)
 
    baseline_df = pd.read_csv(baseline_csv)
    baseline_df = baseline_df[baseline_df['n_estimators'] <= chosen_params['max_rounds']]
    baseline_curve = baseline_df['learning_rate'] == curve['np_boost_baseline']
    baseline_plot = baseline_df[baseline_curve]
    baseline_plot['privacy'] = 'NO-DP'
    baseline_plot['epsilon'] = 'NO-DP'
    baseline_plot['delta'] = 'NO-DP'
    baseline_plot['density'] = 'NO-DP'
    full_compare = pd.concat([cv_to_plot, baseline_plot])
    
    plot_gs_df(full_compare, 'n_estimators', 'privacy',
               'n_est', 'Number of Estimators', np_weak_baseline, title=title)


# plt.ion()
# boosting_oneRule_fig("../results/adult_RandomForest_results.csv", "../results/baseline_adult_results.csv", 'adult', boost_RandomForest_curve)

# special_subset = boosting_oneRule_fig("results/adult_OneRule_results.csv", 'adult')
# plot_gs_df(special_subset, 'n_estimators','privacy', 'n_est', 'Number of Estimators', hardcoded_baseline)  
