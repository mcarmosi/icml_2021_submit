import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from weak_learner.TopDownDT import DPTopDownTree
from weak_learner.DPRandomForest import DPRandomForest

good_symbols = ['o', 's', 'P', '*', 'X', 'v', '1', 'D', '^', '<', '>', '4', 'x']

def cv_as_df(clf_gridsearch):
    cv_results = clf_gridsearch.cv_results_
    params_df = pd.DataFrame(cv_results["params"])
    score_df = pd.DataFrame(cv_results["mean_test_score"], columns=["cv_score_avg"])
    std_score_df = pd.DataFrame(cv_results["std_test_score"], columns=["cv_score_std"])
    return pd.concat([params_df, score_df, std_score_df], axis=1)

def prettyprint_privacy(cv_df):
    epsilons = [x.global_epsilon for x in cv_df['base_estimator']]
    deltas = [x.global_delta for x in cv_df['base_estimator']]
    budget_str = [format_dp(epsilon, delta) for epsilon, delta in
                  zip(epsilons, deltas)]
    budget_df = pd.DataFrame({'epsilon':epsilons, 'delta':deltas})
    pretty_budget_df = pd.DataFrame({'privacy':budget_str})
    return pd.concat([cv_df, pretty_budget_df, budget_df], axis=1)

def format_dp(epsilon, delta):
    return '(' + str(epsilon) + ', ' + str(delta) + ')-DP'

def lock_oneRule_params(cv_df, param_spec, privacy_list):
    return lock_params(cv_df, param_spec, privacy_list)

def lock_DPTopDown_params(cv_df, param_spec, privacy_list):
    filtered_params = lock_params(cv_df, param_spec, privacy_list)
    max_depths = [eval(x).max_depth for x in filtered_params['base_estimator']]
    filtered_params['max_depth'] = max_depths
    test = param_spec['max_depth'] == filtered_params['max_depth']
    return filtered_params[test]

def lock_RandomForest_params(cv_df, param_spec, privacy_list):
    filtered_params = lock_params(cv_df, param_spec, privacy_list)
    max_depths = [eval(x).max_depth for x in filtered_params['base_estimator']]
    filtered_params['max_depth'] = max_depths
    labeling_budget_ratio = [eval(x).labeling_budget_ratio for x in filtered_params['base_estimator']]
    filtered_params['labeling_budget_ratio'] = labeling_budget_ratio
    max_splits = [eval(x).max_splits for x in filtered_params['base_estimator']]
    filtered_params['max_splits'] = max_splits
    max_random_trees = [eval(x).max_random_trees for x in filtered_params['base_estimator']]
    filtered_params['max_random_trees'] = max_random_trees
    test = ((param_spec['max_depth'] == filtered_params['max_depth']) &
            (param_spec['max_splits'] == filtered_params['max_splits']) &
            (param_spec['labeling_budget_ratio'] == filtered_params['labeling_budget_ratio']) &
            (param_spec['max_random_trees'] == filtered_params['max_random_trees']))
    return filtered_params[test]

def lock_params(cv_df, param_spec, privacy_list):
    return cv_df[(cv_df['privacy'].isin(privacy_list)) &
                 (cv_df['density'] == param_spec['density']) &
                 (cv_df['n_estimators'] <= param_spec['max_rounds']) &
                 (cv_df['learning_rate'] == param_spec['learning_rate'])]

def plot_gs_df(cv_df, grid_param_x, grid_param_color,
               name_x, name_color, baseline, error_bars=True, title=None):

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
    if title is None:
        title = "Grid Search Scores"
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.set_xlabel(name_color, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.fill_between(x_pts, baseline['cv_score_avg'] + baseline['cv_score_std'],
                    baseline['cv_score_avg'] - baseline['cv_score_std'],
                    alpha=0.2)

def budget_curves(epsilon, n_rounds):
    fig, ax = plt.subplots()

    rounds = list(range(1, n_rounds))
    
    epsilon_simp = [simple_compose_budget(3000, n_rounds=T, epsilon=epsilon,
                                       density=0.1) for T in rounds]

    epsilon_adv = [drv_compose_budget(3000, 1, n_rounds=T, epsilon=epsilon,
                                      density=0.1) for T in rounds]
    
    plt.plot(rounds, epsilon_simp, label='Simple')

    plt.plot(rounds, epsilon_adv, label='Advanced')

    ax.set_xlabel('# Rounds')  # Add an x-label to the axes.
    ax.set_ylabel('Local Noise')  # Add a y-label to the axes.

    ax.set_title('Budgeting Rules')  # Add a title to the axes.

    ax.legend()  # Add a legend.
