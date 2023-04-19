import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plotRegressionGraph(model) :
    plt.style.use('seaborn')
    residuals = model.resid
    norm_residuals = model.get_influence().resid_studentized_internal
    norm_residuals_abs_sqrt = np.sqrt(np.abs(norm_residuals))
    fitted_value = model.fittedvalues
    leverage = model.get_influence().hat_matrix_diag
    cooks = model.get_influence().cooks_distance[0]
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    sns.residplot(x=fitted_value, y=residuals, ax=ax[0, 0], lowess=True, line_kws={'color': 'red', 'lw': 1, 'alpha': 1})
    ax[0, 0].scatter(x=fitted_value, y=residuals)
    ax[0, 0].set_xlabel('Fitted Values')
    ax[0, 0].set_ylabel('Residuals')
    ax[0, 0].set_title('Residuals vs Fitted Values')

    sm.qqplot(residuals, fit=True, line='45',ax=ax[0, 1])
    ax[0, 1].set_title('Normal Q-Q')
    
    sns.regplot(x=fitted_value, y=norm_residuals_abs_sqrt, ax=ax[1, 0], lowess=True, line_kws={'color': 'red', 'lw': 1, 'alpha': 1});
    ax[1, 0].set_xlabel('Fitted values')
    ax[1, 0].set_ylabel('$\sqrt{|Standardized Residuals|}$')
    ax[1, 0].set_title('Scale-Location Plot')

    sns.regplot(x=leverage, y=norm_residuals, ax=ax[1, 1], lowess=True,line_kws={'color': 'red', 'lw': 1, 'alpha': 1});
    ax[1, 1].set_xlabel('Leverage')
    ax[1, 1].set_ylabel('Standardized Residuals')
    ax[1, 1].set_title('Residuals vs Leverage Plot')

    leverage_top_3 = np.flip(np.argsort(cooks), 0)[:3]
    for i in leverage_top_3:
        ax[1, 1].annotate(i,xy=(leverage[i],norm_residuals[i]));

    plt.tight_layout()
    plt.show()
