import numpy as np
from scipy import stats

ann = np.array([0.3147, 0.2803, 0.2169, 0.1919, 0.2802, 0.2021, 0.4909, 0.1837, 0.2522, 0.3467])
linear_regression = np.array([0.5060, 0.5764, 0.4301, 0.5004, 0.3432, 0.5817, 0.4436, 0.6806, 0.6825, 0.6750])
baseline = np.array([1.3028, 1.0992, 0.8503, 0.9732, 0.9974, 1.2642, 1.0754, 0.9835, 0.6456, 0.8160])

def paired_ttest_with_ci(x, y, alpha=0.05):
    diff = x - y
    t_stat, p_val = stats.ttest_rel(x, y)
    mean_diff = np.mean(diff)
    sem_diff = stats.sem(diff)
    df = len(diff) - 1
    ci = stats.t.interval(1-alpha, df, loc=mean_diff, scale=sem_diff)  # 95% CI by default
    return t_stat, p_val, mean_diff, ci

t_lr_ann, p_lr_ann, mean_diff_lr_ann, ci_lr_ann = paired_ttest_with_ci(linear_regression, ann)
t_base_ann, p_base_ann, mean_diff_base_ann, ci_base_ann = paired_ttest_with_ci(baseline, ann)
t_base_lr, p_base_lr, mean_diff_base_lr, ci_base_lr = paired_ttest_with_ci(baseline, linear_regression)

print("Paired t-test results with 95% confidence intervals :")
print(f"Linear Regression vs ANN: t = {t_lr_ann:.4f}, p = {p_lr_ann:.4f}, 95% CI = ({ci_lr_ann[0]:.4f}, {ci_lr_ann[1]:.4f})")
print(f"Baseline vs ANN: t = {t_base_ann:.4f}, p = {p_base_ann:.4f}, 95% CI = ({ci_base_ann[0]:.4f}, {ci_base_ann[1]:.4f})")
print(f"Baseline vs Linear Regression: t = {t_base_lr:.4f}, p = {p_base_lr:.4f}, 95% CI = ({ci_base_lr[0]:.4f}, {ci_base_lr[1]:.4f})")