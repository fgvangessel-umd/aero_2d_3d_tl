import pandas as pd
from matplotlib import pyplot as plt

# Load the saved dataframe from the CSV file
force_summary_df_tl = pd.read_csv('force_metrics_summary.csv')
force_summary_df_no_tl = pd.read_csv('../../aero_3d_NO/force_metrics_summary.csv')

num_cases = [128, 256, 384, 512, 641]
 # Create a second visualization for MAE metrics
plt.figure(figsize=(16, 20))

# Plot MAE for lift
plt.subplot(2, 1, 1)
plt.plot(num_cases, force_summary_df_no_tl['test_mae_lift_mean'], 'o-', color='blue', label='No TL')
plt.fill_between(num_cases, 
    force_summary_df_no_tl['test_mae_lift_mean'] - force_summary_df_no_tl['test_mae_lift_std'], 
    force_summary_df_no_tl['test_mae_lift_mean'] + force_summary_df_no_tl['test_mae_lift_std'], 
    alpha=0.2, 
    color='blue')

plt.plot(num_cases, force_summary_df_tl['test_mae_lift_mean'], 'o-', color='red', label='TL')
plt.fill_between(num_cases, 
    force_summary_df_tl['test_mae_lift_mean'] - force_summary_df_tl['test_mae_lift_std'], 
    force_summary_df_tl['test_mae_lift_mean'] + force_summary_df_tl['test_mae_lift_std'], 
    alpha=0.2, 
    color='red')

#plt.plot(num_cases, cv_df_tl['test_loss_mean'], 'o-', color='red', label='TL')
#plt.fill_between(num_cases, cv_df_tl['test_loss_mean'] - cv_df_tl['test_loss_std'], cv_df_tl['test_loss_mean'] + cv_df_tl['test_loss_std'], alpha=0.2, color='red')

plt.xlabel('Training Data Percentage', fontsize=24)
plt.ylabel('Mean Absolute Error', fontsize=24)
plt.title('Lift Prediction MAE', fontsize=28)
# Increase tick label size for all axes
plt.tick_params(axis='both', which='major', labelsize=24)
plt.grid(True)

# Plot MAE for drag
plt.subplot(2, 1, 2)
plt.plot(num_cases, force_summary_df_no_tl['test_mae_drag_mean'], 'o-', color='blue', label='No TL')
plt.fill_between(num_cases, 
    force_summary_df_no_tl['test_mae_drag_mean'] - force_summary_df_no_tl['test_mae_drag_std'], 
    force_summary_df_no_tl['test_mae_drag_mean'] + force_summary_df_no_tl['test_mae_drag_std'], 
    alpha=0.2, 
    color='blue')

plt.plot(num_cases, force_summary_df_tl['test_mae_drag_mean'], 'o-', color='red', label='TL')
plt.fill_between(num_cases, 
    force_summary_df_tl['test_mae_drag_mean'] - force_summary_df_tl['test_mae_drag_std'], 
    force_summary_df_tl['test_mae_drag_mean'] + force_summary_df_tl['test_mae_drag_std'], 
    alpha=0.2, 
    color='red')

plt.xlabel('Training Data Percentage', fontsize=24)
plt.ylabel('Mean Absolute Error', fontsize=24)
plt.title('Drag Prediction MAE', fontsize=28)
plt.legend(fontsize=28)
plt.grid(True)
# Increase tick label size for all axes
plt.tick_params(axis='both', which='major', labelsize=24)

plt.tight_layout()
plt.savefig('lift_drag_prediction.png')

###
### Pure loss
###

# Load the saved dataframe from the CSV file
cv_df_tl = pd.read_csv('cross_validation_results.csv')
cv_df_no_tl = pd.read_csv('../../aero_3d_NO/cross_validation_results.csv')

'''
# Create visualizations
# First figure: Test metrics vs training percentage
plt.figure(figsize=(12, 12))

# Plot test RMSE vs training percentage
plt.subplot(3, 1, 1)
plt.errorbar(
    cv_df_tl['train_percent'], 
    cv_df_tl['test_rmse_mean'], 
    yerr=cv_df_tl['test_rmse_std'],
    fmt='o-', 
    capsize=5,
    color='red',
    label='TL'
)
plt.errorbar(
    cv_df_no_tl['train_percent'], 
    cv_df_no_tl['test_rmse_mean'], 
    yerr=cv_df_no_tl['test_rmse_std'],
    fmt='o-', 
    capsize=5,
    color='blue',
    label='No TL'
)
plt.xlabel('Training Data Percentage')
plt.ylabel('Test RMSE')
plt.title('Test RMSE vs Training Data Percentage')
plt.grid(True)

#print((cv_df_tl['test_rmse_mean']-cv_df_no_tl['test_rmse_mean'])/cv_df_tl['test_rmse_mean']*100)
print((cv_df_tl['test_rmse_std']-cv_df_no_tl['test_rmse_std'])/cv_df_tl['test_rmse_std']*100)

# Plot test MAE vs training percentage
plt.subplot(3, 1, 2)
plt.errorbar(
    cv_df_tl['train_percent'], 
    cv_df_tl['test_mae_mean'], 
    yerr=cv_df_tl['test_mae_std'],
    fmt='o-', 
    capsize=5,
    color='red',
    label='TL'
)
plt.errorbar(
    cv_df_no_tl['train_percent'], 
    cv_df_no_tl['test_mae_mean'], 
    yerr=cv_df_no_tl['test_mae_std'],
    fmt='o-', 
    capsize=5,
    color='blue',
    label='No TL'
)
plt.xlabel('Training Data Percentage')
plt.ylabel('Test MAE')
plt.title('Test MAE vs Training Data Percentage')
plt.legend()
plt.grid(True)

#print((cv_df_tl['test_mae_mean']-cv_df_no_tl['test_mae_mean'])/cv_df_tl['test_mae_mean']*100)
print((cv_df_tl['test_mae_std']-cv_df_no_tl['test_mae_std'])/cv_df_tl['test_mae_std']*100)

# Plot test Loss vs training percentage
plt.subplot(3, 1, 3)
plt.errorbar(
    cv_df_tl['train_percent'], 
    cv_df_tl['test_loss_mean'], 
    yerr=cv_df_tl['test_loss_std'],
    fmt='o-', 
    capsize=5,
    color='red',
    label='TL'
)
plt.errorbar(
    cv_df_no_tl['train_percent'], 
    cv_df_no_tl['test_loss_mean'], 
    yerr=cv_df_no_tl['test_loss_std'],
    fmt='o-', 
    capsize=5,
    color='blue',
    label='No TL'
)
plt.xlabel('Training Data Percentage')
plt.ylabel('Test Loss')
plt.title('Test Loss vs Training Data Percentage')
plt.legend()
plt.grid(True)

#print((cv_df_tl['test_loss_mean']-cv_df_no_tl['test_loss_mean'])/cv_df_tl['test_loss_mean']*100)
print((cv_df_tl['test_loss_std']-cv_df_no_tl['test_loss_std'])/cv_df_tl['test_loss_std']*100)

plt.tight_layout()
plt.savefig('cv_results.png')
'''
###
###
###

# Create visualizations
# First figure: Test metrics vs training percentage
plt.figure(figsize=(16, 10))

# Plot test Loss vs training percentage
plt.subplot(1, 1, 1)
'''
plt.errorbar(
    cv_df_tl['train_percent'], 
    cv_df_tl['test_loss_mean'], 
    yerr=cv_df_tl['test_loss_std'],
    fmt='o-', 
    capsize=5,
    color='red',
    label='TL'
)
plt.errorbar(
    cv_df_no_tl['train_percent'], 
    cv_df_no_tl['test_loss_mean'], 
    yerr=cv_df_no_tl['test_loss_std'],
    fmt='o-', 
    capsize=5,
    color='blue',
    label='No TL'
)
'''
plt.plot(num_cases, cv_df_no_tl['test_loss_mean'], 'o-', color='blue', label='No TL')
plt.fill_between(num_cases, cv_df_no_tl['test_loss_mean'] - cv_df_no_tl['test_loss_std'], cv_df_no_tl['test_loss_mean'] + cv_df_no_tl['test_loss_std'], alpha=0.2, color='blue')

plt.plot(num_cases, cv_df_tl['test_loss_mean'], 'o-', color='red', label='TL')
plt.fill_between(num_cases, cv_df_tl['test_loss_mean'] - cv_df_tl['test_loss_std'], cv_df_tl['test_loss_mean'] + cv_df_tl['test_loss_std'], alpha=0.2, color='red')


plt.xlabel('Number of Training Cases', fontsize=24)
plt.ylabel('Test Loss', fontsize=24)
plt.title('Test Loss vs Training Data Percentage', fontsize=28)
plt.legend(fontsize=28)
plt.grid(True)

# Increase tick label size for all axes
plt.tick_params(axis='both', which='major', labelsize=24)

#print((cv_df_tl['test_loss_mean']-cv_df_no_tl['test_loss_mean'])/cv_df_tl['test_loss_mean']*100)
print((cv_df_tl['test_loss_std']-cv_df_no_tl['test_loss_std'])/cv_df_tl['test_loss_std']*100)

plt.tight_layout()
plt.savefig('cv_results_loss.png')