# Ensure the path exists
os.makedirs(save_path, exist_ok=True)

# Plot and save SHAP images for each class
for i in range(5):
    plt.figure(figsize=(10, 10))
    shap.summary_plot(shap_values[i], X_pi, plot_type="dot", max_display=10, show=False)
    plt.savefig(os.path.join(save_path, f'shap_class_{i}.png'), dpi=300)  # Save the image, set resolution to 300 dpi
    plt.clf()

# Define the path to save CSV files
csv_path = r'...'

# Ensure the path exists
os.makedirs(csv_path, exist_ok=True)

# For each class, calculate and save the SHAP values of features
for i in range(5):
    # Calculate the mean absolute SHAP value for each feature
    mean_abs_shap_values = np.mean(np.abs(shap_values[i]), axis=0)

    # Map features to their importance values
    feature_importances = dict(zip(X_pi.columns, mean_abs_shap_values))

    # Sort features by importance
    sorted_feature_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

    # Print sorted feature importances
    for feature, importance in sorted_feature_importances:
        print(f"Feature: {feature}, Importance: {importance}")

    # Create a DataFrame object
    df_importances = pd.DataFrame(sorted_feature_importances, columns=['Feature', 'Importance'])

    # Save the DataFrame as a CSV file
    df_importances.to_csv(os.path.join(csv_path, f'feature_importances_{i}.csv'), index=False)
......
# For the complete code, please contact yongchengl@unr.edu
