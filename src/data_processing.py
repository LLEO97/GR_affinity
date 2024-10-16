# Install necessary libraries

# Data Cleaning and Feature Selection
data = df.iloc[:, 4:]
# Replace strings with NaN values
data = data.applymap(lambda x: np.nan if isinstance(x, str) else x)
# Remove columns with more than 90% missing values
threshold = 0.9 * len(data)
data.dropna(axis=1, thresh=threshold, inplace=True)
names = names[data.index]  # Aligning names after removing columns with many NaNs

# Data Cleaning and Deduplication
data.drop_duplicates(inplace=True)
names = names[data.index]  # Aligning names after removing duplicates
data = data.loc[:, data.apply(pd.Series.nunique) != 1]  # Remove features with zero variance
corr_matrix = data.corr().abs()  # Calculate Pearson correlation between features
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]  # Remove features with high pairwise correlation
data.drop(to_drop, axis=1, inplace=True)

# Remove rows containing NaN values
data.dropna(inplace=True)
names = names[data.index]  # Aligning names after removing rows with NaNs

# Reset index
data.reset_index(drop=True, inplace=True)
names.reset_index(drop=True, inplace=True)  # Aligning names with reset index
