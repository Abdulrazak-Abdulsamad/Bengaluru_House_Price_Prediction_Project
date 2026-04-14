# Bengaluru_House_Price_Prediction_Project


This notebook outlines a machine learning project focused on predicting house prices in Bengaluru, India. The process involves several key stages:

1.  **Data Loading and Initial Exploration**: The project begins by loading house price data from a CSV file into a Pandas DataFrame and performing initial checks on its structure and content.
2.  **Data Cleaning**: Irrelevant columns are dropped, and missing values are handled by removing corresponding rows. The 'size' column is converted into a numerical 'bhk' (Bedroom, Hall, Kitchen) feature.
3.  **Feature Engineering**: The `total_sqft` column, which contains both single values and ranges, is standardized into a numerical format. A new feature, `price_per_sqft`, is calculated to aid in outlier detection.
4.  **Dimensionality Reduction for Location**: Locations with fewer than 10 data points are grouped into an 'other' category to manage the high cardinality of the `location` feature.
5.  **Outlier Removal**: Outliers are identified and removed based on `price_per_sqft` (per location, using standard deviation) and by checking for unrealistic `bhk` vs. `total_sqft` ratios and `bath` vs. `bhk` ratios.
6.  **One-Hot Encoding**: Categorical `location` data is converted into numerical format using one-hot encoding.
7.  **Model Training**: The cleaned and prepared data is used to train a `LinearRegression` model. `ShuffleSplit` and `cross_val_score` are used for cross-validation, and `GridSearchCV` is employed to evaluate and find the best parameters for `LinearRegression`, `Lasso`, and `DecisionTreeRegressor` models.
8.  **Price Prediction Function**: A function is created to predict house prices based on location, square footage, number of bathrooms, and BHK count.
9.  **Model Export**: The trained `LinearRegression` model and the column names are saved using `pickle` and `json` respectively, for future deployment.
