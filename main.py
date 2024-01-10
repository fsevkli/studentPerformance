from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

# Fetch dataset 
student_performance = fetch_ucirepo(id=320)

# Data (as pandas dataframes)
X = student_performance.data.features 
y = student_performance.data.targets 

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select categorical columns (you may need to adjust this based on your dataset)
categorical_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols)
    ])

# Append linear regression model to preprocessing pipeline
lr = Pipeline(steps=[('preprocessor', preprocessor),
                     ('regressor', LinearRegression())])

# Fit the model
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE

# Assuming 'lr' is your trained model and 'categorical_cols' contains the names of categorical columns
categorical_feature_names = lr.named_steps['preprocessor'].named_transformers_['cat']\
                                .named_steps['onehot'].get_feature_names_out(input_features=categorical_cols)

# Get coefficients after the model is trained
lr_model = lr.named_steps['regressor']
coefficients = lr_model.coef_[0]  # Extract coefficients for the first target (if using multi-output regression)

# Create a DataFrame to display coefficients with their corresponding feature names
coef_df = pd.DataFrame({'Feature': categorical_feature_names, 'Coefficient': coefficients})

# Display coefficients
print(coef_df)

data = {
    'school': ['MS'],
    'sex': ['M'],
    'age': [18],
    'address': ['U'],
    'famsize': ['LE3'],
    'Pstatus': ['T'],
    'Medu': [2],
    'Fedu': [3],
    'Mjob': ['services'],
    'Fjob': ['teacher'],
    'reason': ['home'],
    'guardian': ['mother'],
    'traveltime': [2],
    'studytime': [4],
    'failures': [1],
    'schoolsup': ['yes'],
    'famsup': ['yes'],
    'paid': ['no'],
    'activities': ['no'],
    'nursery': ['yes'],
    'higher': ['yes'],
    'internet': ['yes'],
    'romantic': ['yes'],
    'famrel': [3],
    'freetime': [3],
    'goout': [5],
    'Dalc': [3],
    'Walc': [4],
    'health': [2],
    'absences': [8],
    'G1': [12],
    'G2': [11],
    'G3': [10]
}

# Create a DataFrame
new_data = pd.DataFrame(data)

# Predict the final grade
predicted_grade = lr.predict(new_data)
print(predicted_grade)
