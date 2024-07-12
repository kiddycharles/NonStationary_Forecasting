from sklearn.preprocessing import MinMaxScaler, RobustScaler
import numpy as np

# Sample training data
train_data = np.array([[1, 100], [2, 200], [3, 300], [4, 400]])

# Sample test data with out-of-range values
test_data = np.array([[0, 50], [5, 500]])

# Initialize the scaler
scaler = MinMaxScaler()

# Fit the scaler on the training data
scaler.fit(train_data)

# Transform the training data
scaled_train_data = scaler.transform(train_data)
print("Scaled training data:")
print(scaled_train_data)

# Transform the test data
scaled_test_data = scaler.transform(test_data)
print("Scaled test data:")
print(scaled_test_data)

robust_scaler = RobustScaler()

# Fit the scaler on the training data
robust_scaler.fit(train_data)

# Transform the training data
scaled_train_data = robust_scaler.transform(train_data)
print("Scaled training data:")
print(scaled_train_data)

# Transform the test data
scaled_test_data = robust_scaler.transform(test_data)
print("Scaled test data:")
print(scaled_test_data)

