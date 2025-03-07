# Imports
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from pathlib import Path

# Read the csv into a Pandas DataFrame
file_path = "https://static.bc-edx.com/ai/ail-v-1-0/m18/lms/datasets/student-loans.csv"
loans_df = pd.read_csv(file_path)

# Define the target set y using the credit_ranking column
y = loans_df["credit_ranking"]

# Define features set X by selecting all columns but credit_ranking
X = loans_df.drop(columns=["credit_ranking"])

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler to the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test_scaled = scaler.transform(X_test)

# Define the number of input features
input_features = X_train_scaled.shape[1]

# Define the deep neural network model
model = Sequential([
    Dense(units=16, activation="relu", input_dim=input_features),  # First hidden layer with 16 neurons
    Dense(units=8, activation="relu"),  # Second hidden layer with 8 neurons
    Dense(units=1, activation="sigmoid")  # Output layer with 1 neuron for binary classification
])

# Compile the Sequential model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model using 50 epochs and the training data
model.fit(X_train_scaled, y_train, epochs=50, batch_size=10, verbose=1)

# Evaluate the model loss and accuracy metrics using the test data
eval_loss, eval_accuracy = model.evaluate(X_test_scaled, y_test, verbose=1)

# Display the model loss and accuracy results
print(f"Test Loss: {eval_loss:.4f}, Test Accuracy: {eval_accuracy:.4f}")

# Save and export the trained model
model.save("student_loans.keras")
print("Model saved as student_loans.keras")

# Reload the saved model
loaded_model = load_model("student_loans.keras")
print("Model reloaded successfully.")

# Make predictions on the testing data
predictions = loaded_model.predict(X_test_scaled)

# Convert predictions to binary values (0 or 1)
predictions_binary = (predictions > 0.5).astype(int)

# Save predictions to a DataFrame
predictions_df = pd.DataFrame({"Actual": y_test.values, "Predicted": predictions_binary.flatten()})

# Display the first few predictions
print(predictions_df.head())

# Generate and display a classification report
report = classification_report(y_test, predictions_binary)
print("Classification Report:")
print(report)

# Part 4: Recommendation System Discussion
recommendation_discussion = """
To build a recommendation system for student loans, the following data would be required:

1. **Credit Score & Payment History** - Indicates financial responsibility and risk level.
2. **Annual Income & Employment Status** - Helps assess loan repayment capability.
3. **Loan Amount & Interest Rate** - Ensures loans recommended are within a reasonable range.
4. **Education Level & Major** - Different degrees may have different earning potential, impacting repayment ability.
5. **Financial Aid & Scholarships Received** - Determines if a student truly requires additional loans.
6. **Loan Term Preferences** - Some students may prefer lower monthly payments over shorter loan terms.
7. **Cost of Living & Location** - Living expenses vary by region, affecting the recommended loan amount.

This data is relevant as it allows the recommendation system to match students with loan options that best fit their financial profile, minimizing risk while optimizing repayment success.

### Filtering Method Selection
Based on the data chosen, the recommendation system would use **content-based filtering**. This approach analyzes student attributes (such as credit history, income, education level, and loan preferences) to match them with loans that have historically been beneficial to individuals with similar profiles. 

**Justification:**
- **Content-based filtering** is ideal since recommendations are based on individual financial and academic characteristics rather than collective user preferences.
- Unlike **collaborative filtering**, which relies on user behavior similarities, content-based filtering ensures tailored loan options based on a student’s unique financial situation.
- **Context-based filtering** (which adjusts recommendations dynamically) could be an enhancement, but the foundational recommendation model would focus on feature-matching.

### Real-World Challenges in Building a Loan Recommendation System
1. **Data Privacy and Security** - Since financial and personal data is highly sensitive, ensuring compliance with regulations like GDPR and CCPA is critical. Encryption, secure data storage, and limited access control must be implemented to protect user information.
2. **Bias in Data and Algorithmic Fairness** - Loan recommendation models might inherit biases from historical lending patterns, leading to unfair loan suggestions for certain demographics. Addressing bias through diverse training data and fairness-aware algorithms is essential to avoid discrimination in loan approvals.
"""

print(recommendation_discussion)
