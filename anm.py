import pandas as pd  # Importing the pandas library for data manipulation and analysis
import numpy as np  # Importing the numpy library for numerical operations and array manipulation
from sklearn.model_selection import KFold  # Importing KFold for cross-validation splitting
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder  # Importing preprocessing tools: StandardScaler for scaling features, OneHotEncoder for one-hot encoding categorical variables, and LabelEncoder for encoding labels
from sklearn.compose import ColumnTransformer  # Importing ColumnTransformer to apply different preprocessing steps to different columns
from tensorflow.keras import layers, models, optimizers, regularizers  # Importing Keras modules for building and training neural networks
from tensorflow.keras.callbacks import EarlyStopping  # Importing EarlyStopping to stop training early if the validation loss stops improving
from sklearn.metrics import classification_report, confusion_matrix  # Importing metrics for evaluating the performance of the model
from scipy.sparse import csr_matrix  # Importing csr_matrix to handle sparse matrices (used in feature transformation)
import joblib  # Importing joblib for saving and loading the preprocessor and model objects

# Load the datasets
train_data = pd.read_csv('UNSW_NB15_training-set.csv')  # Load training dataset
test_data = pd.read_csv('UNSW_NB15_testing-set.csv')     # Load testing dataset

# Drop unnecessary columns (id) and encode attack categories
X_train = train_data.drop(columns=['id', 'label'])  # Drop 'id' and 'label' columns from training data
y_train = train_data['attack_cat']                  # Extract the 'attack_cat' column as labels for training

X_test = test_data.drop(columns=['id', 'label'])    # Drop 'id' and 'label' columns from testing data
y_test = test_data['attack_cat']                    # Extract the 'attack_cat' column as labels for testing

# Encode attack categories as integers
label_encoder = LabelEncoder()          # Initialize LabelEncoder to convert categorical labels into integers
y_train = label_encoder.fit_transform(y_train)  # Fit and transform the labels for the training data
y_test = label_encoder.transform(y_test)        # Transform the labels for the testing data using the same encoder

# One-hot encode the labels for multi-class classification
y_train = np.eye(len(label_encoder.classes_))[y_train]  # Convert integer labels to one-hot encoded format for training data
y_test = np.eye(len(label_encoder.classes_))[y_test]    # Convert integer labels to one-hot encoded format for testing data

# Identify categorical and numerical features
categorical_features = ['proto', 'service', 'state']  # Specify categorical features that need encoding

# Ensure that no numeric feature has non-numeric data
numeric_features = []  # Initialize a list to hold numeric features
for col in X_train.columns:
    if col not in categorical_features:  # Check if the column is not categorical
        if X_train[col].dtype == 'object':  # If the column has non-numeric data
            categorical_features.append(col)  # Add it to the list of categorical features
        else:
            numeric_features.append(col)  # Otherwise, add it to the list of numeric features

print("Categorical Features:", categorical_features)
print("Numeric Features:", numeric_features)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),  # Standardize numeric features
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)  # One-hot encode categorical features
    ])

# Apply preprocessing
X_train_preprocessed = preprocessor.fit_transform(X_train)  # Fit and transform the training data
X_test_preprocessed = preprocessor.transform(X_test)        # Transform the testing data using the same preprocessor

# Convert the sparse matrix to a dense matrix if not done already
if isinstance(X_train_preprocessed, csr_matrix):
    X_train_preprocessed = X_train_preprocessed.toarray()  # Convert training data to a dense matrix if it's sparse

if isinstance(X_test_preprocessed, csr_matrix):
    X_test_preprocessed = X_test_preprocessed.toarray()    # Convert testing data to a dense matrix if it's sparse

# Verify preprocessing
print(f"Processed training data shape: {X_train_preprocessed.shape}")
print(f"Processed testing data shape: {X_test_preprocessed.shape}")

# Save the preprocessor
joblib.dump(preprocessor, 'preprocessor_anm.joblib')  # Save the preprocessor object to a file for later use
print('Preprocessor saved as preprocessor_anm.joblib')

# Define a function to create the model
def create_model():
    model = models.Sequential()  # Initialize a Sequential model
    model.add(layers.Dense(128, activation='relu', input_shape=(X_train_preprocessed.shape[1],),
                           kernel_regularizer=regularizers.l2(0.001)))  # Add a dense layer with L2 regularization
    model.add(layers.BatchNormalization())  # Add batch normalization to stabilize and speed up training
    model.add(layers.Dropout(0.3))  # Add dropout to prevent overfitting
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))  # Add another dense layer with L2 regularization
    model.add(layers.BatchNormalization())  # Add batch normalization
    model.add(layers.Dropout(0.3))  # Add dropout
    model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))  # Add a third dense layer
    model.add(layers.Dense(len(label_encoder.classes_), activation='softmax'))  # Output layer with softmax for multi-class classification

    optimizer = optimizers.Adam(learning_rate=0.0005)  # Initialize Adam optimizer with a learning rate of 0.0005
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])  # Compile the model with categorical crossentropy loss
    return model  # Return the compiled model

# Implement Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Initialize early stopping to prevent overfitting

# Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Set up 5-fold cross-validation with shuffling

cv_scores = []  # List to store cross-validation scores
fold = 1  # Counter for folds
best_val_accuracy = 0.0  # Track the best validation accuracy
best_model = None  # Placeholder for the best model

for train_index, val_index in kf.split(X_train_preprocessed):
    print(f"\nTraining fold {fold}...")
    fold += 1
    
    X_train_fold, X_val_fold = X_train_preprocessed[train_index], X_train_preprocessed[val_index]  # Split data into training and validation sets for this fold
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]  # Split labels into training and validation sets
    
    # Create a new model instance for each fold
    model = create_model()
    
    # Train the model
    history = model.fit(X_train_fold, y_train_fold,
                        epochs=50,  # Set the number of epochs
                        batch_size=128,  # Set the batch size
                        validation_data=(X_val_fold, y_val_fold),  # Set validation data for this fold
                        callbacks=[early_stopping],  # Use early stopping to avoid overfitting
                        verbose=1)  # Print progress during training
    
    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)  # Evaluate on the validation set
    cv_scores.append(val_accuracy)  # Store the validation accuracy
    print(f"Validation accuracy for this fold: {val_accuracy:.4f}")
    
    # Save the best model during cross-validation
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy  # Update the best validation accuracy
        best_model = model  # Keep the reference to the best model

# Save the best model during cross-validation
if best_model is not None:
    best_model.save('best_cross_validated_model_anm.keras')  # Save the best model found during cross-validation
    print(f"Best model saved with validation accuracy: {best_val_accuracy:.4f}")

# Report the average cross-validated accuracy
average_cv_accuracy = np.mean(cv_scores)  # Calculate the mean accuracy across all folds
std_cv_accuracy = np.std(cv_scores)  # Calculate the standard deviation of the accuracy
print(f"\nAverage cross-validated accuracy: {average_cv_accuracy:.4f} ± {std_cv_accuracy:.4f}")

# Final Evaluation on the Test Set
# Train the model on the full training data before evaluating on the test set
final_model = create_model()  # Create a new model instance
final_model.fit(X_train_preprocessed, y_train, epochs=50, batch_size=128, validation_split=0.2, callbacks=[early_stopping], verbose=1)  # Train on the full training set

# Save the final trained model
final_model.save('anm.keras')  # Save the final model after training on the full dataset
print("Final model saved as 'anm.keras'")

# Predict the labels on the test set
y_pred = np.argmax(final_model.predict(X_test_preprocessed), axis=1)  # Predict and convert probabilities to class labels

# Convert one-hot encoded true labels back to original labels
y_test_decoded = np.argmax(y_test, axis=1)  # Convert one-hot encoded test labels back to integer format

# Evaluate the performance
conf_matrix = confusion_matrix(y_test_decoded, y_pred)  # Generate confusion matrix to evaluate performance
class_report = classification_report(y_test_decoded, y_pred, target_names=label_encoder.classes_)  # Generate classification report with precision, recall, f1-score

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)

# Save the results to a text file
with open('Model Evaluation.txt', 'w') as f:  # Open a text file to save results
    f.write("Confusion Matrix:\n")  # Write confusion matrix
    f.write(np.array2string(conf_matrix))  # Convert and write confusion matrix to the file
    f.write("\n\nClassification Report:\n")  # Write classification report heading
    f.write(class_report)  # Write classification report
    f.write("\n\nAverage Cross-Validated Accuracy: {:.4f} ± {:.4f}".format(average_cv_accuracy, std_cv_accuracy))  # Write cross-validation results

