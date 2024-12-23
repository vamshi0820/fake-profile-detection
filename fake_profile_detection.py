<<<<<<< HEAD
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Step 1: Load Dataset
def load_data():
    # Create a sample dataset (replace with a real one if available)
    data = {
        'profile_feature1': np.random.rand(1000),
        'profile_feature2': np.random.rand(1000),
        'profile_feature3': np.random.rand(1000),
        'is_fake': np.random.choice([0, 1], size=1000)
    }
    df = pd.DataFrame(data)
    return df

# Step 2: Preprocess Data
def preprocess_data(df):
    X = df.drop('is_fake', axis=1)
    y = df['is_fake']

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Step 3: Build Model
def build_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 4: Train Model
def train_model(model, X_train, y_train):
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    return history

# Step 5: Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype('int32')
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Main Execution
if __name__ == "__main__":
    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Build and train model
    model = build_model(input_dim=X_train.shape[1])
    train_model(model, X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Save model
    model.save("fake_profile_model.h5")
    print("Model saved as fake_profile_model.h5")
=======
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Step 1: Load Dataset
def load_data():
    # Create a sample dataset (replace with a real one if available)
    data = {
        'profile_feature1': np.random.rand(1000),
        'profile_feature2': np.random.rand(1000),
        'profile_feature3': np.random.rand(1000),
        'is_fake': np.random.choice([0, 1], size=1000)
    }
    df = pd.DataFrame(data)
    return df

# Step 2: Preprocess Data
def preprocess_data(df):
    X = df.drop('is_fake', axis=1)
    y = df['is_fake']

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Step 3: Build Model
def build_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 4: Train Model
def train_model(model, X_train, y_train):
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    return history

# Step 5: Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype('int32')
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Main Execution
if __name__ == "__main__":
    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Build and train model
    model = build_model(input_dim=X_train.shape[1])
    train_model(model, X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Save model
    model.save("fake_profile_model.h5")
    print("Model saved as fake_profile_model.h5")
>>>>>>> 5fceb5685b08092d7cda4f0f108c188e5c7b2bb0
