import pandas as pd
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the pre-split data
print("Loading training data...")
df_train = pd.read_csv('train.csv')

print("Loading validation data...")
df_val = pd.read_csv('val.csv')

print("Loading test data...")
df_test = pd.read_csv('test.csv')

print(f"Training shape:   {df_train.shape}")
print(f"Validation shape: {df_val.shape}")
print(f"Test shape:       {df_test.shape}")
print(f"\nColumn names: {df_train.columns.tolist()}")
print(f"\nTraining target distribution:\n{df_train['ProdTaken'].value_counts()}")
print(f"\nValidation target distribution:\n{df_val['ProdTaken'].value_counts()}")
print(f"\nTest target distribution:\n{df_test['ProdTaken'].value_counts()}")

# Separate features and target
X_train = df_train.drop('ProdTaken', axis=1)
y_train = df_train['ProdTaken'].values

X_val = df_val.drop('ProdTaken', axis=1)
y_val = df_val['ProdTaken'].values

X_test = df_test.drop('ProdTaken', axis=1)
y_test = df_test['ProdTaken'].values

# Handle categorical variables using one-hot encoding
categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns: {categorical_columns}")

# One-hot encode categorical features
X_train_encoded = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)
X_val_encoded = pd.get_dummies(X_val, columns=categorical_columns, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)

# Align val and test columns with train columns (handle any missing dummy columns)
X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Save the column names for inference
feature_columns = X_train_encoded.columns.tolist()
joblib.dump(feature_columns, 'feature_columns.pkl')
print(f"Features after encoding: {X_train_encoded.shape[1]}")

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_val_scaled = scaler.transform(X_val_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Save the scaler for inference
joblib.dump(scaler, 'scaler.pkl')

print(f"\nTraining set size:   {X_train_scaled.shape[0]}")
print(f"Validation set size: {X_val_scaled.shape[0]}")
print(f"Testing set size:    {X_test_scaled.shape[0]}")

# Build an improved neural network
print("\nBuilding neural network model...")
model = keras.Sequential([
    # First hidden layer (wide)
    keras.layers.Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),

    # Second hidden layer
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    # Output layer
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

# Display model summary
model.summary()


def compile_model_with_custom_loss(model, loss_function, optimizer='adam', metrics=None):
    """
    Compile the model with a custom loss function.

    Parameters:
    -----------
    model : keras.Model
        The model to compile
    loss_function : str or callable
        Either a string for built-in loss functions (e.g., 'binary_crossentropy')
        or a custom loss function that takes (y_true, y_pred) as arguments
    optimizer : str or keras.optimizers.Optimizer
        Optimizer to use (default: 'adam')
    metrics : list
        List of metrics to track (default: ['accuracy', AUC])

    Returns:
    --------
    model : keras.Model
        The compiled model

    Example:
    --------
    # Using built-in loss
    compile_model_with_custom_loss(model, 'binary_crossentropy')

    # Using custom loss
    def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        bce_exp = tf.exp(-bce)
        focal_loss = alpha * tf.pow(1 - bce_exp, gamma) * bce
        return tf.reduce_mean(focal_loss)

    compile_model_with_custom_loss(model, focal_loss)
    """
    if metrics is None:
        metrics = ['accuracy', keras.metrics.AUC(name='auc')]

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics
    )

    return model


def weighted_binary_crossentropy(y_true, y_pred, pos_weight=2.0):
    """
    Weighted binary crossentropy loss for imbalanced datasets.
    Applies higher weight to positive class errors.
    """
    y_true = tf.cast(y_true, tf.float32)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    weights = y_true * (pos_weight - 1.0) + 1.0
    weighted_bce = bce * weights
    return tf.reduce_mean(weighted_bce)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal loss for handling class imbalance.
    Focuses on hard-to-classify examples.

    Parameters:
    -----------
    alpha : float
        Weighting factor (default: 0.25)
    gamma : float
        Focusing parameter (default: 2.0)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    # Calculate focal loss
    cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    weight = alpha * y_true * tf.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma)

    focal_loss_value = weight * cross_entropy
    return tf.reduce_mean(focal_loss_value)


def combined_bce_l1_weights_loss(model, alpha=1.0, beta=0.01):
    """
    Combined loss: alpha * binary_crossentropy + beta * L1_regularization on model weights

    This applies L1 regularization to the model's trainable weights instead of predictions.
    This is more commonly used for feature selection and model sparsity.

    Parameters:
    -----------
    model : keras.Model
        The model whose weights will be regularized
    alpha : float
        Weight for binary cross-entropy term (default: 1.0)
    beta : float
        Weight for L1 regularization on weights (default: 0.01)

    Returns:
    --------
    loss_function : callable
        A loss function that takes (y_true, y_pred) as arguments

    Example:
    --------
    custom_loss = combined_bce_l1_weights_loss(model, alpha=1.0, beta=0.001)
    compile_model_with_custom_loss(model, custom_loss)
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        # Binary cross-entropy component
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        bce_loss = tf.reduce_mean(bce)

        # L1 regularization on model weights
        l1_reg = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in model.trainable_weights])

        # Combined loss
        total_loss = alpha * bce_loss + beta * l1_reg

        return total_loss

    return loss

# compile_model_with_custom_loss(model, combined_bce_l1_weights_loss(model, alpha=1.0, beta=0.001))

# Compile with default loss (or use custom loss by calling the function above)
# To use custom loss, uncomment one of these:
# compile_model_with_custom_loss(model, weighted_binary_crossentropy)
# compile_model_with_custom_loss(model, focal_loss)
# compile_model_with_custom_loss(model, lambda y_true, y_pred: focal_loss(y_true, y_pred, alpha=0.3, gamma=2.5))

# Combined BCE + L1 loss examples:
# compile_model_with_custom_loss(model, combined_bce_l1_loss(alpha=1.0, beta=0.01))

# Train the model
print("\nTraining the model...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=500,
    batch_size=64,
    validation_data=(X_val_scaled, y_val),
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
    ]
)

# Evaluate the model on test set
print("\nEvaluating the model...")
test_loss, test_accuracy, test_auc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Make predictions
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Display classification report
print("\nClassification Report:")
target_names = ['Not Taken', 'Taken']
print(classification_report(y_test, y_pred, target_names=target_names))

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model
model.save('travel_product_model.h5')
print("\nModel saved to travel_product_model.h5")