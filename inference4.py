import pandas as pd
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import os

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Define custom loss functions (must match the ones used in training)
def combined_bce_l1_weights_loss(model, alpha=1.0, beta=0.01):
    """
    Combined loss: alpha * binary_crossentropy + beta * L1_regularization on model weights
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        bce_loss = tf.reduce_mean(bce)
        l1_reg = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in model.trainable_weights])
        total_loss = alpha * bce_loss + beta * l1_reg
        return total_loss
    return loss


# Load the trained model with custom loss function
print("Loading trained model...")
# Load the model without compiling first
model = keras.models.load_model('travel_product_model.h5', compile=False)

# Recompile with the loss function used during training (standard BCE)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

print("Model loaded successfully!")

# Display model summary
model.summary()

# Load the saved scaler and feature columns from training
print("\nLoading scaler and feature columns...")
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')
print(f"Feature columns loaded: {len(feature_columns)} features")

# Load the test data
print("\nLoading test data...")
df_test = pd.read_csv('test.csv')

print(f"Test dataset shape: {df_test.shape}")
print(f"\nTarget distribution:\n{df_test['ProdTaken'].value_counts()}")

# Separate features and target
X_test = df_test.drop('ProdTaken', axis=1)
y_test = df_test['ProdTaken'].values

# Handle categorical variables using one-hot encoding
# Must match the training data preprocessing
categorical_columns = X_test.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns: {categorical_columns}")

# One-hot encode categorical features
X_test_encoded = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)

# Align columns with training data (handle any missing dummy columns)
X_test_encoded = X_test_encoded.reindex(columns=feature_columns, fill_value=0)

print(f"Features after encoding: {X_test_encoded.shape[1]}")

# Standardize the features using the saved scaler from training
X_test_scaled = scaler.transform(X_test_encoded)

# Make predictions
print("\nMaking predictions...")
y_pred_proba = model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Calculate metrics
print("\n" + "="*60)
print("INFERENCE RESULTS")
print("="*60)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

try:
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC Score: {auc_score:.4f}")
except:
    print("AUC Score: Could not calculate")

# Classification report
target_names = ['Not Taken', 'Taken']
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Travel Product Classification', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

# Add additional statistics as text
tn, fp, fn, tp = cm.ravel()
stats_text = f'True Negatives: {tn}\nFalse Positives: {fp}\nFalse Negatives: {fn}\nTrue Positives: {tp}'
stats_text += f'\n\nAccuracy: {accuracy:.4f}'
plt.text(2.5, 0.5, stats_text, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         verticalalignment='center')

plt.tight_layout()

# Save the figure
output_path = 'output/confusion_matrix.jpg'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nConfusion matrix plot saved to: {output_path}")

# Also create a normalized confusion matrix
plt.figure(figsize=(10, 8))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=target_names,
            yticklabels=target_names,
            cbar_kws={'label': 'Percentage'})
plt.title('Normalized Confusion Matrix - Travel Product Classification', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
output_path_normalized = 'output/confusion_matrix_normalized.jpg'
plt.savefig(output_path_normalized, dpi=300, bbox_inches='tight')
print(f"Normalized confusion matrix plot saved to: {output_path_normalized}")

# Save predictions to CSV
results_df = df_test.copy()
results_df['predicted_label'] = ['Taken' if p == 1 else 'Not Taken' for p in y_pred]
results_df['prediction_probability'] = y_pred_proba.flatten()
results_df['true_label'] = ['Taken' if t == 1 else 'Not Taken' for t in y_test]
results_df['correct_prediction'] = (y_test == y_pred)

output_csv_path = 'output/predictions.csv'
results_df.to_csv(output_csv_path, index=False)
print(f"Predictions saved to: {output_csv_path}")

print("\n" + "="*60)
print("Inference completed successfully!")
print("="*60)