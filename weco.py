import os
import cv2
import numpy as np
import glob as gb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from skimage.feature import hog
from skimage.transform import resize
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
def load_images_and_labels(folder_path):
    data = []
    labels = []
    unique_labels = set()
    for subfolder in ["sushi", "cheesecake"]:
        subfolder_path = os.path.join(folder_path, subfolder)
        for filename in os.listdir(subfolder_path):
            if filename.endswith(".jpg"):
                img_path = os.path.join(subfolder_path, filename)
                # Load image
                img = cv2.imread(img_path)
                # Check if the image is loaded successfully
                if img is not None:
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Resize the image to a common size (e.g., 128×256)
                    img = cv2.resize(img, (128, 256))
                    data.append(img)
                    # Use the subfolder name as the label
                    label = subfolder.lower()  # assuming "Dog" and "Cat" are the only subfolders
                    labels.append(label)
                    # Track unique labels
                    unique_labels.add(label)
                else:
                    print(f"Unable to read image: {img_path}")
    return np.array(data), np.array(labels), list(unique_labels)

folder_path = r'C:\Users\DELL\Documents\food_set'
images, labels, unique_labels = load_images_and_labels(folder_path)

# Display sample images
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for i in range(min(5, len(images))):
    axs[i].imshow(images[i])
    axs[i].axis('off')
    axs[i].set_title(f"Label: {labels[i]}")
plt.show()

print("Labels:", labels)

# Loop through each folder in the directory
for folder in os.listdir(folder_path):
    # Use glob to find all images in the current folder
    files = gb.glob(os.path.join(folder_path, folder, '*.[jJ][pP][gG]'))
    # Print out the number of images found in the current folder
    print(f'Found {len(files)} JPG files in folder {folder}.')
    size = []

# Loop over each folder in the directory
for folder in os.listdir(folder_path):
    # Find all images in the current folder
    files = gb.glob(os.path.join(folder_path, folder, '*.[jJ][pP][gG]'))
    # Loop over each image in the folder
    for file in files:
        # Read the image and append its dimensions to the size list
        image = plt.imread(file)
        if image is not None:
            size.append(image.shape)

# Filter out None values
size = [s for s in size if s is not None]
# Print the total number of images and their dimension frequencies
print(f'The Total images we have: {len(size)}')
print(pd.Series(size, dtype='object').value_counts())
print("Number of images:", len(images))
print("Number of labels:", len(unique_labels))

hog_features_list = []

def extract_hog_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    fd, hog_image = hog(gray_image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2-Hys', visualize=True)
    hog_image_rescaled = hog_image / hog_image.max()
    return fd, hog_image_rescaled

for image in images:
    hog_feat, hog_image = extract_hog_features(image)
    hog_features_list.append(hog_feat)
    # Display the original image along with HOG and SIFT features
    plt.figure(figsize=(12, 4))
    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    # HOG Features
    plt.subplot(1, 3, 2)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Features')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

hog_features = np.array(hog_features_list)
print("hog_features shape:", hog_features.shape)

hog_features_reshaped = hog_features.reshape((hog_features.shape[0], -1))
print("hog_features_reshaped shape:", hog_features_reshaped.shape)
print('hog_features_reshaped:',hog_features_reshaped)

# Convert lists to NumPy arrays
features_array = np.array(hog_features_reshaped)
labels_array = np.array(labels)

# Use LabelEncoder to convert class names into numeric labels
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels_array)

# Normalize features using StandardScaler
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features_array)

# Define the hyperparameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)

# Fit GridSearchCV to training data
grid_search.fit(normalized_features, numeric_labels)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Use cross-validation to evaluate the model with the best hyperparameters
dt_best = DecisionTreeClassifier(**best_params, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(dt_best, normalized_features, numeric_labels, cv=5)

print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", np.mean(cv_scores))
print("Standard deviation of CV accuracy:", np.std(cv_scores))

# Make predictions
y_pred = grid_search.predict(normalized_features)

# Evaluate performance (Confusion Matrix)
cm = confusion_matrix(numeric_labels, y_pred)
print("Confusion Matrix:")
print(cm)

# Extract values from the confusion matrix
true_negatives, false_positives, false_negatives, true_positives = cm[0][0], cm[0][1], cm[1][0],cm[1][1]

# Calculate accuracy
accuracy = (true_positives + true_negatives) / sum(sum(cm))
# Print accuracy
print(f"Accuracy: {accuracy}")
# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()