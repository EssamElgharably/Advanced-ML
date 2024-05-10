import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# Define a function to load images from directories
def load_images_from_directories(directory1, directory2):
    images = []
    labels = []
    # Load images from first directory
    for filename in os.listdir(directory1):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(directory1, filename))
            img = img.resize((100, 100))  # Resize images if needed
            images.append(np.array(img))
            labels.append(0)  # Assign label 0 for the first type of food
    # Load images from second directory
    for filename in os.listdir(directory2):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(directory2, filename))
            img = img.resize((100, 100))  # Resize images if needed
            images.append(np.array(img))
            labels.append(1)  # Assign label 1 for the second type of food
    return images, labels

# Load images from both directories
X, y = load_images_from_directories("D:\\ai project\\food_set (2)\\food_set\\sushi", "D:\\ai project\\food_set (2)\\food_set\\hamburger")

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Preprocess the data
X = X / 255.0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')  # Output layer with 2 units for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Print training accuracy
print("Training accuracy:", history.history['accuracy'][-1])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Function to classify image
def classify_image():
    # Load image
    image_path = filedialog.askopenfilename()
    if image_path:
        img = Image.open(image_path)
        img = img.resize((100, 100))  # Resize image to match model input size
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Predict image class
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence_percentage = np.max(prediction) * 100
        
        # Display result
        if predicted_class == 0:
            result_label.config(text="Sushi with {:.2f}% confidence.".format(confidence_percentage))
        else:
            result_label.config(text="Hamburger with {:.2f}% confidence.".format(confidence_percentage))
    else:
        messagebox.showerror("Error", "No image selected.")

# Create Tkinter window
root = tk.Tk()
root.title("Image Classifier")

# Create Browse button
browse_button = tk.Button(root, text="Browse", command=classify_image)
browse_button.pack(pady=10)

# Create label to display result
result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

# Run Tkinter event loop
root.mainloop()
