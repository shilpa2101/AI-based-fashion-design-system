import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm

# Load Pretrained Model (ResNet50)
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = Sequential([
    model,
    GlobalMaxPooling2D()
])

# Directory where images are stored
IMAGE_FOLDER = "images"  # Change this if your images are in a different folder

# Get list of image file paths
filenames = [os.path.join(IMAGE_FOLDER, file) for file in os.listdir(IMAGE_FOLDER) if file.endswith(('.jpg', '.png', '.jpeg'))]

# Feature extraction function
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array).flatten()
    features = features / norm(features)  # Normalize features
    return features

# Extract features for all images
feature_list = np.array([extract_features(file, model) for file in filenames])

# Save features and filenames
pickle.dump(feature_list, open("embeddings.pkl", "wb"))
pickle.dump(filenames, open("filenames.pkl", "wb"))

print("✅ Embeddings generated and saved successfully!")
