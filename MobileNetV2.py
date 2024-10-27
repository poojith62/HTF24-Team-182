import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the pretrained MobileNetV2 model
model = MobileNetV2(weights='imagenet', include_top=True)

# Function to extract dominant color
def extract_dominant_color(image_path, k=3):
    img = Image.open(image_path)
    img = img.resize((100, 100))  # Resize for faster processing
    img_array = np.array(img)
    img_array = img_array.reshape(-1, 3)  # Flatten to (pixels, RGB)

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(img_array)
    dominant_color = kmeans.cluster_centers_[0]
    return tuple(dominant_color.astype(int))  # Return as RGB tuple

# Function to classify clothing type
def classify_clothing(image_path):
    # Load and preprocess image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict the class
    preds = model.predict(img_array)
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]
    
    # Get the top predicted class
    clothing_type = decoded_preds[0][1]  # Get the label of the highest prediction
    return clothing_type

# Main function
def analyze_clothing(image_path):
    clothing_type = classify_clothing(image_path)
    dominant_color = extract_dominant_color(image_path)
    return clothing_type, dominant_color

# Usage
image_path = 'jacket.jpg'  # Provide the path to your image
clothing_type, color = analyze_clothing(image_path)
print(f'Clothing Type: {clothing_type}, Dominant Color (RGB): {color}')
