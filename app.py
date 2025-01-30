from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('./Flood Semantic Segmentation.keras')

def preprocess_image(image):
    # Ensure the image is RGB
    if image.mode != "RGB":
        '''
        I was experiencing an issue with predicting from the model because the input images were not consistently in the RGB format, which the model requires. Some images were in grayscale or had an alpha channel, causing dimension mismatches during processing. To resolve this, I added a check to ensure all images are converted to RGB using image.convert("RGB"), which standardizes the input format and prevents errors.
        For example: A grayscale image will have image.mode = "L". and An image with transparency will have image.mode = "RGBA".
        If the image is grayscale ("L"), it duplicates the intensity values across the three channels.
        If the image has an alpha channel ("RGBA"), the alpha channel is discarded, and the remaining channels are adjusted to fit the RGB format.
        '''
        image = image.convert("RGB")
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Resize to model input size
    img_resized = cv2.resize(img_array, (256, 256))
    
    # Normalize to [0, 1]
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Check validity
    if img_batch.shape != (1, 256, 256, 3):
        raise ValueError(f"Invalid input shape: {img_batch.shape}")
    if img_batch.min() < 0 or img_batch.max() > 1:
        raise ValueError("Image normalization failed.")
    
    return img_batch


def get_colored_mask(mask):
    # Convert to RGB colormap using viridis
    colormap = plt.get_cmap('viridis')
    colored_mask = (colormap(mask.squeeze()) * 255).astype(np.uint8)
    return colored_mask

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the POST request
        file = request.files['image']
        img = Image.open(file.stream)
        
        # Preprocess the image
        processed_img = preprocess_image(img)

        print(f'PREPROCESSED IMAGE: {processed_img}')
        
        # Make prediction
        prediction = model.predict(processed_img)

        print(f'PREDICTION: {prediction}')
        
        # Convert prediction to binary mask
        binary_mask = (prediction > 0.5).astype(np.float32)
        
        # Convert mask to colored visualization
        colored_mask = get_colored_mask(binary_mask[0])
        
        # Convert the colored mask to PIL Image
        mask_image = Image.fromarray(colored_mask)
        
        # Convert PIL image to base64 string
        buffered = io.BytesIO()
        mask_image.save(buffered, format="PNG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({'mask': mask_base64})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)