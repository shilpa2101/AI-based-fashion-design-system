import os
import cv2
import numpy as np
import mediapipe as mp
import base64
from io import BytesIO
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create necessary directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_image_base64(image_path):
    """Convert image to base64 for embedding in HTML"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

class VirtualTryOn:
    def __init__(self):
        # Initialize MediaPipe Pose for body landmark detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        
    def process_images(self, human_image_path, clothing_image_path, output_path):
        """
        Main function to process the human and clothing images and create a virtual try-on
        
        Args:
            human_image_path: Path to the full-size human image
            clothing_image_path: Path to the clothing/dress image
            output_path: Path to save the final output image
        
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            # Load images
            human_img = cv2.imread(human_image_path)
            if human_img is None:
                return False, f"Failed to load human image from {human_image_path}"
            
            human_img = cv2.cvtColor(human_img, cv2.COLOR_BGR2RGB)
            
            clothing_img = cv2.imread(clothing_image_path, cv2.IMREAD_UNCHANGED)
            if clothing_img is None:
                return False, f"Failed to load clothing image from {clothing_image_path}"
            
            # If clothing doesn't have alpha channel, create one
            if clothing_img.shape[2] == 3:
                clothing_img = cv2.cvtColor(clothing_img, cv2.COLOR_BGR2RGBA)
                # Create alpha mask (using background removal)
                clothing_gray = cv2.cvtColor(clothing_img[:,:,:3], cv2.COLOR_RGB2GRAY)
                _, alpha_mask = cv2.threshold(clothing_gray, 240, 255, cv2.THRESH_BINARY_INV)
                clothing_img[:,:,3] = alpha_mask
            
            # Get body landmarks
            landmarks = self.detect_landmarks(human_img)
            if landmarks is None:
                return False, "No body detected in the human image. Please use a full body image."
            
            # Calculate clothing placement based on body landmarks
            clothing_transformed = self.transform_clothing(clothing_img, landmarks, human_img.shape)
            
            # Overlay clothing on human image
            result = self.overlay_images(human_img, clothing_transformed)
            
            # Save result
            result_rgb = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_rgb)
            return True, "Virtual try-on completed successfully"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Error processing images: {str(e)}"
        
    def detect_landmarks(self, image):
        """Detect body landmarks using MediaPipe Pose"""
        results = self.pose.process(image)
        if not results.pose_landmarks:
            return None
        
        landmarks = {}
        # Extract key landmarks for clothing placement
        landmark_points = results.pose_landmarks.landmark
        h, w, _ = image.shape
        
        # Map important landmarks for clothing alignment
        landmarks['left_shoulder'] = (int(landmark_points[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                                     int(landmark_points[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
        landmarks['right_shoulder'] = (int(landmark_points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                                      int(landmark_points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
        landmarks['left_hip'] = (int(landmark_points[self.mp_pose.PoseLandmark.LEFT_HIP].x * w),
                                int(landmark_points[self.mp_pose.PoseLandmark.LEFT_HIP].y * h))
        landmarks['right_hip'] = (int(landmark_points[self.mp_pose.PoseLandmark.RIGHT_HIP].x * w),
                                 int(landmark_points[self.mp_pose.PoseLandmark.RIGHT_HIP].y * h))
        landmarks['neck'] = ((landmarks['left_shoulder'][0] + landmarks['right_shoulder'][0]) // 2,
                            (landmarks['left_shoulder'][1] + landmarks['right_shoulder'][1]) // 2)
        
        return landmarks
    
    def transform_clothing(self, clothing_img, landmarks, target_shape):
        """Transform clothing to fit the person's body"""
        # Calculate clothing dimensions based on body landmarks
        person_width = abs(landmarks['left_shoulder'][0] - landmarks['right_shoulder'][0]) * 1.5
        person_height = abs(landmarks['neck'][1] - landmarks['left_hip'][1]) * 2.2
        
        # Resize clothing maintaining aspect ratio
        clothing_h, clothing_w = clothing_img.shape[:2]
        aspect_ratio = clothing_w / clothing_h
        
        new_height = int(person_height)
        new_width = int(new_height * aspect_ratio)
        
        # Ensure width is appropriate
        if new_width < person_width:
            new_width = int(person_width)
            new_height = int(new_width / aspect_ratio)
        
        resized_clothing = cv2.resize(clothing_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create a transparent canvas of the target image size
        result = np.zeros((target_shape[0], target_shape[1], 4), dtype=np.uint8)
        
        # Calculate position (center clothing on torso)
        x_offset = max(0, landmarks['neck'][0] - new_width // 2)
        # Place top of clothing at estimated collar position (slightly below neck)
        y_offset = max(0, landmarks['neck'][1] - int(new_height * 0.1))
        
        # Ensure the offsets don't place the clothing outside the image
        x_offset = min(x_offset, target_shape[1] - new_width)
        y_offset = min(y_offset, target_shape[0] - new_height)
        
        # Place the clothing on the canvas
        if x_offset >= 0 and y_offset >= 0:
            alpha_s = resized_clothing[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            
            # Region of interest in the target image
            roi_h = min(new_height, target_shape[0] - y_offset)
            roi_w = min(new_width, target_shape[1] - x_offset)
            
            for c in range(3):
                result[y_offset:y_offset+roi_h, x_offset:x_offset+roi_w, c] = (
                    alpha_s[:roi_h, :roi_w] * resized_clothing[:roi_h, :roi_w, c] + 
                    alpha_l[:roi_h, :roi_w] * 0
                ).astype(np.uint8)
                
            result[y_offset:y_offset+roi_h, x_offset:x_offset+roi_w, 3] = (
                alpha_s[:roi_h, :roi_w] * 255
            ).astype(np.uint8)
        
        return result
    
    def overlay_images(self, human_img, clothing_img_with_alpha):
        """Overlay the clothing on the human image"""
        # Convert human image to RGBA
        human_rgba = np.zeros((human_img.shape[0], human_img.shape[1], 4), dtype=np.uint8)
        human_rgba[:, :, :3] = human_img
        human_rgba[:, :, 3] = 255  # Fully opaque
        
        # Alpha blending
        alpha_clothing = clothing_img_with_alpha[:, :, 3] / 255.0
        alpha_human = 1.0 - alpha_clothing
        
        result = np.zeros_like(human_rgba)
        
        for c in range(3):
            result[:, :, c] = (alpha_clothing * clothing_img_with_alpha[:, :, c] + 
                              alpha_human * human_rgba[:, :, c]).astype(np.uint8)
        
        result[:, :, 3] = 255  # Final result is fully opaque
        
        return result

# Initialize the virtual try-on processor
try_on_processor = VirtualTryOn()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'human_image' not in request.files or 'dress_image' not in request.files:
        return render_template('index.html', error='Both human and dress images are required')
    
    human_file = request.files['human_image']
    dress_file = request.files['dress_image']
    
    if human_file.filename == '' or dress_file.filename == '':
        return render_template('index.html', error='No selected file')
    
    if not (human_file and allowed_file(human_file.filename) and 
            dress_file and allowed_file(dress_file.filename)):
        return render_template('index.html', error='Invalid file format. Please use JPG, JPEG, or PNG')
    
    # Save uploaded files
    human_filename = secure_filename(human_file.filename)
    dress_filename = secure_filename(dress_file.filename)
    
    human_path = os.path.join(app.config['UPLOAD_FOLDER'], human_filename)
    dress_path = os.path.join(app.config['UPLOAD_FOLDER'], dress_filename)
    
    human_file.save(human_path)
    dress_file.save(dress_path)
    
    # Generate unique result filename
    import time
    timestamp = int(time.time())
    result_filename = f"result_{timestamp}.jpg"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    
    # Process images
    success, message = try_on_processor.process_images(human_path, dress_path, result_path)
    
    if success:
        # Get base64 encoded images for templates
        human_b64 = get_image_base64(human_path)
        dress_b64 = get_image_base64(dress_path)
        result_b64 = get_image_base64(result_path)
        
        return render_template('result.html', 
                               result_image=result_filename,
                               human_image_b64=human_b64,
                               dress_image_b64=dress_b64,
                               result_image_b64=result_b64)
    else:
        return render_template('index.html', error=message)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/uploads/<filename>')
def upload_file_view(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

