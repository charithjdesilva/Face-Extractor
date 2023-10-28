import dlib
import cv2
import numpy as np
import os

def detect_and_crop_faces(image_path, min_face_size=30):
    # Load the image
    image = cv2.imread(image_path)

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Load the face landmarks predictor
    face_landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # Run face detection
    detected_faces = face_detector(image, 1)

    # Create a directory to save the cropped faces
    output_dir = 'cropped_faces'
    os.makedirs(output_dir, exist_ok=True)

    # Process each detected face
    cropped_faces = []
    for i, face_rect in enumerate(detected_faces):
        left, top, right, bottom = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()

        # Check if the detected face size is above the threshold
        if (right - left) >= min_face_size and (bottom - top) >= min_face_size:
            # Crop the detected face region
            face = image[top:bottom, left:right]

            # Save the cropped face as an image
            output_path = os.path.join(output_dir, f'cropped_face_{i}.jpg')
            cv2.imwrite(output_path, face)

            cropped_faces.append(output_path)

    return cropped_faces

# Example usage
img_path = 'image.jpg'
cropped_faces = detect_and_crop_faces(img_path)

print("Cropped faces saved in 'cropped_faces' directory.")
