import numpy as np
import cv2
import mediapipe as mp
import time
from matplotlib import pyplot as plt


# Initialize MediaPipe FaceMesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=2, circle_radius=1)

def compute_head_pose_from_image(face_landmarks, image):
    """
    Compute head pose based on facial landmarks detected by MediaPipe FaceMesh.

    Parameters:
        face_landmarks: The facial landmarks detected by MediaPipe.
        image (np.ndarray): The input image containing the face.

    Returns:
        rotation_vec (np.ndarray): Rotation vector indicating the orientation of the head.
        translation_vec (np.ndarray): Translation vector indicating the position of the head.
    """
    # Get image dimensions
    img_h, img_w, img_c = image.shape

    # Prepare 2D and 3D landmark arrays
    face_2d = []
    face_3d = []

    # Select specific landmarks
    for idx, lm in enumerate(face_landmarks.landmark):
        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
            if idx == 1:
                nose_2d = (lm.x * img_w, lm.y * img_h)
                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
            x, y = int(lm.x * img_w), int(lm.y * img_h)

            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

    # Get 2D and 3D coordinates
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    # Set camera parameters
    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                           [0, focal_length, img_w / 2],
                           [0, 0, 1]])
    distortion_matrix = np.zeros((4, 1), dtype=np.float64)

    # Pose estimation
    success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

    ### For drawing the results only
    # Get rotation angles
    rmat, jac = cv2.Rodrigues(rotation_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360

    # Determine head direction
    if y < -10:
        text = "Looking Left"
    elif y > 10:
        text = "Looking Right"
    elif x < -10:
        text = "Looking Down"
    elif x > 10:
        text = "Looking Up"
    else:
        text = "Forward"

    # Draw results
    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rotation_vec, translation_vec, cam_matrix,
                                                     distortion_matrix)
    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

    cv2.line(image, p1, p2, (255, 0, 0), 3)
    cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw face mesh
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec
    )

    image_rgb1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb1)
    plt.axis('off')  # Hide axes
    plt.show()

    return rotation_vec, translation_vec

# Load the image
image_path = '/mnt/data1_nvme/haoyu/face2ear_data/ear_data/CollectionA/train/train_0003.png'  # Replace with your image path
image = cv2.imread(image_path)

# Check if the image is loaded successfully
if image is None:
    print("Image loading failed. Please check the path.")
else:
    # Start processing
    start = time.time()

    # Convert color space to RGB, as MediaPipe model requires RGB input
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    # Process face landmarks
    if results.multi_face_landmarks:
        num_faces = len(results.multi_face_landmarks)
        print("Number of faces detected:", num_faces)
        for face_landmarks in results.multi_face_landmarks:
            rotation_vec, translation_vec = compute_head_pose_from_image(face_landmarks, image)

    # Display the processed image
    end = time.time()
    print("Processing time: ", end - start)

