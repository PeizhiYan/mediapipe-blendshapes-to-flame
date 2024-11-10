# Real-time Mediapipe-based FLAME Animation Driver

Mapping **Mediapipe**'s 52 blendshapes to **FLAME**'s 50 expression coefficients and poses (jaw and eyeballs).

This can be used to drive the FLAME mesh using Mediapipe in **real-time**.


## 🔑 Method

We use both public datasets (NerSemble, IMAvatar) and our own data. First, we estimate the FLAME model coefficients (expression, pose, and eye pose) along with Mediapipe blendshape scores for each image. We then compute the linear mappings using [```./compute_mappings.ipynb```](./compute_mappings.ipynb).


## ⚖️ Disclaimer

This code and the associated mapping weights are provided for research and educational purposes only. Since public datasets were utilized in the development of the mapping weights, these weights may not be used for commercial purposes without obtaining the necessary rights. For commercial use, we recommend collecting and training on your own dataset to ensure compliance with legal and licensing requirements.

This code and the weights are provided "as-is" without any express or implied warranties, including, but not limited to, implied warranties of merchantability and fitness for a particular purpose. We make no guarantees regarding the accuracy, reliability, or fitness of the code and weights for any specific use. Use of this code and weights is entirely at your own risk, and we shall not be liable for any claims, damages, or liabilities arising from their use.



## ✨ Examples

![image](./assets/flame_animation_1.gif)
![image](./assets/flame_animation_2.gif)
![image](./assets/flame_animation_3.gif)



## 🧸 How to Use

### Convert Mediapipe blendshapes to FLAME coefficients (expression, pose, and eye pose)
```
from mp_2_flame import MP_2_FLAME
mp2flame = MP_2_FLAME(mappings_path='./mappings')

# blendshape_scores is the np.array object with shape [N,52],
# N is the number of samples, and by default N=1
exp, pose, eye_pose = mp2flame.convert(blendshape_scores=blendshape_scores)
```

### Estimate Head Pose
```
from mp_2_flame import compute_head_pose_from_mp_landmarks_3d


# download model weights from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
mediapipe_detector = vision.FaceLandmarker.create_from_options(options)

# assume img is an RGB image
img_h, img_w, img_c = img.shape
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img) # convert numpy image to Mediapipe Image

detection_result = mediapipe_detector.detect(image)

if len(detection_result.face_blendshapes) > 0:
    img_h = img.shape[0]
    img_w = img.shape[1]
    lmks_3d = detection_result.face_landmarks[0]
    lmks_3d = np.array(list(map(lambda l: np.array([l.x, l.y, l.z]), lmks_3d))) # [478, 3]
    lmks_3d[:, 0] = lmks_3d[:, 0] * img_w
    lmks_3d[:, 1] = lmks_3d[:, 1] * img_h

    # estimate the head pose
    rotation_vec, translation_vec = compute_head_pose_from_mp_landmarks_3d(face_landmarks=lmks_3d, img_h=img_h, img_w=img_w)
    print(rotation_vec, translation_vec)

```








## 🥕 Data Distributions

Following are distrubution histograms of the data we use to calculate the mappings.

### Mediapipe Blendshape Scores 😌

![image](./assets/distribution_blendshape_scores.png)

### FLAME Expression Coefficients 😃

![image](./assets/distribution_flame_expression_coefficients.png)

### FLAME Jaw Pose 😮

![image](./assets/distribution_flame_jaw_pose_distribution.png)

### FLAME Eyeballs Pose 👁️👁️

![image](./assets/distribution_flame_eye_pose_distribution.png)


