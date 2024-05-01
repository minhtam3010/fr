import dlib
import cv2
import numpy as np
from ultralytics import YOLO

pose_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks_GTX.dat')
face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
model = YOLO("yolov8n-face.pt")

def test(img2_path, outProcess = False):
    img_2 = cv2.imread(img2_path)
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
    results_2 = model(img_2, device="mps")
    result_2 = results_2[0]

    bboxes_2 = np.array(result_2.boxes.xyxy.cpu(), dtype=int)

    bbox_2 = bboxes_2[0]

    face_location_2 = dlib.rectangle(bbox_2[0], bbox_2[1], bbox_2[2], bbox_2[3])
    face_landmarks_2 = pose_predictor(img_2, face_location_2)


    landmarks_tuple_2 = []

    for i in range(0, 68):
        x = face_landmarks_2.part(i).x
        y = face_landmarks_2.part(i).y

        landmarks_tuple_2.append((x, y))

    routes_2 = [i for i in range(16, -1, -1)] + [i for i in range(17, 19)] + [i for i in range(24, 26)] + [16]

    routes_coordinates_2 = []
    for i in range(0, len(routes_2) - 1):
        source_point = routes_2[i]

        source_coordinate = landmarks_tuple_2[source_point]

        routes_coordinates_2.append(source_coordinate)
    
    mask_2 = np.zeros((img_2.shape[0], img_2.shape[1]))

    mask_2 = cv2.fillConvexPoly(mask_2, np.array(routes_coordinates_2), 1)
    mask_2 = mask_2.astype(bool)

    out_2 = np.zeros_like(img_2)
    # out[mask] = img[mask]
    out_2[mask_2] = img_2[mask_2]
    
    if outProcess == False:
        out_2 = img_2
    face_chip_2 = dlib.get_face_chip(out_2, face_landmarks_2)

    img_represent_2 = np.array(face_encoder.compute_face_descriptor(face_chip_2))

    return img_represent_2


outProcess = True
rep_1 = test("first_image.jpeg", outProcess=outProcess)

rep_2 = test("second_image.jpeg", outProcess=outProcess)

# Compare the distance between the two images
distance = np.linalg.norm(rep_1-rep_2)

if distance < 0.4:
    print("Same person")
else:
    print("Different person")


