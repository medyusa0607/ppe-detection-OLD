import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# Load your models (make sure best.pt and yolov8n.pt are in this folder)
ppe_model = YOLO('best.pt')
person_model = YOLO('yolov8n.pt')

# Setup MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def check_face_touch(person_crop, tau=0.4):
    if person_crop.size == 0:
        return False
    h, w = person_crop.shape[:2]
    rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_crop)
    if not results.pose_landmarks:
        return False
    lm = results.pose_landmarks.landmark
    
    def pt(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h])
        
    nose = pt(mp_pose.PoseLandmark.NOSE)
    lwrist = pt(mp_pose.PoseLandmark.LEFT_WRIST)
    rwrist = pt(mp_pose.PoseLandmark.RIGHT_WRIST)
    lshoulder = pt(mp_pose.PoseLandmark.LEFT_SHOULDER)
    rshoulder = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    shoulder_width = np.linalg.norm(lshoulder - rshoulder)
    
    if shoulder_width == 0: return False
    
    ldist = np.linalg.norm(nose - lwrist) / shoulder_width
    rdist = np.linalg.norm(nose - rwrist) / shoulder_width
    return min(ldist, rdist) < tau

def is_contained(item_box, person_box, threshold=0.6):
    ix1 = max(item_box[0], person_box[0])
    iy1 = max(item_box[1], person_box[1])
    ix2 = min(item_box[2], person_box[2])
    iy2 = min(item_box[3], person_box[3])
    item_area = (item_box[2] - item_box[0]) * (item_box[3] - item_box[1])
    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if item_area == 0: return False
    return (inter_area / item_area) >= threshold

def detect(frame_content, camera_id, session_id):
    """
    This function is called by Kavya's worker.
    It must return a list of person dictionaries.
    """
    # 1. Decode the image bytes from the cloud
    nparr = np.frombuffer(frame_content, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return []

    # 2. Run inference
    ppe_results = ppe_model(frame, verbose=False)[0]
    person_results = person_model.track(frame, persist=True, tracker='bytetrack.yaml', verbose=False)[0]

    persons = []
    ppe_items = []

    # Extract Persons
    if person_results.boxes is not None and person_results.boxes.id is not None:
        p_boxes = person_results.boxes.xyxy.cpu().numpy()
        p_ids = person_results.boxes.id.cpu().numpy().astype(int)
        for box, track_id in zip(p_boxes, p_ids):
            persons.append({'track_id': track_id, 'box': box})

    # Extract PPE
    if ppe_results.boxes is not None:
        ppe_boxes = ppe_results.boxes.xyxy.cpu().numpy()
        ppe_classes = ppe_results.boxes.cls.cpu().numpy().astype(int)
        for box, cls_id in zip(ppe_boxes, ppe_classes):
            class_name = ppe_model.names[cls_id]
            ppe_items.append({'class_name': class_name, 'box': box})

    # 3. Format for the cloud backend
    output_persons = []
    for person in persons:
        px1, py1, px2, py2 = map(int, person['box'])
        person_crop = frame[max(0, py1): min(frame.shape[0], py2), max(0, px1): min(frame.shape[1], px2)]
        
        face_touch = check_face_touch(person_crop)
        associated_ppe = []
        
        for item in ppe_items:
            if is_contained(item['box'], person['box']):
                associated_ppe.append(item['class_name'])

        # Determine PPE states
        ppe_status = {
            "hairnet": "missing" if "no_hairnet" in associated_ppe else "present" if "hairnet" in associated_ppe else "unknown",
            "mask": "missing" if "maskoff" in associated_ppe else "present" if "maskon" in associated_ppe else "unknown",
            "gloves": "missing" if "no_glove" in associated_ppe else "present" if "glove" in associated_ppe else "unknown"
        }

        output_persons.append({
            "track_id": int(person['track_id']),
            "box": [px1, py1, px2, py2],
            "ppe": ppe_status,
            "face_touch": bool(face_touch)
        })

    return output_persons