import mediapipe as mp
import cv2
import torch
import numpy as np

class FaceMeshSegmenter:
    """Uses MediaPipe FaceMesh to generate semantic region masks for face parts."""
    def __init__(self, static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence
        )
        # Define landmark sets for regions of interest
        self.regions = {
            'face_oval': mp.solutions.face_mesh.FACEMESH_FACE_OVAL,
            'left_eye': mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
            'right_eye': mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
            'lips': mp.solutions.face_mesh.FACEMESH_LIPS
        }

    def get_region_masks(self, image_np, target_size=None):
        """Return a dict of region_name -> mask tensor (1xH x W) in float32."""
        h, w, _ = image_np.shape
        # Run FaceMesh
        rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return {}
        landmarks = results.multi_face_landmarks[0].landmark
        masks = {}
        for name, mesh in self.regions.items():
            # collect unique landmark indices
            idxs = set(idx for seg in mesh for idx in seg)
            pts = np.array(
                [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in idxs],
                dtype=np.int32
            )
            hull = cv2.convexHull(pts)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)
            # resize to match face tensor if requested
            if target_size is not None:
                mask = cv2.resize(mask, target_size)
            masks[name] = torch.from_numpy(mask / 255.0).unsqueeze(0).float()
        return masks
