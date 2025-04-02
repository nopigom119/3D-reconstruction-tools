import cv2
import numpy as np
import os
import glob
import open3d as o3d
from typing import List, Tuple, Dict, Optional, Any
import threading
import time # 스레드 예시용

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext

# --- 기본 파라미터 값 (GUI 기본값으로 사용) ---
DEFAULT_PARAMS = {
    "RATIO_TEST_THRESHOLD": 0.75,
    "INLOOP_SOR_NEIGHBORS": 20,
    "INLOOP_SOR_STD_RATIO": 2.0,
    "FINAL_SOR_NEIGHBORS": 20,
    "FINAL_SOR_STD_RATIO": 1.5,
    "MIN_MATCHES_PNP": 5,
    "RANSAC_ITERATIONS_PNP": 100,
    "RANSAC_REPROJECTION_ERROR_PNP": 8.0,
    "RANSAC_CONFIDENCE_PNP": 0.99,
}

# --- Configuration (이제 main 함수에서 params 딕셔너리로 관리) ---
# OUTPUT_PLY_INCREMENTAL = 'output_incremental.ply' # 출력 파일명 고정 또는 GUI 추가 가능
# OUTPUT_DIR_INCREMENTAL_STEPS = 'output_ply_by_process'
K = None # 전역 변수 -> main 함수 내 지역 변수로 변경 고려

# --- 데이터 구조 (변경 없음) ---
class MapPoint:
    def __init__(self, pt_id: int, coords: np.ndarray):
        self.id = pt_id
        self.coords = coords # 3D 좌표 (3,)
        self.observations: List[Tuple[int, int]] = [] # (view_idx, keypoint_idx)
        self.descriptor: Optional[np.ndarray] = None # 대표 기술자
        self.color: Optional[np.ndarray] = None      # 색상 정보 (BGR, uint8)

class GlobalMap:
    def __init__(self):
        self.points: Dict[int, MapPoint] = {}
        self.next_point_id = 0
        self.descriptors_list = []
        self.point_id_from_descriptor_idx = []

    def add_point(self, coords: np.ndarray, descriptor: np.ndarray, view_idx: int, kp_idx: int, color: np.ndarray):
        pt_id = self.next_point_id
        map_point = MapPoint(pt_id, coords)
        map_point.observations.append((view_idx, kp_idx))
        map_point.descriptor = descriptor
        map_point.color = color # 색상 저장

        self.points[pt_id] = map_point
        # 기술자 리스트 및 ID 매핑 업데이트 (중복 기술자 고려 필요 시 로직 변경)
        self.descriptors_list.append(descriptor)
        self.point_id_from_descriptor_idx.append(pt_id)
        self.next_point_id += 1
        return pt_id

    def get_points_coords(self) -> np.ndarray:
        if not self.points: return np.empty((0, 3))
        return np.array([p.coords for p in self.points.values()])

    def get_points_colors(self) -> np.ndarray:
        if not self.points: return np.empty((0, 3))
        colors = []
        default_color = np.array([128, 128, 128], dtype=np.uint8) # BGR 회색
        for p in self.points.values():
            colors.append(p.color if p.color is not None else default_color)
        # 색상 배열이 비어있는 경우 처리
        if not colors: return np.empty((0, 3), dtype=np.uint8)
        return np.array(colors, dtype=np.uint8) # (N, 3) 형태의 BGR 색상 배열

    def get_descriptors_for_matching(self) -> Optional[np.ndarray]:
        if not self.descriptors_list: return None
        try:
            # float32 타입 확인 및 변환
            return np.array(self.descriptors_list, dtype=np.float32)
        except ValueError as e:
            print(f"Error converting descriptors list to numpy array: {e}")
            # 리스트 내 요소 타입 확인
            # for i, desc in enumerate(self.descriptors_list):
            #     if not isinstance(desc, np.ndarray) or desc.dtype != np.float32:
            #         print(f"Invalid descriptor at index {i}: type={type(desc)}, dtype={getattr(desc, 'dtype', 'N/A')}")
            return None # 오류 발생 시 None 반환

    def get_point_id_from_desc_idx(self, desc_idx: int) -> Optional[int]:
         # 인덱스 범위 확인
        if 0 <= desc_idx < len(self.point_id_from_descriptor_idx):
            return self.point_id_from_descriptor_idx[desc_idx]
        else:
            print(f"Warning: Descriptor index {desc_idx} out of bounds.")
            return None

    def get_point_coords_by_id(self, pt_id: int) -> Optional[np.ndarray]:
        return self.points[pt_id].coords if pt_id in self.points else None

# --- Helper Functions (파라미터 전달 방식으로 수정) ---

def load_intrinsic_matrix(filepath: str, status_callback) -> Optional[np.ndarray]:
    """ Intrinsic 행렬 K를 파일에서 로드합니다. """
    try:
        K_loaded = np.loadtxt(filepath)
        if K_loaded.shape == (3, 3):
            status_callback(f"로드된 K 행렬:\n{K_loaded}")
            return K_loaded
        else:
            status_callback(f"오류: K 행렬 파일 {filepath}에 3x3 행렬이 없습니다.")
            return None
    except Exception as e:
        status_callback(f"파일 {filepath}에서 K 행렬 로드 중 오류 발생: {e}")
        return None

def extract_features(image: np.ndarray, status_callback) -> Tuple[Optional[List[cv2.KeyPoint]], Optional[np.ndarray]]:
    """ 이미지에서 SIFT 특징점과 기술자를 추출합니다. """
    if image is None:
        status_callback("오류: 입력 이미지가 None입니다.")
        return None, None
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        gray = image
    else:
        status_callback(f"오류: 예상치 못한 이미지 형태: {image.shape}")
        return None, None

    try:
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
    except cv2.error as e:
        status_callback(f"SIFT 특징점 추출 중 OpenCV 오류: {e}")
        return None, None


    if keypoints is None or descriptors is None:
        status_callback("경고: 특징점 추출 실패.")
        return None, None
    # status_callback(f"  특징점 {len(keypoints)}개 발견.") # 로그가 너무 많아질 수 있음
    return keypoints, descriptors

def match_features(des1: np.ndarray, des2: np.ndarray, ratio_threshold: float, status_callback) -> List[cv2.DMatch]:
    """ 두 기술자 집합 간의 특징점을 매칭합니다 (Ratio Test 적용). """
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        # status_callback("경고: 매칭을 위한 기술자 부족.")
        return []

    # BFMatcher 생성 (한 번만 생성하여 재사용하는 것이 효율적일 수 있음)
    try:
        bf = cv2.BFMatcher(cv2.NORM_L2)
        # knnMatch 수행
        matches = bf.knnMatch(des1, des2, k=2)
    except cv2.error as e:
        status_callback(f"특징점 매칭 중 OpenCV 오류: {e}")
        return []


    # Ratio Test 적용
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    # status_callback(f"  Ratio Test 후 좋은 매칭 {len(good_matches)}개 발견.")
    return good_matches

def get_color_from_image(image: np.ndarray, kp: cv2.KeyPoint) -> np.ndarray:
    """ 이미지의 특정 키포인트 위치에서 BGR 색상을 추출합니다. """
    if image is None: return np.array([128, 128, 128], dtype=np.uint8) # 기본 회색
    pt = tuple(map(int, kp.pt))
    h, w = image.shape[:2]
    if 0 <= pt[1] < h and 0 <= pt[0] < w:
        return image[pt[1], pt[0]]
    else:
        return np.array([128, 128, 128], dtype=np.uint8)

def estimate_essential_matrix(kp1, kp2, matches, K, status_callback):
    # 이전과 유사, 최소 매칭 수 하드코딩 대신 파라미터 사용 가능
    min_matches_for_e = 15 # 또는 params에서 가져오기
    if len(matches) < min_matches_for_e:
        status_callback(f" E 추정 실패: 매칭 부족 ({len(matches)} < {min_matches_for_e})")
        return None, None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    try:
        E, mask_e = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=0.4) # threshold 파라미터화 가능
    except cv2.error as e:
        status_callback(f"Essential Matrix 추정 중 OpenCV 오류: {e}")
        return None, None

    if E is None or mask_e is None: return None, None
    num_inliers = int(np.sum(mask_e))
    status_callback(f" E 추정: {num_inliers} inliers")
    if num_inliers < min_matches_for_e: return None, None
    return E, mask_e.ravel() == 1

def recover_pose_from_essential(E, kp1, kp2, matches, mask_e, K, status_callback):
    # 이전 코드와 유사, cv2.recoverPose 사용
    pts1_inliers = np.float32([kp1[m.queryIdx].pt for m_idx, m in enumerate(matches) if mask_e[m_idx]]).reshape(-1, 1, 2)
    pts2_inliers = np.float32([kp2[m.trainIdx].pt for m_idx, m in enumerate(matches) if mask_e[m_idx]]).reshape(-1, 1, 2)

    if len(pts1_inliers) < 5: # recoverPose 최소 포인트 수
         status_callback("recoverPose 실패: E-matrix 인라이어 부족")
         return None, None, None

    try:
        points, R, t, mask_rp = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)
    except cv2.error as e:
        status_callback(f"recoverPose 중 OpenCV 오류: {e}")
        return None, None, None


    if points > 10 and R is not None and t is not None: # points 임계값 조정 가능
        final_mask = np.zeros(len(matches), dtype=bool)
        e_indices = np.where(mask_e)[0] # E-matrix 인라이어의 원래 matches 인덱스

        if mask_rp is not None:
            # --- 여기가 수정 포인트 ---
            # mask_rp에서 0이 아닌 값을 인라이어로 간주 (== 1 대신 != 0 사용)
            rp_inlier_indices_rel = np.where(mask_rp.ravel() != 0)[0]
            # --- 수정 끝 ---

            if len(rp_inlier_indices_rel) > 0: # recoverPose가 보고한 인라이어가 실제로 있는지 확인
                    # 인덱스 범위 체크 강화
                if len(e_indices) > 0 and max(rp_inlier_indices_rel) < len(e_indices):
                    final_original_indices = e_indices[rp_inlier_indices_rel] # 상대 인덱스를 원래 matches 인덱스로 변환
                    final_mask[final_original_indices] = True
                    status_callback(f" recoverPose 성공: {len(final_original_indices)}개의 최종 인라이어 사용.") # 성공 로그 추가
                    return R, t, final_mask
                else:
                        status_callback("경고: recoverPose 마스크 변환 중 인덱스 범위 문제 발생.")
                        # 안전하게 실패 처리
                        return None, None, None
            else:
                # cv2.recoverPose가 points > 10을 반환했지만, 마스크에 0만 있는 경우
                    status_callback("recoverPose 실패: 반환된 마스크에 인라이어 없음 (모두 0).")
                    return None, None, None
        else: # mask_rp가 None인 경우
            status_callback("경고: cv2.recoverPose가 마스크를 반환하지 않음.")
            # 실패 처리
            return None, None, None
    else:
            # points <= 10 이거나 R 또는 t가 None인 경우
        status_callback(f"recoverPose 실패: 유효한 R, t 얻기 실패 또는 인라이어 부족 (points={points})")
        return None, None, None

def triangulate_points_incremental(kp1, kp2, matches, mask, P1, P2, status_callback):
    # 이전과 유사
    valid_indices = np.where(mask)[0]
    if len(valid_indices) == 0: return None, None

    pts1_indices = [matches[i].queryIdx for i in valid_indices]
    pts2_indices = [matches[i].trainIdx for i in valid_indices]

    # 키포인트 인덱스 유효성 검사 강화
    if any(idx >= len(kp1) for idx in pts1_indices) or any(idx >= len(kp2) for idx in pts2_indices):
        status_callback("오류: 삼각측량 중 유효하지 않은 키포인트 인덱스.")
        return None, None

    pts1 = np.float32([kp1[idx].pt for idx in pts1_indices]).reshape(-1, 2).T
    pts2 = np.float32([kp2[idx].pt for idx in pts2_indices]).reshape(-1, 2).T

    if pts1.shape[1] == 0 or pts2.shape[1] == 0: return None, None # 입력 포인트 없음

    try:
        points_4d_hom = cv2.triangulatePoints(P1, P2, pts1, pts2)
    except cv2.error as e:
        status_callback(f"삼각측량 중 OpenCV 오류: {e}")
        return None, None

    w = points_4d_hom[3]
    # w가 너무 작은 경우(무한대 포인트) 필터링 강화
    valid_depth_mask = np.abs(w) > 1e-6
    points_4d_hom = points_4d_hom[:, valid_depth_mask]
    w = w[valid_depth_mask]

    if points_4d_hom.shape[1] == 0: return None, None

    points_3d = points_4d_hom[:3, :] / w

    # Cheirality check (선택적이지만 중요)
    P1_R, P1_t = P1[:,:3], P1[:,3]
    P2_R, P2_t = P2[:,:3], P2[:,3]
    points_cam1 = P1_R @ points_3d + P1_t.reshape(-1,1)
    points_cam2 = P2_R @ points_3d + P2_t.reshape(-1,1)
    cheirality_mask = (points_cam1[2,:] > 0) & (points_cam2[2,:] > 0)

    points_3d = points_3d[:, cheirality_mask]
    success_indices_relative_to_valid = np.where(cheirality_mask)[0]

    if points_3d.shape[1] == 0: return None, None

    # 성공한 점들의 원래 매치 리스트 내 인덱스 반환
    original_indices_masked = valid_indices[valid_depth_mask] # w > 0 인 점들의 원래 인덱스
    success_original_match_indices = original_indices_masked[success_indices_relative_to_valid]

    return points_3d.T, success_original_match_indices


def match_features_2d_3d(des_new: np.ndarray, map_des: np.ndarray, matcher: cv2.BFMatcher, ratio_threshold: float, status_callback) -> List[Tuple[int, int]]:
    """ 새로운 이미지의 기술자와 맵의 기술자 매칭 (PnP 용) """
    matches_2d_3d = []
    if des_new is None or map_des is None or len(des_new) == 0 or len(map_des) == 0:
        # status_callback("2D-3D 매칭: 입력 기술자 없음.")
        return matches_2d_3d

    try:
        raw_matches = matcher.knnMatch(des_new, map_des, k=2)
    except cv2.error as e:
         status_callback(f"2D-3D 매칭 중 OpenCV 오류: {e}")
         return matches_2d_3d


    for match_pair in raw_matches:
         # k=2 이지만 가끔 결과가 부족할 수 있음
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                # (queryIdx: des_new의 인덱스, trainIdx: map_des의 인덱스)
                matches_2d_3d.append((m.queryIdx, m.trainIdx))
    # status_callback(f" 2D-3D 매칭: {len(matches_2d_3d)} matches found.")
    return matches_2d_3d

def estimate_pose_pnp(kp_new: List[cv2.KeyPoint], matches_2d_3d: List[Tuple[int, int]], global_map: GlobalMap, K: np.ndarray, params: Dict, status_callback) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """ 2D-3D 매칭을 이용해 카메라의 전역 포즈 추정 """
    min_matches_pnp = params.get("MIN_MATCHES_PNP", 5)

    if len(matches_2d_3d) < min_matches_pnp:
        status_callback(f" PnP 실패: 매칭 부족 ({len(matches_2d_3d)} < {min_matches_pnp})")
        return None

    image_points = []
    object_points = []
    valid_match_indices_in_matches = [] # matches_2d_3d 리스트 내의 유효한 매칭 인덱스

    for idx, (kp_idx, map_desc_idx) in enumerate(matches_2d_3d):
        # 키포인트 인덱스 유효성 검사
        if kp_idx >= len(kp_new):
            # status_callback(f"Warning: PnP 매칭에서 유효하지 않은 키포인트 인덱스 {kp_idx}")
            continue

        map_point_id = global_map.get_point_id_from_desc_idx(map_desc_idx)
        # MapPoint ID 유효성 검사
        if map_point_id is None:
            # status_callback(f"Warning: PnP 매칭에서 유효하지 않은 맵 디스크립터 인덱스 {map_desc_idx}")
            continue

        map_point_coords = global_map.get_point_coords_by_id(map_point_id)

        if map_point_coords is not None:
            image_points.append(kp_new[kp_idx].pt)
            object_points.append(map_point_coords)
            valid_match_indices_in_matches.append(idx)

    if len(image_points) < min_matches_pnp:
        status_callback(f" PnP 실패: 유효 3D 포인트 부족 ({len(image_points)} < {min_matches_pnp})")
        return None

    image_points_arr = np.array(image_points, dtype=np.float32)
    object_points_arr = np.array(object_points, dtype=np.float32)

    try:
        # PnP 파라미터 사용
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points_arr, image_points_arr, K, None, # distCoeffs
            iterationsCount=params.get("RANSAC_ITERATIONS_PNP", 100),
            reprojectionError=params.get("RANSAC_REPROJECTION_ERROR_PNP", 8.0),
            confidence=params.get("RANSAC_CONFIDENCE_PNP", 0.99),
            flags=cv2.SOLVEPNP_EPNP # 또는 다른 PnP 알고리즘
        )
    except cv2.error as e:
        status_callback(f" PnP cv2.error: {e}")
        return None

    if success and inliers is not None and len(inliers) >= min_matches_pnp:
        R, _ = cv2.Rodrigues(rvec)
        status_callback(f" PnP 성공: {len(inliers)} inliers.")
        # PnP inlier에 해당하는 원래 매칭 인덱스 반환
        # inliers는 pnp_object_points_arr/pnp_image_points_arr 기준 인덱스 배열
        # valid_match_indices_in_matches는 pnp에 사용된 점들이 원래 matches_2d_3d에서 몇 번째였는지를 저장
        pnp_inlier_match_indices = np.array(valid_match_indices_in_matches)[inliers.flatten()]
        return R, tvec, pnp_inlier_match_indices # 전역 포즈 (R, t), PnP 인라이어의 원래 매칭 인덱스
    else:
        status_callback(f" PnP 실패: RANSAC 실패 (inliers: {len(inliers) if inliers is not None else 'None'})")
        return None

def save_ply(filepath: str, points_3d: np.ndarray, colors_bgr: Optional[np.ndarray] = None, status_callback=print):
    """ 최종 포인트 클라우드를 PLY 파일로 저장 (색상 정보 포함) """
    pcd = o3d.geometry.PointCloud()
    if points_3d is None or len(points_3d) == 0:
        status_callback(f"경고: '{filepath}'에 저장할 포인트 없음"); return

    pcd.points = o3d.utility.Vector3dVector(points_3d)

    if colors_bgr is not None and colors_bgr.shape[0] == points_3d.shape[0]:
        if colors_bgr.dtype == np.uint8:
            colors_rgb = colors_bgr[:, ::-1]
            colors_normalized = colors_rgb.astype(np.float64) / 255.0
             # 값 범위 클램핑 (안전 장치)
            colors_normalized = np.clip(colors_normalized, 0.0, 1.0)
            pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
            # status_callback(f"색상 정보 ({len(colors_normalized)} points) 함께 저장 준비됨.")
        else:
            status_callback("경고: 예상치 못한 색상 데이터 타입, 회색으로 대체.")
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
    else:
        # status_callback("경고: 유효한 색상 정보 없음, 회색으로 저장.")
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

    output_dir = os.path.dirname(filepath);
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
             status_callback(f"오류: 출력 폴더 생성 실패 '{output_dir}': {e}")
             return # 폴더 생성 실패 시 저장 불가

    try:
        o3d.io.write_point_cloud(filepath, pcd, write_ascii=False) # 바이너리 저장이 효율적
        status_callback(f"포인트 클라우드 ({len(points_3d)} points) 저장 완료: {filepath}")
    except Exception as e_save:
        status_callback(f"PLY 저장 오류: {e_save}")


# --- Main Incremental Reconstruction Function ---
def main_incremental_reconstruction(data_directory: str, params: Dict, status_callback):
    """
    이미지 시퀀스를 사용하여 점진적으로 3D 포인트 클라우드를 재구성합니다.
    GUI에서 설정된 파라미터를 사용합니다.
    """
    output_ply_incremental = 'output_incremental.ply' # 고정 또는 파라미터화
    output_dir_incremental_steps = 'output_ply_by_process'

    status_callback(f"--- 점진적 재구성 시작 (데이터 폴더: {data_directory}) ---")
    status_callback(f"사용 파라미터: {params}")

    # K 행렬 로드
    k_matrix = load_intrinsic_matrix(os.path.join(data_directory, 'K.txt'), status_callback)
    if k_matrix is None:
        status_callback("치명적 오류: K 행렬 로드 실패. 프로그램을 종료합니다.")
        return

    # 이미지 파일 목록 가져오기
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
    all_files_in_dir = glob.glob(os.path.join(data_directory, '*'))
    image_files = sorted([
        f for f in all_files_in_dir
        if os.path.isfile(f) and f.lower().endswith(supported_extensions)
    ])
    num_images = len(image_files)
    if num_images < 2:
        status_callback(f"오류: '{data_directory}' 폴더에서 최소 2개 이상의 지원되는 이미지 파일을 찾을 수 없습니다.")
        return
    status_callback(f"총 {num_images}개의 이미지 파일을 찾았습니다.")


    # 중간 단계 저장 폴더 생성
    try:
        os.makedirs(output_dir_incremental_steps, exist_ok=True)
        status_callback(f"중간 단계 PLY 저장 폴더: '{output_dir_incremental_steps}'")
    except OSError as e:
        status_callback(f"오류: 중간 저장 폴더 생성 실패: {e}")
        # 계속 진행할지 여부 결정 (저장 없이)
        # return

    # BFMatcher 생성 (재사용)
    bf_matcher = cv2.BFMatcher(cv2.NORM_L2)

    # 데이터 저장용 딕셔너리 초기화
    poses = {} # 전역 포즈 (R, t) 저장
    features = {} # 특징점 (keypoints, descriptors) 저장
    images_loaded = {} # 로드된 이미지 저장 (색상 추출용)

    # --- 1. 초기화 (첫 번째 쌍 처리) ---
    status_callback("\n--- 초기화 단계 (Image 0 - Image 1) ---")
    img0_path, img1_path = image_files[0], image_files[1]
    img0 = cv2.imread(img0_path); img1 = cv2.imread(img1_path)
    if img0 is None or img1 is None:
        status_callback("오류: 초기 이미지 로드 실패.")
        return
    images_loaded[0] = img0
    images_loaded[1] = img1

    kp0, des0 = extract_features(img0, status_callback); kp1, des1 = extract_features(img1, status_callback)
    if kp0 is None or des0 is None or kp1 is None or des1 is None:
        status_callback("오류: 초기 특징점 추출 실패.")
        return
    features[0] = (kp0, des0); features[1] = (kp1, des1)

    matches_01 = match_features(des0, des1, params.get("RATIO_TEST_THRESHOLD", 0.75), status_callback)
    if not matches_01 or len(matches_01) < 15: # E 추정 최소 매칭수 확인
        status_callback(f"오류: 초기 이미지 간 매칭 부족 ({len(matches_01)}).")
        return

    E_01, mask_e_01 = estimate_essential_matrix(kp0, kp1, matches_01, k_matrix, status_callback)
    if E_01 is None:
        status_callback("오류: 초기 E-Matrix 추정 실패.")
        return

    R_01, t_01, final_mask_01 = recover_pose_from_essential(E_01, kp0, kp1, matches_01, mask_e_01, k_matrix, status_callback)
    if R_01 is None or t_01 is None or final_mask_01 is None:
        status_callback("오류: 초기 Pose 복구 실패.")
        return
    poses[0] = (np.identity(3), np.zeros((3, 1))) # Image 0 포즈
    poses[1] = (R_01, t_01)                      # Image 1 포즈

    P0 = k_matrix @ np.hstack(poses[0])
    P1 = k_matrix @ np.hstack(poses[1])

    points_3d_init, success_match_indices_init = triangulate_points_incremental(kp0, kp1, matches_01, final_mask_01, P0, P1, status_callback)

    if points_3d_init is None or len(points_3d_init) == 0:
        status_callback("오류: 초기 삼각측량 실패.")
        return
    status_callback(f" 초기화: {len(points_3d_init)} 개의 3D 포인트 생성됨.")

    global_map = GlobalMap()
    for i, match_idx in enumerate(success_match_indices_init):
        pt_3d = points_3d_init[i]
        match = matches_01[match_idx]
        kp_idx_0 = match.queryIdx
        if kp_idx_0 < len(des0) and kp_idx_0 < len(kp0):
            desc_0 = des0[kp_idx_0]
            kp_0 = kp0[kp_idx_0]
            color_0 = get_color_from_image(img0, kp_0)
            global_map.add_point(pt_3d, desc_0, 0, kp_idx_0, color_0)
        # else: status_callback(f"Warning: 초기화 중 유효하지 않은 키포인트 인덱스 {kp_idx_0}")

    # --- 2. 점진적 단계 ---
    for i in range(2, num_images):
        status_callback(f"\n--- 점진적 단계: Image {i} 추가 ---")
        img_prev_idx = i - 1 # 이전 이미지 인덱스
        img_curr_idx = i   # 현재 이미지 인덱스

        # 현재 이미지 로드 및 특징점 추출
        img_curr_path = image_files[img_curr_idx]
        img_curr = cv2.imread(img_curr_path)
        if img_curr is None: status_callback(f" Image {i} 로드 실패, 건너<0xEB><0x8D>."); continue
        images_loaded[img_curr_idx] = img_curr

        kp_curr, des_curr = extract_features(img_curr, status_callback)
        if kp_curr is None or des_curr is None: status_callback(f" Image {i} 특징점 추출 실패, 건너<0xEB><0x8D>."); continue
        features[img_curr_idx] = (kp_curr, des_curr)

        # --- 2a. PnP로 현재 카메라 포즈 추정 ---
        status_callback(" PnP 단계: 현재 맵과 매칭 시도...")
        map_descriptors = global_map.get_descriptors_for_matching()

        if map_descriptors is None or len(map_descriptors) == 0:
            status_callback(" PnP 실패: 맵에 기술자 없음."); continue

        matches_2d_3d = match_features_2d_3d(des_curr, map_descriptors, bf_matcher, params.get("RATIO_TEST_THRESHOLD", 0.75), status_callback)

        pnp_result = estimate_pose_pnp(kp_curr, matches_2d_3d, global_map, k_matrix, params, status_callback)

        if pnp_result is None:
            status_callback(f" PnP 실패 또는 매칭 부족. Image {i} 건너<0xEB><0x8D>.")
            # 실패 시 현재 이미지 데이터 제거 (선택적)
            del images_loaded[i]; del features[i]
            continue

        R_curr, t_curr, pnp_inlier_match_indices = pnp_result
        poses[img_curr_idx] = (R_curr, t_curr) # 추정된 전역 포즈 저장

        # --- 2b. 새로운 포인트 삼각측량 ---
        status_callback(" 삼각측량 단계: 이전 이미지와 매칭하여 새 포인트 생성 시도...")
        # PnP 인라이어로 사용된 현재 이미지 키포인트 인덱스 집합 만들기
        pnp_used_kp_indices_curr = set()
        if len(pnp_inlier_match_indices) > 0:
             # pnp_inlier_match_indices는 matches_2d_3d 리스트 내의 인덱스
            for match_list_idx in pnp_inlier_match_indices:
                 if match_list_idx < len(matches_2d_3d):
                    kp_idx = matches_2d_3d[match_list_idx][0] # 현재 이미지의 kp 인덱스
                    pnp_used_kp_indices_curr.add(kp_idx)

        # PnP에 사용되지 않은 현재 키포인트 필터링
        kp_indices_curr_unused = [idx for idx in range(len(kp_curr)) if idx not in pnp_used_kp_indices_curr]
        if not kp_indices_curr_unused:
            status_callback(" 삼각측량 단계: PnP에서 모든 특징점이 사용되어 새 포인트 삼각측량 건너<0xEB><0x8D>.")
        else:
            des_curr_unused = des_curr[np.array(kp_indices_curr_unused)]

            # 이전 이미지의 특징점 가져오기
            if img_prev_idx not in features:
                status_callback(f"오류: 이전 이미지 {img_prev_idx} 특징점 정보 없음. 삼각측량 불가.")
            else:
                kp_prev, des_prev = features[img_prev_idx]
                if des_prev is None:
                    status_callback(f"경고: 이전 이미지 {img_prev_idx} 기술자 없음. 삼각측량 불가.")
                else:
                    # 사용 안 된 현재 특징점 <-> 이전 특징점 매칭
                    matches_new_2d2d = match_features(des_curr_unused, des_prev, params.get("RATIO_TEST_THRESHOLD", 0.75), status_callback)

                    # 매칭 인덱스를 원래 kp_curr, kp_prev 기준으로 변환
                    original_matches_for_tri = []
                    for m in matches_new_2d2d:
                        if m.queryIdx < len(kp_indices_curr_unused):
                           original_query_idx = kp_indices_curr_unused[m.queryIdx]
                           original_train_idx = m.trainIdx # kp_prev 기준 인덱스
                           original_matches_for_tri.append(cv2.DMatch(original_query_idx, original_train_idx, m.distance))
                        # else: status_callback(f"Warning: Invalid queryIdx {m.queryIdx} in unused features matching.")

                    if not original_matches_for_tri:
                         status_callback(" 삼각측량 단계: 새로운 2D-2D 매칭 없음.")
                    else:
                        # 삼각측량 수행
                        valid_mask_tri = np.ones(len(original_matches_for_tri), dtype=bool) # 마스크 필요시 구현
                        P_curr = k_matrix @ np.hstack(poses[img_curr_idx])
                        P_prev = k_matrix @ np.hstack(poses[img_prev_idx])

                        new_points_3d, success_indices_new = triangulate_points_incremental(
                            kp_curr, kp_prev, original_matches_for_tri, valid_mask_tri, P_curr, P_prev, status_callback
                        )

                        if new_points_3d is not None and len(new_points_3d) > 0:
                            status_callback(f" 삼각측량 성공: 새로운 3D 포인트 {len(new_points_3d)} 개 생성됨.")
                            # 새로 생성된 포인트를 GlobalMap에 추가
                            for k, original_match_list_idx in enumerate(success_indices_new):
                                if original_match_list_idx < len(original_matches_for_tri):
                                    pt_3d_new = new_points_3d[k]
                                    original_match = original_matches_for_tri[original_match_list_idx]
                                    kp_idx_curr = original_match.queryIdx # kp_curr 기준 인덱스

                                    if kp_idx_curr < len(des_curr) and kp_idx_curr < len(kp_curr):
                                        desc_curr_new = des_curr[kp_idx_curr]
                                        kp_curr_obj = kp_curr[kp_idx_curr]
                                        color_curr = get_color_from_image(img_curr, kp_curr_obj)
                                        global_map.add_point(pt_3d_new, desc_curr_new, img_curr_idx, kp_idx_curr, color_curr)
                                    # else: status_callback(f"Warning: Invalid keypoint index {kp_idx_curr} during map update from triangulation.")
                                # else: status_callback(f"Warning: Invalid match index {original_match_list_idx} during map update.")
                        else:
                            status_callback(" 삼각측량 단계: 유효한 새 3D 포인트 생성 실패.")

        # --- 2c. 루프 내 맵 클리닝 ---
        status_callback(f" 맵 클리닝 단계 (In-Loop SOR) 시작...")
        current_points = global_map.get_points_coords()
        if current_points is not None and len(current_points) > params.get("INLOOP_SOR_NEIGHBORS", 20): # 포인트 수가 이웃 수보다 많을 때만 의미 있음
            pcd_current_raw = o3d.geometry.PointCloud()
            pcd_current_raw.points = o3d.utility.Vector3dVector(current_points)
            points_before_clean = len(pcd_current_raw.points)
            # status_callback(f"  SOR 적용 전 포인트 수: {points_before_clean}")

            _, indices = pcd_current_raw.remove_statistical_outlier(
                nb_neighbors=params.get("INLOOP_SOR_NEIGHBORS", 20),
                std_ratio=params.get("INLOOP_SOR_STD_RATIO", 2.0)
            )

            if indices is not None and len(indices) < points_before_clean:
                status_callback(f"  SOR 적용 후 남은 포인트 수: {len(indices)} (제거: {points_before_clean - len(indices)})")
                inlier_indices = np.array(indices)

                # 맵 갱신
                status_callback("  GlobalMap 갱신 중...")
                current_descriptors = global_map.get_descriptors_for_matching() # None일 수 있음
                current_point_ids = np.array(global_map.point_id_from_descriptor_idx)

                # 갱신 가능한지 확인 (데이터 길이 일치 중요)
                can_update = (current_descriptors is not None and
                              len(current_descriptors) == points_before_clean and
                              len(current_point_ids) == points_before_clean)

                if can_update:
                    new_global_map = GlobalMap()
                    new_global_map.next_point_id = global_map.next_point_id

                    inlier_point_ids = current_point_ids[inlier_indices]
                    inlier_descriptors = current_descriptors[inlier_indices]

                    new_global_map.descriptors_list = list(inlier_descriptors)
                    new_global_map.point_id_from_descriptor_idx = list(inlier_point_ids)

                    valid_new_map = True
                    for pt_id in inlier_point_ids:
                        if pt_id in global_map.points:
                            new_global_map.points[pt_id] = global_map.points[pt_id] # 객체 참조 복사
                        else:
                            status_callback(f"치명적 오류: Inlier 포인트 ID {pt_id}를 원본 맵에서 찾을 수 없음. 맵 갱신 중단.")
                            valid_new_map = False; break

                    if valid_new_map:
                        global_map = new_global_map # 맵 교체
                        status_callback(f"  GlobalMap 갱신 완료. 현재 포인트 수: {len(global_map.points)}")
                    else:
                         status_callback("  GlobalMap 갱신 오류 발생. 이전 맵 상태 유지.")
                else:
                     status_callback("경고: 클리닝 중 데이터 배열 크기 불일치. 맵 갱신 건너<0xEB><0x8D>.")
            # else: status_callback("  제거된 이상치 없음 또는 SOR 실패.")
        else:
             status_callback("  포인트 수가 적어 In-Loop SOR 건너<0xEB><0x8D>.")
        status_callback(" 맵 클리닝 단계 완료.")

        # --- 2d. 중간 단계 결과 저장 ---
        status_callback(f" 중간 결과 저장 시도 (단계 {i})...")
        current_points_cleaned = global_map.get_points_coords()
        current_colors_bgr_cleaned = global_map.get_points_colors()
        if current_points_cleaned is not None and len(current_points_cleaned) > 0:
            step_filename = os.path.join(output_dir_incremental_steps, f"step_{i:04d}_map.ply")
            save_ply(step_filename, current_points_cleaned, current_colors_bgr_cleaned, status_callback)
        # else: status_callback(f" 단계 {i}: 저장할 포인트 없음.")

        # 메모리 관리 (선택적): 오래된 이미지/특징점 삭제
        # if i > 5: # 예: 5 프레임 이전 데이터 삭제
        #     if (i - 5) in images_loaded: del images_loaded[i - 5]
        #     if (i - 5) in features: del features[i - 5]


    # --- 3. 최종 결과 처리 및 저장 ---
    status_callback("\n--- 최종 결과 처리 ---")
    final_points = global_map.get_points_coords()
    final_colors_bgr = global_map.get_points_colors()

    if final_points is None or len(final_points) == 0:
        status_callback("오류: 최종 포인트 클라우드가 비어 있습니다.")
        return

    pcd_final_raw = o3d.geometry.PointCloud()
    pcd_final_raw.points = o3d.utility.Vector3dVector(final_points)
    # 색상 설정 (save_ply에서 처리하므로 여기서는 생략 가능, 단 최종 SOR 전 색상 확인용)
    if final_colors_bgr is not None and final_colors_bgr.shape[0] == len(final_points) and final_colors_bgr.dtype == np.uint8:
        colors_rgb_final = final_colors_bgr[:, ::-1]
        colors_norm_final = np.clip(colors_rgb_final.astype(np.float64) / 255.0, 0.0, 1.0)
        pcd_final_raw.colors = o3d.utility.Vector3dVector(colors_norm_final)
        status_callback("최종 포인트 클라우드 생성됨 (색상 포함).")
    else:
         status_callback("최종 포인트 클라우드 생성됨 (색상 없음 또는 오류).")
         pcd_final_raw.paint_uniform_color([0.5, 0.5, 0.5])


    # 최종 이상치 제거
    status_callback(f"\n--- 최종 이상치 제거 시작 (nb_neighbors={params.get('FINAL_SOR_NEIGHBORS', 20)}, std_ratio={params.get('FINAL_SOR_STD_RATIO', 1.5)}) ---")
    points_before_final_clean = len(pcd_final_raw.points)
    status_callback(f"최종 처리 전 포인트 수: {points_before_final_clean}")

    pcd_final_filtered = None
    if points_before_final_clean > params.get('FINAL_SOR_NEIGHBORS', 20):
        pcd_final_filtered, _ = pcd_final_raw.remove_statistical_outlier(
            nb_neighbors=params.get("FINAL_SOR_NEIGHBORS", 20),
            std_ratio=params.get("FINAL_SOR_STD_RATIO", 1.5)
        )
        status_callback(f"최종 처리 후 포인트 수: {len(pcd_final_filtered.points) if pcd_final_filtered else 0}")
    else:
        status_callback("포인트 수가 적어 최종 SOR 건너<0xEB><0x8D>.")
        pcd_final_filtered = pcd_final_raw # 필터링 안 함


    if pcd_final_filtered is None or not pcd_final_filtered.has_points():
        status_callback("경고: 최종 이상치 제거 실패 또는 모든 포인트 제거됨. 필터링 전 맵(마지막 단계 맵)을 저장합니다.")
        pcd_final_to_save = pcd_final_raw
    else:
        pcd_final_to_save = pcd_final_filtered

    # 최종 저장할 데이터 추출 (NumPy 배열로)
    final_points_to_save = np.asarray(pcd_final_to_save.points)
    final_colors_to_save = None
    if pcd_final_to_save.has_colors():
         colors_rgb_norm_final = np.asarray(pcd_final_to_save.colors)
         # 범위 재확인 및 변환
         if np.all(colors_rgb_norm_final >= 0) and np.all(colors_rgb_norm_final <= 1.0):
             colors_rgb_uint8_final = (colors_rgb_norm_final * 255).astype(np.uint8)
             final_colors_to_save = colors_rgb_uint8_final[:, ::-1] # RGB -> BGR
         else: status_callback("경고: 저장 전 최종 색상 값 범위 오류.")

    # 최종 결과 저장
    final_output_path = output_ply_incremental # 설정된 최종 파일 이름 사용
    save_ply(final_output_path, final_points_to_save, final_colors_to_save, status_callback)

    status_callback("\n--- 점진적 재구성 완료 ---")


# --- GUI 관련 클래스 및 함수 ---
class ToolTip:
    """간단한 툴팁 클래스"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = tk.Label(self.tooltip, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         wraplength=300) # wrap text if long
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
        self.tooltip = None

class SfMGUI:
    def __init__(self, master):
        self.master = master
        master.title("점진적 SfM 재구성")
        master.geometry("750x700") # 창 크기 증가

        self.params = DEFAULT_PARAMS.copy() # 현재 파라미터 저장
        self.param_entries = {} # 파라미터 Entry 위젯 저장

        # --- 프레임 설정 ---
        control_frame = ttk.Frame(master, padding="10")
        control_frame.pack(fill=tk.X, side=tk.TOP)

        param_frame = ttk.LabelFrame(master, text="SfM 파라미터", padding="10")
        param_frame.pack(fill=tk.X, side=tk.TOP, padx=10, pady=5)

        action_frame = ttk.Frame(master, padding="10")
        action_frame.pack(fill=tk.X, side=tk.TOP)

        log_frame = ttk.LabelFrame(master, text="로그 및 상태", padding="10")
        log_frame.pack(expand=True, fill='both', side=tk.TOP, padx=10, pady=5)

        # --- 제어 프레임 (폴더 선택) ---
        ttk.Label(control_frame, text="데이터 폴더:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.data_dir_entry = ttk.Entry(control_frame, width=60, state='readonly')
        self.data_dir_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        self.browse_button = ttk.Button(control_frame, text="찾아보기...", command=self.browse_directory)
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)
        control_frame.columnconfigure(1, weight=1) # Entry가 늘어나도록

        # --- 파라미터 프레임 ---
        # 파라미터 레이아웃 (Grid 사용)
        row_idx = 0
        col_idx = 0
        max_cols = 2 # 한 행에 표시할 파라미터 그룹 수

        param_defs = [
            # (Key, Label Text, Description, Default Value)
            ("RATIO_TEST_THRESHOLD", "Ratio Test 임계값:", "특징점 매칭 시 거리 비율 임계값 (0.0 ~ 1.0)", DEFAULT_PARAMS["RATIO_TEST_THRESHOLD"]),
            ("MIN_MATCHES_PNP", "PnP 최소 매칭 수:", "PnP 포즈 추정을 위한 최소 2D-3D 매칭 수 (정수)", DEFAULT_PARAMS["MIN_MATCHES_PNP"]),
            ("RANSAC_ITERATIONS_PNP", "PnP RANSAC 반복:", "PnP RANSAC 최대 반복 횟수 (정수)", DEFAULT_PARAMS["RANSAC_ITERATIONS_PNP"]),
            ("RANSAC_REPROJECTION_ERROR_PNP", "PnP 최대 재투영 오차:", "PnP RANSAC 재투영 오차 임계값 (픽셀)", DEFAULT_PARAMS["RANSAC_REPROJECTION_ERROR_PNP"]),
            ("RANSAC_CONFIDENCE_PNP", "PnP RANSAC 신뢰도:", "PnP RANSAC 신뢰도 수준 (0.0 ~ 1.0)", DEFAULT_PARAMS["RANSAC_CONFIDENCE_PNP"]),
            ("INLOOP_SOR_NEIGHBORS", "루프 내 SOR 이웃 수:", "각 단계 맵 클리닝 시 사용할 이웃 수 (정수)", DEFAULT_PARAMS["INLOOP_SOR_NEIGHBORS"]),
            ("INLOOP_SOR_STD_RATIO", "루프 내 SOR 표준편차:", "각 단계 맵 클리닝 시 표준편차 비율 (클수록 덜 제거)", DEFAULT_PARAMS["INLOOP_SOR_STD_RATIO"]),
            ("FINAL_SOR_NEIGHBORS", "최종 SOR 이웃 수:", "최종 맵 클리닝 시 사용할 이웃 수 (정수)", DEFAULT_PARAMS["FINAL_SOR_NEIGHBORS"]),
            ("FINAL_SOR_STD_RATIO", "최종 SOR 표준편차:", "최종 맵 클리닝 시 표준편차 비율 (클수록 덜 제거)", DEFAULT_PARAMS["FINAL_SOR_STD_RATIO"]),
        ]

        for key, label_text, desc, default_value in param_defs:
            # 파라미터 그룹 프레임 (Label + Entry)
            p_group = ttk.Frame(param_frame)
            p_group.grid(row=row_idx, column=col_idx, sticky=tk.W, padx=10, pady=3)

            label = ttk.Label(p_group, text=label_text, width=25, anchor='w') # 너비 고정
            label.pack(side=tk.LEFT)
            entry = ttk.Entry(p_group, width=10)
            entry.pack(side=tk.LEFT, padx=5)
            entry.insert(0, str(default_value))
            self.param_entries[key] = entry # 엔트리 위젯 저장
            ToolTip(label, desc) # 레이블에 툴팁 추가
            ToolTip(entry, desc) # 엔트리에도 툴팁 추가

            # 다음 위치 계산
            col_idx += 1
            if col_idx >= max_cols:
                col_idx = 0
                row_idx += 1

        # --- 실행 프레임 ---
        self.start_button = ttk.Button(action_frame, text="재구성 시작", command=self.start_reconstruction_thread)
        self.start_button.pack(pady=10)

        # --- 로그 프레임 ---
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=15, state='disabled')
        self.log_text.pack(expand=True, fill='both')

    def browse_directory(self):
        """데이터 폴더 선택 대화상자 열기"""
        directory = filedialog.askdirectory(title="데이터 폴더 선택 (Images + K.txt)")
        if directory:
            self.data_dir_entry.config(state='normal')
            self.data_dir_entry.delete(0, tk.END)
            self.data_dir_entry.insert(0, directory)
            self.data_dir_entry.config(state='readonly')
            self.update_log(f"데이터 폴더 선택됨: {directory}")

    def update_log(self, message):
        """로그 텍스트 영역에 메시지 추가 (GUI 스레드 안전)"""
        if self.master.winfo_exists(): # 창이 존재하는지 확인
            self.log_text.config(state='normal')
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END) # 마지막 줄로 스크롤
            self.log_text.config(state='disabled')
            self.master.update_idletasks() # GUI 업데이트

    def update_log_safe(self, message):
         """다른 스레드에서 로그 업데이트를 안전하게 호출"""
         self.master.after(0, self.update_log, message)

    def validate_and_get_params(self) -> Optional[Dict]:
        """GUI에서 파라미터 값을 읽고 유효성 검사 후 딕셔너리로 반환"""
        params = {}
        try:
            for key, entry in self.param_entries.items():
                value_str = entry.get().strip()
                if not value_str: # 빈 값 허용 안 함
                    raise ValueError(f"'{key}' 파라미터 값이 비어 있습니다.")

                # 타입에 따라 변환 시도
                if key in ["RATIO_TEST_THRESHOLD", "INLOOP_SOR_STD_RATIO", "FINAL_SOR_STD_RATIO", "RANSAC_REPROJECTION_ERROR_PNP", "RANSAC_CONFIDENCE_PNP"]:
                    value = float(value_str)
                    if key == "RATIO_TEST_THRESHOLD" and not (0.0 < value <= 1.0):
                         raise ValueError(f"'{key}' 값은 0.0 초과 1.0 이하여야 합니다.")
                    if key == "RANSAC_CONFIDENCE_PNP" and not (0.0 < value <= 1.0):
                         raise ValueError(f"'{key}' 값은 0.0 초과 1.0 이하여야 합니다.")
                    if "STD_RATIO" in key and value <= 0:
                         raise ValueError(f"'{key}' 값은 0보다 커야 합니다.")
                    if "REPROJECTION_ERROR" in key and value <= 0:
                         raise ValueError(f"'{key}' 값은 0보다 커야 합니다.")
                else: # 정수형 파라미터
                    value = int(value_str)
                    if ("NEIGHBORS" in key or "MIN_MATCHES" in key or "ITERATIONS" in key) and value <= 0:
                         raise ValueError(f"'{key}' 값은 0보다 큰 정수여야 합니다.")
                params[key] = value
            return params
        except ValueError as e:
            messagebox.showerror("파라미터 오류", f"파라미터 값에 오류가 있습니다:\n{e}")
            return None

    def enable_controls(self, enable: bool):
        """컨트롤 활성화/비활성화 (재구성 중/후)"""
        state = 'normal' if enable else 'disabled'
        self.start_button.config(state=state)
        self.browse_button.config(state=state)
        for entry in self.param_entries.values():
            entry.config(state='normal' if enable else 'disabled') # 파라미터 엔트리도 조절
        # 데이터 디렉토리 엔트리는 항상 readonly 유지하므로 변경 안 함

    def run_reconstruction_in_thread(self, directory, params_dict):
        """실제 재구성 함수를 스레드에서 실행하는 래퍼 함수"""
        try:
            # 여기에 main_incremental_reconstruction 호출
            main_incremental_reconstruction(directory, params_dict, self.update_log_safe)
            # 성공 메시지 추가
            self.update_log_safe("\n*** 재구성 프로세스 완료 ***")

        except Exception as e:
            # 스레드 내에서 발생한 예상치 못한 오류 처리
            import traceback
            error_details = traceback.format_exc()
            self.update_log_safe(f"\n*** 재구성 중 치명적 오류 발생 ***\n{e}\n상세 정보:\n{error_details}")
            messagebox.showerror("실행 오류", f"재구성 중 오류가 발생했습니다:\n{e}")
        finally:
            # 완료 또는 오류 발생 시 컨트롤 다시 활성화 (GUI 스레드에서)
            self.master.after(0, self.enable_controls, True)


    def start_reconstruction_thread(self):
        """재구성 시작 버튼 클릭 시 호출"""
        data_directory = self.data_dir_entry.get()
        if not data_directory or not os.path.isdir(data_directory):
            messagebox.showerror("오류", "유효한 데이터 폴더를 선택해주세요.")
            return

        # 파라미터 유효성 검사 및 가져오기
        current_params = self.validate_and_get_params()
        if current_params is None:
            return # 유효성 검사 실패

        # 로그 지우기
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')
        self.update_log("재구성 프로세스를 시작합니다...")
        self.update_log(f"데이터 폴더: {data_directory}")
        self.update_log(f"사용 파라미터: {current_params}")


        # 컨트롤 비활성화
        self.enable_controls(False)

        # 스레드 생성 및 시작
        # daemon=True: 메인 창 종료 시 스레드도 강제 종료
        thread = threading.Thread(target=self.run_reconstruction_in_thread,
                                  args=(data_directory, current_params),
                                  daemon=True)
        thread.start()


# --- 메인 실행 부분 ---
if __name__ == "__main__":
    root = tk.Tk()
    app = SfMGUI(root)
    root.mainloop()