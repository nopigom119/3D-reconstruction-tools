import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext

# --- 코어 로직 함수 ---
def estimate_k_matrix(image_folder, focal_length_heuristic_value, status_callback):
    """
    주어진 폴더의 첫 이미지를 기반으로 K 행렬을 추정합니다.

    Args:
        image_folder (str): 이미지가 있는 폴더 경로.
        focal_length_heuristic_value (str or float): 초점 거리 추정 방식 ('max' 또는 숫자 값).
        status_callback (function): GUI 상태 업데이트를 위한 콜백 함수.

    Returns:
        tuple: (추정된 K 행렬 (np.array) 또는 None, 이미지 정보 딕셔너리 또는 None)
               오류 발생 시 None, None 반환.
    """
    status_callback(f"--- '{image_folder}' 폴더의 이미지 기반 K 행렬 추정 시작 ---")

    # 1. 이미지 파일 목록 확인
    try:
        image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if not image_files:
            raise FileNotFoundError(f"'{image_folder}' 폴더에 이미지 파일(.jpg, .jpeg, .png)이 없습니다.")
        status_callback(f"  총 {len(image_files)}개의 이미지 파일을 찾았습니다.")
        status_callback(f"  첫 번째 이미지 '{image_files[0]}'를 사용하여 이미지 크기를 결정합니다.")
    except FileNotFoundError as e:
        status_callback(f"  오류: {e}")
        return None, None
    except Exception as e:
        status_callback(f"  이미지 파일 목록 로드 중 오류 발생: {e}")
        return None, None

    # 2. 첫 번째 이미지 로드 및 크기 확인
    first_image_path = os.path.join(image_folder, image_files[0])
    first_img = cv2.imread(first_image_path)
    if first_img is None:
        status_callback(f"  오류: 첫 번째 이미지({first_image_path})를 로드할 수 없습니다.")
        return None, None

    height, width = first_img.shape[:2]
    status_callback(f"  이미지 크기: Width={width}, Height={height}")

    # 3. 가정에 기반한 K 행렬 파라미터 추정
    cx = width / 2.0
    cy = height / 2.0

    f = 0.0 # 초기화
    if isinstance(focal_length_heuristic_value, str) and focal_length_heuristic_value.lower() == "max":
        f = float(max(width, height))
        status_callback(f"  초점 거리 추정 (max(width, height)): f = {f:.4f}")
    elif isinstance(focal_length_heuristic_value, (int, float)):
        f = float(focal_length_heuristic_value)
        status_callback(f"  초점 거리 직접 지정: f = {f:.4f}")
    else: # 이 경우는 GUI 검증에서 걸러지지만 안전을 위해 추가
        status_callback(f"  경고: 초점 거리 설정 오류. 기본값 max(width, height) 사용.")
        f = float(max(width, height))

    status_callback(f"  추정된 주점: cx = {cx:.4f}, cy = {cy:.4f}")
    status_callback(f"  추정된 초점 거리: fx = fy = {f:.4f}")
    status_callback(f"  (가정: skew = 0)")

    # 4. 추정된 K 행렬 생성
    K_est = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    status_callback("\n  추정된 K 행렬 (K_est):")
    status_callback(str(K_est)) # 콜백은 문자열을 기대하므로 변환

    image_info = {
        "filename": image_files[0],
        "width": width,
        "height": height,
        "cx": cx,
        "cy": cy,
        "f": f
    }
    return K_est, image_info

def save_k_matrix(K_matrix, dest_folder, filename, status_callback):
    """추정된 K 행렬을 지정된 파일에 저장합니다."""
    if K_matrix is None or not dest_folder or not filename:
        status_callback("오류: K 행렬 데이터 또는 저장 경로/파일 이름이 유효하지 않아 저장할 수 없습니다.")
        return False

    output_path = os.path.join(dest_folder, filename)
    try:
        # 대상 폴더 생성 (없는 경우) - 안전하게 저장하기 위함
        os.makedirs(dest_folder, exist_ok=True)
        np.savetxt(output_path, K_matrix, fmt='%.8f', delimiter=' ')
        status_callback(f"\n성공적으로 추정된 K 행렬을 '{output_path}' 파일로 저장했습니다.")
        return True
    except Exception as e:
        status_callback(f"\n오류: K 행렬을 '{output_path}' 파일로 저장하는 중 문제가 발생했습니다: {e}")
        return False

# --- GUI 관련 함수 ---
def browse_folder(entry_widget, title="폴더 선택"):
    """폴더 선택 대화상자를 열고 선택된 경로를 지정된 엔트리에 표시"""
    folder_selected = filedialog.askdirectory(title=title)
    if folder_selected:
        entry_widget.config(state='normal')
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, folder_selected)
        entry_widget.config(state='readonly')

def update_status(message):
    """상태 메시지 텍스트 영역에 메시지 추가"""
    status_text.config(state='normal')
    status_text.insert(tk.END, message + "\n")
    status_text.see(tk.END) # 마지막 줄로 스크롤
    status_text.config(state='disabled')
    root.update_idletasks() # GUI 업데이트 강제

def toggle_focal_entry():
    """초점 거리 직접 입력 엔트리 활성화/비활성화"""
    if focal_choice_var.get() == "direct":
        focal_value_entry.config(state='normal')
    else:
        focal_value_entry.config(state='disabled')

def start_estimation_process():
    """GUI에서 설정값을 가져와 K 행렬 추정 및 저장 프로세스를 시작"""
    # 이전 결과 지우기
    status_text.config(state='normal')
    status_text.delete(1.0, tk.END)
    status_text.config(state='disabled')

    # 입력값 가져오기
    source_folder = source_folder_entry.get()
    dest_folder = dest_folder_entry.get()
    output_filename = filename_entry.get().strip()
    focal_choice = focal_choice_var.get()

    # 입력값 유효성 검사
    if not source_folder:
        messagebox.showerror("입력 오류", "이미지가 있는 폴더를 선택해주세요.")
        return
    if not dest_folder:
        messagebox.showerror("입력 오류", "K 행렬을 저장할 폴더를 선택해주세요.")
        return
    if not output_filename:
        messagebox.showerror("입력 오류", "저장할 파일 이름을 입력해주세요.")
        return

    focal_heuristic_value = None
    if focal_choice == "max":
        focal_heuristic_value = "max"
    elif focal_choice == "direct":
        try:
            focal_heuristic_value = float(focal_value_entry.get())
            if focal_heuristic_value <= 0:
                 raise ValueError("초점 거리는 0보다 커야 합니다.")
        except ValueError as e:
            messagebox.showerror("입력 오류", f"초점 거리 값 오류: 유효한 숫자를 입력해주세요.\n({e})")
            return
    else: # Should not happen
        messagebox.showerror("오류", "알 수 없는 초점 거리 선택입니다.")
        return

    # 코어 로직 실행
    K_estimated, _ = estimate_k_matrix(source_folder, focal_heuristic_value, update_status)

    # 결과 저장
    if K_estimated is not None:
        save_k_matrix(K_estimated, dest_folder, output_filename, update_status)
    else:
        update_status("K 행렬 추정에 실패하여 저장할 수 없습니다.")


# --- GUI 설정 ---
root = tk.Tk()
root.title("이미지 기반 K 행렬 추정기")
root.geometry("600x500") # 창 크기 조절

main_frame = ttk.Frame(root, padding="10")
main_frame.pack(expand=True, fill='both')

# --- 입력 섹션 ---
input_frame = ttk.LabelFrame(main_frame, text="설정", padding="10")
input_frame.pack(fill='x', pady=5)

# 1. 소스 폴더 선택
source_frame = ttk.Frame(input_frame)
source_frame.pack(fill='x', pady=2)
ttk.Label(source_frame, text="이미지 폴더:", width=15, anchor='w').pack(side=tk.LEFT)
source_folder_entry = ttk.Entry(source_frame, state='readonly', width=50)
source_folder_entry.pack(side=tk.LEFT, expand=True, fill='x', padx=5)
source_browse_button = ttk.Button(source_frame, text="찾아보기...",
                                  command=lambda: browse_folder(source_folder_entry, title="이미지 폴더 선택"))
source_browse_button.pack(side=tk.LEFT)

# 2. 저장 폴더 선택
dest_frame = ttk.Frame(input_frame)
dest_frame.pack(fill='x', pady=2)
ttk.Label(dest_frame, text="저장 폴더:", width=15, anchor='w').pack(side=tk.LEFT)
dest_folder_entry = ttk.Entry(dest_frame, state='readonly', width=50)
dest_folder_entry.pack(side=tk.LEFT, expand=True, fill='x', padx=5)
dest_browse_button = ttk.Button(dest_frame, text="찾아보기...",
                                command=lambda: browse_folder(dest_folder_entry, title="K 행렬 저장 폴더 선택"))
dest_browse_button.pack(side=tk.LEFT)

# 3. 저장 파일 이름
filename_frame = ttk.Frame(input_frame)
filename_frame.pack(fill='x', pady=2)
ttk.Label(filename_frame, text="저장 파일 이름:", width=15, anchor='w').pack(side=tk.LEFT)
filename_entry = ttk.Entry(filename_frame, width=30)
filename_entry.pack(side=tk.LEFT, padx=5)
filename_entry.insert(0, "K_estimated.txt") # 기본 파일 이름

# 4. 초점 거리 추정 방식
focal_frame = ttk.Frame(input_frame)
focal_frame.pack(fill='x', pady=5)
ttk.Label(focal_frame, text="초점 거리 방식:", width=15, anchor='w').pack(side=tk.LEFT)

focal_choice_var = tk.StringVar(value="max") # 기본값 'max'

max_radio = ttk.Radiobutton(focal_frame, text="max(너비, 높이)", variable=focal_choice_var, value="max", command=toggle_focal_entry)
max_radio.pack(side=tk.LEFT, padx=5)

direct_radio = ttk.Radiobutton(focal_frame, text="직접 입력:", variable=focal_choice_var, value="direct", command=toggle_focal_entry)
direct_radio.pack(side=tk.LEFT, padx=5)

focal_value_entry = ttk.Entry(focal_frame, width=10, state='disabled') # 기본 비활성화
focal_value_entry.pack(side=tk.LEFT)

# --- 실행 버튼 ---
run_button = ttk.Button(main_frame, text="K 행렬 추정 및 저장", command=start_estimation_process)
run_button.pack(pady=10)

# --- 결과 및 상태 메시지 출력 영역 ---
status_frame = ttk.LabelFrame(main_frame, text="결과 및 상태", padding="10")
status_frame.pack(expand=True, fill='both', pady=5)

status_text = scrolledtext.ScrolledText(status_frame, wrap=tk.WORD, height=15, state='disabled')
status_text.pack(expand=True, fill='both')

# --- 메인 루프 시작 ---
root.mainloop()