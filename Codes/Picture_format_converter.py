import os
import glob
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image
import threading

# --- 기존 로직 수정 ---
def rename_and_convert_images(source_folder_path, dest_folder_path, output_format, output_extension, status_callback):
    """
    주어진 소스 폴더의 모든 이미지를 사용자가 선택한 형식과 확장자로 변환하고
    순서대로 이름을 변경하여 대상 폴더에 저장합니다.

    Args:
        source_folder_path (str): 원본 이미지가 있는 폴더의 경로.
        dest_folder_path (str): 변환된 이미지를 저장할 폴더의 경로.
        output_format (str): PIL(Pillow)에서 인식하는 저장 형식 (예: 'JPEG', 'PNG').
        output_extension (str): 사용자가 지정한 파일 확장자 (예: '.jpg', '.png').
        status_callback (function): GUI 상태 업데이트를 위한 콜백 함수.
    """
    # 대상 폴더가 지정되지 않았으면 소스 폴더로 설정
    if not dest_folder_path:
        dest_folder_path = source_folder_path
        status_callback("정보: 저장 폴더가 지정되지 않아 원본 폴더에 저장합니다.")

    # 입력값 유효성 검사 (확장자에 . 이 포함되어 있는지)
    if not output_extension.startswith('.'):
        status_callback(f"오류: 확장자는 '.'으로 시작해야 합니다 (예: .jpg). '{output_extension}'은(는) 잘못된 형식입니다.")
        return False # 처리 중단

    try:
        # 대상 폴더 생성 (없는 경우)
        os.makedirs(dest_folder_path, exist_ok=True)
        status_callback(f"정보: 이미지를 '{dest_folder_path}' 폴더에 저장합니다.")

        # 소스 폴더 경로에 있는 모든 지원 이미지 파일을 찾습니다.
        supported_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
        image_files = []
        for ext in supported_extensions:
            image_files.extend(glob.glob(os.path.join(source_folder_path, f'*{ext}')))
            image_files.extend(glob.glob(os.path.join(source_folder_path, f'*{ext.upper()}')))

        # 중복 제거 및 정렬
        image_files = sorted(list(set(image_files)))

        if not image_files:
            status_callback("선택한 원본 폴더에 지원하는 이미지 파일이 없습니다.")
            return False # 처리 중단

        total_files = len(image_files)
        status_callback(f"총 {total_files}개의 이미지 파일을 원본 폴더에서 찾았습니다. 변환을 시작합니다...")

        # 이미지 파일들을 순회하면서 변환하고 이름을 변경합니다.
        processed_count = 0
        source_abs = os.path.abspath(source_folder_path)
        dest_abs = os.path.abspath(dest_folder_path)

        for index, image_file in enumerate(image_files):
            try:
                # 이미지를 불러옵니다.
                img = Image.open(image_file)
                img_format = img.format # 원본 포맷 확인 (디버깅/정보용)

                # 새 파일 이름을 대상 폴더 경로 기준으로 생성합니다. (순번 + 사용자 지정 확장자)
                new_file_name = os.path.join(dest_folder_path, f'{index + 1:04d}{output_extension}')

                # 이미지를 사용자가 선택한 형식으로 저장합니다.
                save_img = img
                if output_format == 'JPEG' and img.mode in ('RGBA', 'P'): # PNG/GIF -> JPEG (투명도 처리)
                    if img.mode == 'RGBA':
                        status_callback(f"정보: '{os.path.basename(image_file)}' 알파 채널 포함. 흰색 배경으로 변환.")
                        save_img = Image.new("RGB", img.size, (255, 255, 255))
                        save_img.paste(img, mask=img.split()[3]) # 알파 채널을 마스크로 사용
                    elif img.mode == 'P': # 팔레트 모드 처리
                         save_img = img.convert('RGB')
                elif img.mode == 'P' and output_format in ['PNG', 'TIFF', 'WEBP']: # 팔레트 -> RGBA
                     save_img = img.convert('RGBA')
                elif img.mode != 'RGB' and output_format == 'JPEG': # 기타 모드 -> RGB
                    save_img = img.convert('RGB')


                save_img.save(new_file_name, output_format)

                processed_count += 1
                status_callback(f'처리 중 ({processed_count}/{total_files}): "{os.path.basename(image_file)}" -> "{os.path.basename(new_file_name)}" ({output_format})')

                # 기존 파일 삭제 로직 수정:
                # 소스 폴더와 대상 폴더가 동일하고, 변환 전후 파일 경로가 다른 경우에만 원본 삭제 시도
                if source_abs == dest_abs and os.path.abspath(image_file) != os.path.abspath(new_file_name):
                    try:
                        os.remove(image_file)
                        # status_callback(f'정보: 원본 파일 "{os.path.basename(image_file)}" 삭제됨 (동일 폴더 저장).')
                    except Exception as remove_err:
                        status_callback(f'경고: 원본 파일 "{os.path.basename(image_file)}" 삭제 실패: {remove_err}')

            except Exception as e:
                status_callback(f'오류: "{os.path.basename(image_file)}" 처리 중 오류 발생: {e}')
                # 오류 발생 시 해당 파일은 건너뛰고 계속 진행

        status_callback(f"총 {processed_count}개의 이미지 변환 및 이름 변경 완료! ('{dest_folder_path}'에 저장)")
        return True # 성공적으로 완료

    except Exception as e:
        status_callback(f"전체 작업 중 예상치 못한 오류 발생: {e}")
        return False # 처리 중단

# --- GUI 관련 함수 ---
def browse_folder(entry_widget, title="폴더 선택"):
    """폴더 선택 대화상자를 열고 선택된 경로를 지정된 엔트리에 표시"""
    folder_selected = filedialog.askdirectory(title=title)
    if folder_selected:
        entry_widget.config(state='normal')
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, folder_selected)
        entry_widget.config(state='readonly')
        if entry_widget == source_folder_path_entry:
            update_status("원본 폴더가 선택되었습니다.")
        elif entry_widget == dest_folder_path_entry:
            update_status("저장 폴더가 선택되었습니다.")

def update_status(message):
    """상태 메시지 라벨 업데이트 (GUI 스레드에서 안전하게 호출)"""
    status_label.config(text=message)

def update_extension_suggestion(*args):
    """콤보박스 선택에 따라 추천 확장자 업데이트"""
    selected_format = format_combobox.get()
    suggestions = {
        'JPEG': '.jpg',
        'PNG': '.png',
        'BMP': '.bmp',
        'GIF': '.gif',
        'TIFF': '.tif',
        'WEBP': '.webp'
    }
    suggested_ext = suggestions.get(selected_format, '')
    extension_entry.delete(0, tk.END)
    extension_entry.insert(0, suggested_ext)


def run_conversion_thread():
    """별도의 스레드에서 이미지 처리 함수를 실행"""
    source_folder_path = source_folder_path_entry.get()
    dest_folder_path = dest_folder_path_entry.get() # 저장 폴더 경로 가져오기
    output_format = format_combobox.get()
    output_extension = extension_entry.get().strip()

    if not source_folder_path:
        messagebox.showerror("오류", "원본 이미지가 있는 폴더를 선택해주세요.")
        return
    # 저장 폴더는 선택 사항이므로 필수로 체크하지 않음 (선택 안 하면 원본 폴더에 저장됨)
    if not output_format:
        messagebox.showerror("오류", "출력할 이미지 형식을 선택해주세요.")
        return
    if not output_extension:
        messagebox.showerror("오류", "사용할 파일 확장자를 입력해주세요 (예: .jpg).")
        return
    if not output_extension.startswith('.'):
         messagebox.showerror("오류", "확장자는 '.'으로 시작해야 합니다 (예: .jpg).")
         return

    # 시작 버튼 비활성화
    start_button.config(state='disabled')
    update_status("작업을 준비 중입니다...")

    # 스레드 생성 및 시작
    thread = threading.Thread(target=process_images_in_thread,
                              args=(source_folder_path, dest_folder_path, output_format, output_extension),
                              daemon=True) # 메인 윈도우 종료 시 스레드도 함께 종료
    thread.start()

def process_images_in_thread(source_folder_path, dest_folder_path, output_format, output_extension):
    """스레드에서 실행될 함수: 이미지 처리 및 GUI 업데이트 호출"""
    try:
        # rename_and_convert_images 함수 실행, 상태 업데이트는 update_status_safe 호출
        success = rename_and_convert_images(source_folder_path, dest_folder_path, output_format, output_extension, update_status_safe)
        if success:
            messagebox.showinfo("완료", "이미지 변환 및 이름 변경 작업이 완료되었습니다.")
        else:
            # 오류 메시지는 rename_and_convert_images 내부에서 콜백으로 전달하므로, 여기서는 추가 메시지 X
             messagebox.showwarning("작업 완료 (일부 오류)", "작업이 완료되었지만 일부 파일 처리 중 오류가 발생했을 수 있습니다. 상태 메시지를 확인하세요.")
    except Exception as e:
        update_status_safe(f"스레드 오류: {e}")
        messagebox.showerror("치명적 오류", f"이미지 처리 스레드에서 예상치 못한 오류 발생: {e}")
    finally:
        # 작업 완료 후 시작 버튼 다시 활성화 (GUI 스레드에서 안전하게)
        root.after(0, enable_start_button)

def update_status_safe(message):
    """다른 스레드에서 GUI 상태 라벨을 안전하게 업데이트하기 위해 root.after 사용"""
    root.after(0, update_status, message)

def enable_start_button():
    """시작 버튼을 다시 활성화"""
    start_button.config(state='normal')


# --- GUI 설정 ---
root = tk.Tk()
root.title("이미지 일괄 변환 및 이름 변경")
root.geometry("550x300") # 창 크기 조절 (세로 약간 늘림)

# 프레임 생성 (위젯 그룹화)
frame = ttk.Frame(root, padding="10")
frame.pack(expand=True, fill='both')

# 1. 원본 폴더 선택
source_folder_frame = ttk.Frame(frame)
source_folder_frame.pack(fill='x', pady=3)
ttk.Label(source_folder_frame, text="원본 폴더:").pack(side=tk.LEFT, padx=(0, 5), anchor='w')
source_folder_path_entry = ttk.Entry(source_folder_frame, state='readonly', width=50)
source_folder_path_entry.pack(side=tk.LEFT, expand=True, fill='x')
# browse_folder 함수에 어떤 엔트리를 업데이트할지 알려주기 위해 lambda 사용
source_browse_button = ttk.Button(source_folder_frame, text="찾아보기...",
                                  command=lambda: browse_folder(source_folder_path_entry, title="원본 폴더 선택"))
source_browse_button.pack(side=tk.LEFT, padx=(5, 0))

# 2. 저장 폴더 선택 (신규 추가)
dest_folder_frame = ttk.Frame(frame)
dest_folder_frame.pack(fill='x', pady=3)
ttk.Label(dest_folder_frame, text="저장 폴더:").pack(side=tk.LEFT, padx=(0, 5), anchor='w')
dest_folder_path_entry = ttk.Entry(dest_folder_frame, state='readonly', width=50)
dest_folder_path_entry.pack(side=tk.LEFT, expand=True, fill='x')
dest_browse_button = ttk.Button(dest_folder_frame, text="찾아보기...",
                                command=lambda: browse_folder(dest_folder_path_entry, title="저장 폴더 선택"))
dest_browse_button.pack(side=tk.LEFT, padx=(5, 0))
ttk.Label(frame, text="(비워두면 원본 폴더에 저장됩니다)", foreground="gray").pack(anchor='w', padx=60) # 안내 문구


# 3. 출력 형식 선택
format_frame = ttk.Frame(frame)
format_frame.pack(fill='x', pady=5, padx=55) # 라벨 너비 고려하여 왼쪽 패딩
ttk.Label(format_frame, text="출력 형식:").pack(side=tk.LEFT, padx=(0, 5))
# 사용 가능한 형식 (Pillow이 지원하고 일반적인 형식 위주)
supported_formats = ('JPEG', 'PNG', 'BMP', 'GIF', 'TIFF', 'WEBP')
format_var = tk.StringVar()
format_combobox = ttk.Combobox(format_frame, textvariable=format_var, values=supported_formats, state='readonly', width=10)
format_combobox.pack(side=tk.LEFT)
format_combobox.set('JPEG') # 기본값 설정
format_combobox.bind('<<ComboboxSelected>>', update_extension_suggestion) # 선택 시 확장자 추천 함수 연결

# 4. 출력 확장자 입력
extension_frame = ttk.Frame(frame)
extension_frame.pack(fill='x', pady=5, padx=55) # 라벨 너비 고려하여 왼쪽 패딩
ttk.Label(extension_frame, text="출력 확장자:").pack(side=tk.LEFT, padx=(0, 5))
extension_entry = ttk.Entry(extension_frame, width=10)
extension_entry.pack(side=tk.LEFT)
extension_entry.insert(0, ".jpg") # 기본값 (JPEG에 맞춰)
ttk.Label(extension_frame, text="(예: .jpg, .png)").pack(side=tk.LEFT, padx=(5,0))


# 5. 실행 버튼
start_button = ttk.Button(frame, text="변환 시작", command=run_conversion_thread)
start_button.pack(pady=10)

# 6. 상태 메시지 라벨
status_label = ttk.Label(frame, text="원본 폴더를 선택하고 변환 시작 버튼을 누르세요.", wraplength=530) # 자동 줄바꿈
status_label.pack(fill='x', pady=5)

# --- 메인 루프 시작 ---
root.mainloop()