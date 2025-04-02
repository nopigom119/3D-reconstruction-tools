import tkinter as tk
from tkinter import filedialog
import open3d as o3d
import os

def visualize_ply():
    file_path = file_path_entry.get()
    if not file_path:
        status_label.config(text="오류: PLY 파일 경로를 입력해주세요.")
        return

    if not file_path.endswith(".ply"):
        status_label.config(text="오류: PLY 파일만 선택해주세요.")
        return

    if not os.path.exists(file_path):
        status_label.config(text=f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return

    try:
        pcd = o3d.io.read_point_cloud(file_path)
        o3d.visualization.draw_geometries([pcd])
        status_label.config(text=f"'{os.path.basename(file_path)}' 파일 시각화 완료. 창을 닫으면 프로그램이 종료됩니다.")
    except Exception as e:
        status_label.config(text=f"오류: PLY 파일 로드 실패 - {e}")

def open_file_dialog():
    file_path = filedialog.askopenfilename(
        defaultextension=".ply",
        filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]
    )
    if file_path:
        file_path_entry.delete(0, tk.END)
        file_path_entry.insert(0, file_path)

if __name__ == "__main__":
    window = tk.Tk()
    window.title("PLY 파일 뷰어")

    file_path_label = tk.Label(window, text="PLY 파일 경로:")
    file_path_label.pack(pady=5)

    file_path_entry = tk.Entry(window, width=50)
    file_path_entry.pack(pady=5)

    browse_button = tk.Button(window, text="찾아보기", command=open_file_dialog)
    browse_button.pack(pady=5)

    visualize_button = tk.Button(window, text="PLY 파일 시각화", command=visualize_ply)
    visualize_button.pack(pady=10)

    status_label = tk.Label(window, text="", bd=1, relief="sunken", anchor="w")
    status_label.pack(fill="x", side="bottom")

    window.mainloop()