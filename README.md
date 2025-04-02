## 사용 방법 (실행 파일)

본 도구 모음은 별도의 설치 과정 없이 바로 사용할 수 있는 독립 실행 파일(`.exe`) 형태로 제공됩니다.

1.  아래 Google Drive 링크에 접속하여 각 도구의 `.exe` 파일을 다운로드 받으세요:
    * **[프로그램 다운로드 링크](https://drive.google.com/drive/folders/1zDhQXKZ3tVrTjAo_0T5F1FbfvhAuYhkS?usp=drive_link)**
2.  다운로드 받은 폴더에서 사용하고자 하는 기능의 `.exe` 파일을 더블 클릭하여 실행합니다.
3.  각 도구의 상세한 사용법 및 기능은 아래의 해당 도구 설명을 참고하시기 바랍니다.

--------------------------------------------------------------------------

## How to Use (Executable Files)

This tool suite is provided as standalone executable files (`.exe`) that can be used immediately without any installation process.

1.  Access the Google Drive link below to download the `.exe` files for each tool:
    * **[Program Download Link](https://drive.google.com/drive/folders/1zDhQXKZ3tVrTjAo_0T5F1FbfvhAuYhkS?usp=drive_link)**
2.  From the downloaded folder, double-click the `.exe` file corresponding to the function you wish to use.
3.  For detailed instructions and features of each tool, please refer to the respective tool descriptions below.

##########################################################################
# 2D 이미지 기반 3D 재구성 도구 모음

이 저장소는 2D 이미지 시퀀스로부터 3D 포인트 클라우드 모델을 생성하기 위한 GUI 기반 도구 모음입니다. 제공되는 도구들은 다음의 주요 단계를 순차적으로 지원합니다:

1.  **이미지 일괄 변환 및 이름 변경:** 이미지 파일 형식을 변환하고 순서대로 이름을 변경하여 데이터 준비를 돕습니다.
2.  **이미지 기반 K 행렬 추정:** 이미지 정보를 바탕으로 카메라 내부 파라미터(K 행렬)를 추정하여 `K.txt` 파일을 생성합니다.
3.  **점진적 SfM 재구성:** 이미지 시퀀스와 K 행렬을 사용하여 점진적 Structure from Motion (SfM) 방식으로 3D 포인트 클라우드(`.ply`)를 재구성합니다.
4.  **PLY 파일 뷰어:** 생성된 `.ply` 형식의 3D 모델 파일을 시각적으로 확인합니다.

각 도구는 독립적인 실행 파일(`.exe`)로 제공되어 복잡한 설정 없이 간편하게 사용할 수 있도록 만들어졌습니다. 아래에서 각 도구의 상세한 기능과 사용 방법을 확인할 수 있습니다.

---
*(이 아래에 각 도구별 README 내용을 순서대로 추가하시면 됩니다.)*

--------------------------------------------------------------------------

# 2D Image-Based 3D Reconstruction Tool Suite

This repository contains a suite of GUI-based tools designed to facilitate the creation of a 3D point cloud model from a sequence of 2D images. The provided tools support the following key steps in sequence:

1.  **Image Batch Convert & Rename:** Helps prepare data by converting image file formats and renaming them sequentially.
2.  **Image-based K Matrix Estimator:** Estimates the camera intrinsic parameters (K matrix) based on image information and generates a `K.txt` file.
3.  **Incremental SfM Reconstruction:** Reconstructs a 3D point cloud (`.ply`) using an image sequence and the K matrix via the incremental Structure from Motion (SfM) method.
4.  **PLY File Viewer:** Visually inspects the generated 3D model files in `.ply` format.

Each tool is provided as a standalone executable (`.exe`) designed for ease of use without complex setup. Detailed functionality and usage instructions for each tool can be found below.

---
*(You can add the README content for each tool sequentially below this section.)*

##########################################################################
# 이미지 일괄 변환 및 이름 변경 프로그램

## 기능 및 목적

본 프로그램은 Tkinter 라이브러리를 사용하여 GUI 기반으로 개발된 이미지 처리 애플리케이션입니다. 사용자는 이 앱을 통해 다음과 같은 기능을 수행할 수 있습니다.

* **이미지 형식 변환:** 지정된 폴더 내의 이미지 파일들('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')을 사용자가 선택한 형식('JPEG', 'PNG', 'BMP', 'GIF', 'TIFF', 'WEBP')으로 일괄 변환합니다.
* **파일 이름 변경:** 변환된 이미지 파일들의 이름을 숫자 순서대로 (예: 0001.jpg, 0002.jpg) 자동으로 변경합니다.
* **저장 경로 지정:** 변환된 파일들을 저장할 폴더를 별도로 지정할 수 있습니다. 지정하지 않으면 원본 폴더에 저장됩니다.
* **원본 파일 관리:** 저장 폴더를 지정하지 않고 원본 폴더에 저장할 경우, **변환 과정에서 원본 파일과 이름/형식이 달라지면 원본 파일이 삭제될 수 있으니 주의가 필요합니다.**
* **사용자 인터페이스:** 직관적인 GUI를 통해 원본 폴더, 저장 폴더, 출력 형식, 확장자를 쉽게 선택하고 변환 작업을 시작할 수 있습니다.
* **처리 상태 표시:** 하단 상태 메시지 라벨을 통해 작업 진행 상황과 결과를 실시간으로 확인할 수 있습니다.

이 앱은 여러 이미지 파일의 형식을 통일하거나, 파일 이름을 순서대로 정리해야 할 때 유용하게 활용될 수 있습니다.

## 사용 방법

1.  프로그램(`exe` 파일)을 실행합니다.
2.  **원본 폴더** 항목 옆의 "찾아보기..." 버튼을 클릭하여 변환할 이미지들이 있는 폴더를 선택합니다.
3.  (선택 사항) **저장 폴더** 항목 옆의 "찾아보기..." 버튼을 클릭하여 변환된 이미지들을 저장할 폴더를 선택합니다.
    * **주의:** 이 항목을 비워두면 원본 폴더에 저장되며, 변환 후 이름이나 형식이 달라진 경우 원본 파일이 삭제될 수 있습니다. 원본 보존이 필요하면 반드시 다른 저장 폴더를 지정하세요.
4.  **출력 형식** 콤보박스에서 원하는 이미지 형식(예: JPEG, PNG)을 선택합니다.
5.  **출력 확장자** 입력 필드에 원하는 파일 확장자(반드시 '.'으로 시작, 예: .jpg, .png)를 확인하거나 입력합니다. 출력 형식 선택 시 자동으로 추천 확장자가 입력됩니다.
6.  "변환 시작" 버튼을 클릭합니다.
7.  하단의 상태 메시지를 통해 변환 진행 상황을 확인합니다. 작업이 완료되면 알림창이 뜹니다.

## 저작권

본 프로그램은 다음 조건에 따라 이용할 수 있습니다.
**크리에이티브 커먼즈 저작자표시-비영리-동일조건변경허락 4.0 국제 라이선스 (CC BY-NC-SA 4.0)**

* **출처 표시:** 본 프로그램의 출처 (작성자 또는 개발팀)를 명시해야 합니다.
* **비상업적 이용:** 본 프로그램을 상업적인 목적으로 이용할 수 없습니다.
* **변경 가능:** 본 프로그램을 수정하거나 2차 저작물을 만들 수 있습니다.
* **동일 조건 변경 허락:** 2차 저작물에 대해서도 동일한 조건으로 이용 허락해야 합니다.
**자세한 내용은 크리에이티브 커먼즈 홈페이지 (https://creativecommons.org/licenses/by-nc-sa/4.0/deed.ko) 에서 확인하실 수 있습니다.**

## 문의

본 프로그램에 대한 문의사항은 [rycbabd@gmail.com] 로 연락주시기 바랍니다.

--------------------------------------------------------------------------

# Image Batch Convert & Rename Program

## Functionality and Purpose

This program is a GUI-based image processing application developed using the Tkinter library. Users can perform the following functions through this app:

* **Image Format Conversion:** Batch convert image files ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp') within a specified folder to a user-selected format ('JPEG', 'PNG', 'BMP', 'GIF', 'TIFF', 'WEBP').
* **File Renaming:** Automatically rename the converted image files sequentially with numbers (e.g., 0001.jpg, 0002.jpg).
* **Destination Path Specification:** You can specify a separate folder to save the converted files. If not specified, they will be saved in the source folder.
* **Source File Management:** When saving to the source folder (no destination folder specified), **please be aware: original files might be deleted if their names or formats are changed during conversion.** Caution is advised.
* **User Interface:** An intuitive GUI allows easy selection of the source folder, destination folder, output format, and extension to start the conversion process.
* **Processing Status Display:** Real-time progress and results can be monitored via the status message label at the bottom.

This app is useful when you need to unify the format of multiple image files or organize file names sequentially.

## How to Use

1.  Run the program (`.exe` file).
2.  Click the "Browse..." button next to the **Source Folder** field and select the folder containing the images to be converted.
3.  (Optional) Click the "Browse..." button next to the **Destination Folder** field and select a folder to save the converted images.
    * **Caution:** If this field is left blank, files will be saved in the source folder. If the name or format changes after conversion, the original file might be deleted. If you need to preserve the original files, be sure to specify a different destination folder.
4.  Select the desired image format (e.g., JPEG, PNG) from the **Output Format** dropdown list.
5.  Verify or enter the desired file extension (must start with '.', e.g., .jpg, .png) in the **Output Extension** input field. A suggested extension is automatically filled in when you select the output format.
6.  Click the "Start Conversion" button.
7.  Monitor the conversion progress via the status messages at the bottom. A notification window will appear upon completion.

## License

**This program is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**

* **Attribution:** You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in a way that suggests the licensor endorses you or your use.
* **Non-Commercial Use:** You may not use this program for commercial purposes.
* **Modification Allowed:** You can modify this program or create derivative works.
* **Same Conditions for Change Permission:** If you modify or create derivative works of this program, you must distribute your contributions under the same license as the original.
**You can check the details on the Creative Commons website (https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).**

## Contact

For inquiries about this program, please contact [rycbabd@gmail.com].

##########################################################################
# 이미지 기반 K 행렬 추정기

## 기능 및 목적

본 프로그램은 Tkinter, OpenCV, NumPy 라이브러리를 사용하여 GUI 기반으로 개발된 카메라 내부 파라미터(K 행렬) 추정 애플리케이션입니다. 사용자는 이 앱을 통해 다음과 같은 기능을 수행할 수 있습니다.

* **K 행렬 추정:** 지정된 폴더 내의 이미지 파일(`.jpg`, `.jpeg`, `.png`) 중 **첫 번째 이미지**를 분석하여 카메라의 내부 파라미터 행렬(K 행렬)을 추정합니다.
* **추정 방식:**
    * 주점 (`cx`, `cy`)은 이미지의 정중앙으로 가정합니다 (`width/2`, `height/2`).
    * 초점 거리 (`fx`, `fy`)는 사용자가 선택한 방식에 따라 추정됩니다:
        * **max(너비, 높이):** 이미지의 너비와 높이 중 더 큰 값을 초점 거리로 사용합니다 (fx = fy = max(width, height)).
        * **직접 입력:** 사용자가 양의 실수 값을 직접 입력하여 초점 거리로 사용합니다 (fx = fy = 입력값).
    * 비대칭 계수(skew)는 0으로 가정합니다.
* **결과 저장:** 추정된 3x3 K 행렬을 사용자가 지정한 폴더에 원하는 파일 이름(`.txt` 형식 권장)으로 저장합니다. 파일 내용은 공백으로 구분된 숫자 값으로 저장됩니다.
* **사용자 인터페이스:** GUI를 통해 이미지 폴더, 저장 폴더, 저장 파일 이름, 초점 거리 추정 방식을 쉽게 설정할 수 있습니다.
* **상세 로그:** 하단의 스크롤 가능한 텍스트 영역을 통해 이미지 검색, 크기 분석, 파라미터 추정, K 행렬 결과, 저장 과정 등 상세한 처리 상태 및 결과를 확인할 수 있습니다.

이 앱은 카메라 캘리브레이션 정보가 없을 때, 기본적인 가정을 통해 K 행렬의 근사값을 빠르게 얻고자 할 때 유용하게 활용될 수 있습니다.

## 사용 방법

1.  프로그램(`exe` 파일)을 실행합니다.
2.  **이미지 폴더** 항목 옆의 "찾아보기..." 버튼을 클릭하여 K 행렬 추정에 사용할 이미지가 포함된 폴더를 선택합니다. (폴더 내에 `.jpg`, `.jpeg`, `.png` 파일이 최소 1개 이상 있어야 합니다.)
3.  **저장 폴더** 항목 옆의 "찾아보기..." 버튼을 클릭하여 추정된 K 행렬 파일을 저장할 폴더를 선택합니다.
4.  **저장 파일 이름** 항목에 원하는 파일 이름을 입력합니다. (기본값: `K_estimated.txt`)
5.  **초점 거리 방식**을 선택합니다:
    * **max(너비, 높이):** 첫 번째 이미지의 너비와 높이 중 큰 값을 초점 거리로 사용하려면 이 옵션을 선택합니다.
    * **직접 입력:** 특정 초점 거리 값을 사용하려면 이 옵션을 선택하고, 옆의 입력란에 양의 실수 값을 입력합니다.
6.  "K 행렬 추정 및 저장" 버튼을 클릭합니다.
7.  하단의 **결과 및 상태** 영역에서 처리 과정과 추정된 K 행렬 값, 저장 성공 여부를 확인합니다.

## 저작권

본 프로그램은 다음 조건에 따라 이용할 수 있습니다.
**크리에이티브 커먼즈 저작자표시-비영리-동일조건변경허락 4.0 국제 라이선스 (CC BY-NC-SA 4.0)**

* **출처 표시:** 본 프로그램의 출처 (작성자 또는 개발팀)를 명시해야 합니다.
* **비상업적 이용:** 본 프로그램을 상업적인 목적으로 이용할 수 없습니다.
* **변경 가능:** 본 프로그램을 수정하거나 2차 저작물을 만들 수 있습니다.
* **동일 조건 변경 허락:** 2차 저작물에 대해서도 동일한 조건으로 이용 허락해야 합니다.
**자세한 내용은 크리에이티브 커먼즈 홈페이지 (https://creativecommons.org/licenses/by-nc-sa/4.0/deed.ko) 에서 확인하실 수 있습니다.**

## 문의

본 프로그램에 대한 문의사항은 [rycbabd@gmail.com] 로 연락주시기 바랍니다.

--------------------------------------------------------------------------

# Image-based K Matrix Estimator

## Functionality and Purpose

This program is a GUI-based application developed using Tkinter, OpenCV, and NumPy libraries to estimate camera intrinsic parameters (K matrix). Users can perform the following functions through this app:

* **K Matrix Estimation:** Estimates the camera's intrinsic parameter matrix (K matrix) by analyzing the **first image** found (`.jpg`, `.jpeg`, `.png`) within a specified folder.
* **Estimation Method:**
    * The principal point (`cx`, `cy`) is assumed to be the image center (`width/2`, `height/2`).
    * The focal length (`fx`, `fy`) is estimated based on the user's choice:
        * **max(width, height):** Uses the larger value between the image width and height as the focal length (fx = fy = max(width, height)).
        * **Direct Input:** Uses a positive floating-point value entered by the user as the focal length (fx = fy = input value).
    * The skew coefficient is assumed to be 0.
* **Result Saving:** Saves the estimated 3x3 K matrix to a user-specified folder with a desired file name (recommend `.txt` format). The file content consists of space-separated numeric values.
* **User Interface:** The GUI allows easy configuration of the image folder, save folder, save file name, and focal length estimation method.
* **Detailed Log:** A scrollable text area at the bottom provides detailed processing status and results, including image search, size analysis, parameter estimation, the estimated K matrix, and the saving process.

This app can be useful for quickly obtaining an approximate K matrix based on basic assumptions when camera calibration information is unavailable.

## How to Use

1.  Run the program (`.exe` file).
2.  Click the "Browse..." button next to the **Image Folder** field and select the folder containing the image(s) to be used for K matrix estimation. (The folder must contain at least one `.jpg`, `.jpeg`, or `.png` file.)
3.  Click the "Browse..." button next to the **Save Folder** field and select the folder where the estimated K matrix file will be saved.
4.  Enter the desired file name in the **Save File Name** field. (Default: `K_estimated.txt`)
5.  Select the **Focal Length Method**:
    * **max(width, height):** Choose this option to use the maximum of the first image's width and height as the focal length.
    * **Direct Input:** Choose this option to use a specific focal length value, then enter a positive floating-point number in the adjacent entry field.
6.  Click the "**K 행렬 추정 및 저장**" (Estimate & Save K Matrix) button.
7.  Check the **Results and Status** area at the bottom for the processing steps, the estimated K matrix values, and whether the save was successful.

## License

**This program is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**

* **Attribution:** You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in a way that suggests the licensor endorses you or your use.
* **Non-Commercial Use:** You may not use this program for commercial purposes.
* **Modification Allowed:** You can modify this program or create derivative works.
* **Same Conditions for Change Permission:** If you modify or create derivative works of this program, you must distribute your contributions under the same license as the original.
**You can check the details on the Creative Commons website (https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).**

## Contact

For inquiries about this program, please contact [rycbabd@gmail.com].

##########################################################################
# 점진적 SfM 재구성 프로그램

## 기능 및 목적

본 프로그램은 Tkinter, OpenCV, NumPy, Open3D 라이브러리를 사용하여 GUI 기반으로 개발된 점진적 Structure from Motion (SfM) 3D 재구성 애플리케이션입니다. 사용자는 이 앱을 통해 다음과 같은 기능을 수행할 수 있습니다.

* **3D 포인트 클라우드 재구성:** 지정된 폴더 내의 순차적인 2D 이미지들과 카메라 내부 파라미터(K 행렬) 파일을 이용하여 3D 포인트 클라우드(`.ply` 형식)를 생성합니다.
* **점진적 방식 (Incremental SfM):**
    * 첫 두 이미지로 초기 3D 맵을 생성합니다 (특징점 매칭, Essential Matrix 추정, 상대 포즈 복구, 삼각측량).
    * 이후 이미지를 하나씩 추가하며 다음 과정을 반복합니다:
        * **PnP (Perspective-n-Point):** 현재 이미지의 특징점과 기존 3D 맵 포인트 간의 2D-3D 매칭을 통해 현재 카메라의 전역 포즈(회전, 이동)를 추정합니다.
        * **새 포인트 삼각측량:** 현재 이미지와 이전 이미지 간의 2D-2D 매칭을 통해 새로운 3D 포인트를 계산하고 맵에 추가합니다.
        * **맵 업데이트 및 클리닝:** 맵 포인트를 갱신하고, 각 단계마다 Statistical Outlier Removal (SOR) 알고리즘을 적용하여 맵의 이상치를 제거합니다 (선택적 파라미터 조절).
* **입력 요구사항:**
    * **이미지 폴더:** 처리할 이미지 시퀀스 파일들(`.jpg`, `.jpeg`, `.png` 등 지원)이 포함된 폴더. 파일 이름 순서대로 처리됩니다.
    * **K 행렬 파일:** 이미지 폴더 내에 `K.txt` 파일 이름으로 저장된 3x3 카메라 내부 파라미터 행렬 파일. (두 번째 프로그램으로 생성 가능)
* **출력:**
    * **중간 결과:** 각 이미지 처리 단계별 3D 맵 포인트 클라우드를 `output_ply_by_process` 폴더 내에 `step_*.ply` 파일로 저장합니다.
    * **최종 결과:** 모든 이미지를 처리하고 최종 이상치 제거를 거친 후의 3D 포인트 클라우드를 `output_incremental.ply` 파일로 저장합니다.
* **사용자 인터페이스 및 제어:**
    * GUI를 통해 데이터 폴더를 선택할 수 있습니다.
    * SfM 파이프라인의 주요 파라미터(특징점 매칭 임계값, PnP RANSAC 설정, SOR 설정 등)를 사용자가 직접 조절할 수 있습니다. 각 파라미터에 대한 설명은 마우스 커서를 올리면 툴팁으로 표시됩니다.
    * 상세한 처리 과정과 로그 메시지가 GUI 하단에 실시간으로 표시됩니다.
    * 백그라운드 스레드에서 재구성 작업을 수행하여 GUI 응답성을 유지합니다.

이 앱은 연속된 이미지 시퀀스로부터 3D 구조를 복원하는 연구나 프로젝트에 활용될 수 있습니다.

## 사용 방법

1.  **데이터 폴더 준비:**
    * 3D로 재구성할 대상의 이미지 시퀀스 파일(예: `0001.jpg`, `0002.jpg`, ...)를 하나의 폴더에 넣습니다.
    * 해당 이미지들에 대한 3x3 카메라 내부 파라미터 행렬을 `K.txt` 파일로 저장하여 같은 폴더 안에 넣습니다. (`K.txt` 파일은 두 번째 K 행렬 추정 프로그램으로 생성할 수 있습니다.)
2.  프로그램(`exe` 파일)을 실행합니다.
3.  **데이터 폴더** 항목 옆의 "찾아보기..." 버튼을 클릭하여 1번에서 준비한 폴더를 선택합니다.
4.  (선택 사항) **SfM 파라미터** 섹션에서 각 파라미터 값을 필요에 따라 조절합니다. 기본값이 설정되어 있으며, 각 항목에 마우스를 올리면 해당 파라미터에 대한 설명을 볼 수 있습니다.
5.  "재구성 시작" 버튼을 클릭합니다.
6.  **로그 및 상태** 영역에 출력되는 상세한 처리 과정을 확인합니다. 이미지 수와 컴퓨터 성능에 따라 처리 시간이 오래 걸릴 수 있습니다.
7.  작업이 완료되면 프로그램이 실행된 폴더(또는 `.exe` 파일이 있는 위치)에 다음과 같은 결과가 생성됩니다:
    * `output_ply_by_process` 폴더: 각 이미지 처리 단계별 중간 결과 `.ply` 파일들이 저장됩니다.
    * `output_incremental.ply` 파일: 최종적으로 생성된 3D 포인트 클라우드 파일입니다. (MeshLab, CloudCompare 등의 뷰어로 확인할 수 있습니다.)

## 저작권

본 프로그램은 다음 조건에 따라 이용할 수 있습니다.
**크리에이티브 커먼즈 저작자표시-비영리-동일조건변경허락 4.0 국제 라이선스 (CC BY-NC-SA 4.0)**

* **출처 표시:** 본 프로그램의 출처 (작성자 또는 개발팀)를 명시해야 합니다.
* **비상업적 이용:** 본 프로그램을 상업적인 목적으로 이용할 수 없습니다.
* **변경 가능:** 본 프로그램을 수정하거나 2차 저작물을 만들 수 있습니다.
* **동일 조건 변경 허락:** 2차 저작물에 대해서도 동일한 조건으로 이용 허락해야 합니다.
**자세한 내용은 크리에이티브 커먼즈 홈페이지 (https://creativecommons.org/licenses/by-nc-sa/4.0/deed.ko) 에서 확인하실 수 있습니다.**

## 문의

본 프로그램에 대한 문의사항은 [rycbabd@gmail.com] 로 연락주시기 바랍니다.

--------------------------------------------------------------------------

# Incremental SfM Reconstruction Program

## Functionality and Purpose

This program is a GUI-based application for incremental Structure from Motion (SfM) 3D reconstruction, developed using Tkinter, OpenCV, NumPy, and Open3D libraries. Users can perform the following functions through this app:

* **3D Point Cloud Reconstruction:** Generates a 3D point cloud (`.ply` format) from a sequence of 2D images located in a specified folder and a camera intrinsic matrix (K matrix) file.
* **Incremental SfM Approach:**
    * Initializes a 3D map using the first two images (feature matching, Essential Matrix estimation, relative pose recovery, triangulation).
    * Iteratively processes subsequent images one by one:
        * **PnP (Perspective-n-Point):** Estimates the global pose (rotation, translation) of the current camera by matching 2D features in the current image to existing 3D map points.
        * **New Point Triangulation:** Calculates new 3D points by matching 2D features between the current and previous images, then triangulates using their global poses. Adds these new points to the map.
        * **Map Update and Cleaning:** Updates the map points and applies the Statistical Outlier Removal (SOR) algorithm at each step to remove outliers from the map (parameters are adjustable).
* **Input Requirements:**
    * **Image Folder:** A folder containing the image sequence files (supports `.jpg`, `.jpeg`, `.png`, etc.) to be processed. Files are processed in alphabetical/numerical order of their names.
    * **K Matrix File:** A 3x3 camera intrinsic matrix saved as `K.txt` within the same image folder. (This can be generated using the second K Matrix Estimator program).
* **Output:**
    * **Intermediate Results:** Saves the 3D map point cloud at each processing step as `step_*.ply` files inside the `output_ply_by_process` folder.
    * **Final Result:** Saves the final 3D point cloud, after processing all images and applying a final outlier removal step, as `output_incremental.ply`.
* **User Interface and Control:**
    * Provides a GUI to select the data folder.
    * Allows users to tune key parameters of the SfM pipeline (e.g., feature matching threshold, PnP RANSAC settings, SOR settings). Tooltips appear on hover, explaining each parameter.
    * Displays detailed processing steps and log messages in real-time at the bottom of the GUI.
    * Performs the reconstruction task in a background thread to maintain GUI responsiveness.

This app can be utilized for research or projects involving 3D structure recovery from sequential image data.

## How to Use

1.  **Prepare the Data Folder:**
    * Place the image sequence files (e.g., `0001.jpg`, `0002.jpg`, ...) for the target scene into a single folder.
    * Save the corresponding 3x3 camera intrinsic matrix as a file named `K.txt` inside the *same* folder. (The `K.txt` file can be generated using the second K Matrix Estimator program).
2.  Run the program (`.exe` file).
3.  Click the "Browse..." button next to the **Data Folder** field and select the folder prepared in step 1.
4.  (Optional) Adjust the parameter values in the **SfM Parameters** section as needed. Default values are provided, and hovering over each item will show a tooltip explaining the parameter.
5.  Click the "**재구성 시작**" (Start Reconstruction) button.
6.  Monitor the detailed processing steps in the **Log and Status** area. Processing time can be significant depending on the number of images and system performance.
7.  Upon completion, check the directory where the program was run (or where the `.exe` file is located) for the following outputs:
    * `output_ply_by_process` folder: Contains intermediate `.ply` files from each step.
    * `output_incremental.ply` file: The final reconstructed 3D point cloud. (You can view this file using software like MeshLab or CloudCompare).

## License

**This program is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**

* **Attribution:** You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in a way that suggests the licensor endorses you or your use.
* **Non-Commercial Use:** You may not use this program for commercial purposes.
* **Modification Allowed:** You can modify this program or create derivative works.
* **Same Conditions for Change Permission:** If you modify or create derivative works of this program, you must distribute your contributions under the same license as the original.
**You can check the details on the Creative Commons website (https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).**

## Contact

For inquiries about this program, please contact [rycbabd@gmail.com].

##########################################################################
# PLY 파일 뷰어

## 기능 및 목적

본 프로그램은 Tkinter와 Open3D 라이브러리를 사용하여 GUI 기반으로 개발된 간단한 3D 포인트 클라우드(`.ply`) 뷰어 애플리케이션입니다. 사용자는 이 앱을 통해 다음과 같은 기능을 수행할 수 있습니다.

* **PLY 파일 선택:** "찾아보기" 버튼을 통해 로컬 시스템에서 `.ply` 파일을 쉽게 선택하거나, 파일 경로를 직접 입력할 수 있습니다.
* **3D 시각화:** 선택된 `.ply` 파일을 로드하여 Open3D의 시각화 창을 통해 3D 포인트 클라우드를 표시합니다.
* **인터랙티브 뷰:** Open3D 시각화 창에서 마우스를 사용하여 3D 모델을 회전, 확대/축소, 이동하며 살펴볼 수 있습니다.
* **상태 알림:** 파일 선택 오류, 로딩 오류, 시각화 성공 여부 등 기본적인 상태 메시지를 메인 창 하단에 표시합니다.

이 앱은 생성된 3D 포인트 클라우드(`.ply`) 파일을 빠르고 간편하게 확인하는 데 사용될 수 있습니다. (예: 이전 단계의 SfM 결과 확인)

## 사용 방법

1.  프로그램(`exe` 파일)을 실행합니다.
2.  **PLY 파일 경로** 입력란 옆의 "찾아보기" 버튼을 클릭합니다.
3.  파일 탐색기 창이 열리면 시각화하려는 `.ply` 파일을 선택하고 "열기" 버튼을 클릭합니다. 선택한 파일의 경로가 입력란에 자동으로 채워집니다.
    * 또는, 파일의 전체 경로를 알고 있다면 직접 입력란에 붙여넣거나 입력할 수도 있습니다.
4.  "PLY 파일 시각화" 버튼을 클릭합니다.
5.  성공적으로 파일을 로드하면, 별도의 Open3D 시각화 창이 열리며 3D 포인트 클라우드가 나타납니다.
6.  새로 열린 시각화 창에서 마우스를 사용하여 뷰를 조작할 수 있습니다 (왼쪽 클릭 드래그: 회전, 휠 스크롤: 확대/축소, 오른쪽 클릭 드래그: 이동).
7.  메인 프로그램 창 하단의 상태 표시줄에서 파일 로딩 상태나 오류 메시지를 확인할 수 있습니다.
8.  확인이 끝나면 Open3D 시각화 창을 닫습니다.

## 저작권

본 프로그램은 다음 조건에 따라 이용할 수 있습니다.
**크리에이티브 커먼즈 저작자표시-비영리-동일조건변경허락 4.0 국제 라이선스 (CC BY-NC-SA 4.0)**

* **출처 표시:** 본 프로그램의 출처 (작성자 또는 개발팀)를 명시해야 합니다.
* **비상업적 이용:** 본 프로그램을 상업적인 목적으로 이용할 수 없습니다.
* **변경 가능:** 본 프로그램을 수정하거나 2차 저작물을 만들 수 있습니다.
* **동일 조건 변경 허락:** 2차 저작물에 대해서도 동일한 조건으로 이용 허락해야 합니다.
**자세한 내용은 크리에이티브 커먼즈 홈페이지 (https://creativecommons.org/licenses/by-nc-sa/4.0/deed.ko) 에서 확인하실 수 있습니다.**

## 문의

본 프로그램에 대한 문의사항은 [rycbabd@gmail.com] 로 연락주시기 바랍니다.

--------------------------------------------------------------------------

# PLY File Viewer

## Functionality and Purpose

This program is a simple GUI-based viewer application for 3D point clouds (`.ply` files), developed using Tkinter and the Open3D library. Users can perform the following functions through this app:

* **PLY File Selection:** Easily select a `.ply` file from the local system using the "Browse" button, or directly input the file path.
* **3D Visualization:** Loads the selected `.ply` file and displays the 3D point cloud in a separate visualization window provided by Open3D.
* **Interactive View:** Allows users to rotate, zoom, and pan the 3D model interactively using the mouse within the Open3D visualization window.
* **Status Notification:** Displays basic status messages (e.g., file selection errors, loading errors, visualization success) at the bottom of the main window.

This app can be used to quickly and easily inspect generated 3D point cloud (`.ply`) files (e.g., checking the results from the previous SfM step).

## How to Use

1.  Run the program (`.exe` file).
2.  Click the "Browse" button next to the **PLY File Path** entry field.
3.  A file explorer window will open. Navigate to and select the `.ply` file you wish to visualize, then click "Open". The path to the selected file will automatically populate the entry field.
    * Alternatively, if you know the full path to the file, you can paste or type it directly into the entry field.
4.  Click the "**PLY 파일 시각화**" (Visualize PLY File) button.
5.  If the file is loaded successfully, a separate Open3D visualization window will open, displaying the 3D point cloud.
6.  You can interact with the view in the newly opened visualization window using your mouse (Left-click drag: Rotate, Wheel scroll: Zoom, Right-click drag: Pan).
7.  Check the status bar at the bottom of the main program window for file loading status or error messages.
8.  Close the Open3D visualization window when you are finished viewing.

## License

**This program is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**

* **Attribution:** You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in a way that suggests the licensor endorses you or your use.
* **Non-Commercial Use:** You may not use this program for commercial purposes.
* **Modification Allowed:** You can modify this program or create derivative works.
* **Same Conditions for Change Permission:** If you modify or create derivative works of this program, you must distribute your contributions under the same license as the original.
**You can check the details on the Creative Commons website (https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).**

## Contact

For inquiries about this program, please contact [rycbabd@gmail.com].
