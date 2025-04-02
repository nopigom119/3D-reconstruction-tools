# 2D 이미지 기반 3D 재구성 도구 모음에 기여하기

2D 이미지 기반 3D 재구성 도구 모음에 기여해주셔서 감사합니다! 모든 기여는 환영합니다.
버그 수정, 기능 추가, 문서 개선 등 어떤 형태의 기여든 감사하게 생각합니다.

## 기여 방법

### 1. 이슈 등록

* 새로운 기능 제안이나 버그 보고는 **[[Your Project Repository Name] Issues 페이지](https://github.com/[Your GitHub Username]/[Your Project Repository Name]/issues)** 에 등록해주세요.
* 이슈 제목은 간결하고 명확하게 작성해주세요. (예: "[SfM] PnP 실패 시 로그 개선", "[Viewer] 특정 PLY 파일 로딩 오류")
* 이슈 내용에는 문제 상황(어떤 도구에서, 어떤 입력으로, 어떤 문제가 발생하는지)이나 제안 내용을 상세하게 설명해주세요. 스크린샷이나 로그 메시지를 첨부하면 도움이 됩니다.

### 2. 코드 기여

1.  **[[Your Project Repository Name] 저장소](https://github.com/[Your GitHub Username]/[Your Project Repository Name])** 를 포크해주세요.
2.  포크한 저장소에서 새로운 브랜치를 만들고 코드를 작성해주세요.
    * 브랜치 이름은 기여하려는 도구 이름, 기능 이름이나 이슈 번호를 포함해주세요. (예: `feature/sfm-bundle-adjustment`, `bugfix/k-estimator-path-error-15`)
3.  작성한 코드는 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 코딩 스타일 가이드라인을 준수해주세요.
4.  변경 사항을 테스트하고, 특히 로직 변경 시에는 다양한 입력에 대해 테스트해주세요. (필요한 경우 테스트 코드 작성)
5.  커밋 메시지는 간결하고 명확하게 작성해주세요. 변경 사항이 어떤 도구에 영향을 미치는지 명시하면 좋습니다. (예: `feat(SfM): Add bundle adjustment option`, `fix(Viewer): Handle PLY loading exception for #22`)
6.  **[[Your Project Repository Name] Pull Requests 페이지](https://github.com/[Your GitHub Username]/[Your Project Repository Name]/pulls)** 로 풀 리퀘스트를 보내주세요.

### 3. 문서 기여

* README 파일의 설명 개선, 각 도구의 사용법 명확화, 코드 내 주석 추가 등 문서 개선에 기여해주실 수 있습니다.
* 문서 내용은 명확하고 이해하기 쉽게 작성해주세요.

## 코드 작성 규칙

* [PEP 8](https://www.python.org/dev/peps/pep-0008/) 코딩 스타일 가이드라인을 준수해주세요.
* 주요 함수 및 클래스에 대한 설명 (docstring)을 작성해주세요.
* 코드 변경 시 필요한 경우, 다른 개발자가 이해하기 쉽도록 주석을 추가해주세요.
* 각 도구는 독립적으로 실행 가능해야 하므로, 특정 도구 수정 시 다른 도구에 예기치 않은 영향을 주지 않도록 주의해주세요.

## 커밋 메시지 규칙

* 커밋 메시지는 간결하고 명확하게 작성해주세요. (참고: [Conventional Commits](https://www.conventionalcommits.org/))
* 커밋 메시지 제목은 변경 사항을 요약하여 작성하고(가능하면 관련 도구 명시), 본문에는 상세한 내용을 설명해주세요.

## 풀 리퀘스트 절차

1.  포크한 저장소에서 브랜치를 만들고 코드를 작성해주세요.
2.  변경 사항을 커밋해주세요.
3.  원격 저장소(자신의 포크)에 변경 사항을 푸시해주세요.
4.  **[[Your Project Repository Name] Pull Requests 페이지](https://github.com/[Your GitHub Username]/[Your Project Repository Name]/pulls)** 에서 원본 저장소로 Pull Request를 생성해주세요.
5.  풀 리퀘스트 내용을 확인하고, 어떤 문제를 해결하거나 어떤 기능을 추가하는지 명확히 설명해주세요.
6.  리뷰어의 리뷰를 반영하여 풀 리퀘스트를 최종 확정해주세요.

## 문의

* 본 도구 모음에 대한 문의는 **[[Your Project Repository Name] Issues 페이지](https://github.com/[Your GitHub Username]/[Your Project Repository Name]/issues)** 를 이용해주세요.

## 라이선스

본 도구 모음은 CC BY-NC-SA 4.0 라이선스에 따라 배포됩니다.

--------------------------------------------------------------------------

# Contributing to 2D Image-Based 3D Reconstruction Tool Suite

Thank you for contributing to the 2D Image-Based 3D Reconstruction Tool Suite! All contributions are welcome.
We appreciate any form of contribution, be it bug fixes, feature additions, or documentation improvements.

## How to Contribute

### 1. Reporting Issues

* Please register new feature suggestions or bug reports on the **[[Your Project Repository Name] Issues page](https://github.com/[Your GitHub Username]/[Your Project Repository Name]/issues)**.
* Make sure the issue title is concise and clear (e.g., "[SfM] Improve logging on PnP failure", "[Viewer] Loading error for specific PLY file").
* Provide detailed descriptions of the problem (which tool, what input, what is the issue) or suggestion in the issue content. Attaching screenshots or log messages is helpful.

### 2. Contributing Code

1.  Fork the **[[Your Project Repository Name] repository](https://github.com/[Your GitHub Username]/[Your Project Repository Name])**.
2.  Create a new branch in the forked repository and write the code.
    * Include the tool name being contributed to, feature name, or issue number in the branch name (e.g., `feature/sfm-bundle-adjustment`, `bugfix/k-estimator-path-error-15`).
3.  Follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) coding style guidelines when writing the code.
4.  Test the changes, especially with various inputs if logic is modified. (Write test code if necessary).
5.  Write concise and clear commit messages. Specifying which tool the change affects is recommended. (e.g., `feat(SfM): Add bundle adjustment option`, `fix(Viewer): Handle PLY loading exception for #22`).
6.  Submit a pull request to the **[[Your Project Repository Name] Pull Requests page](https://github.com/[Your GitHub Username]/[Your Project Repository Name]/pulls)**.

### 3. Contributing to Documentation

* You can contribute to improving documentation, such as clarifying the README file, enhancing usage instructions for each tool, or adding comments within the code.
* Make sure the documentation is clear and easy to understand.

## Code Writing Rules

* Follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) coding style guidelines.
* Write descriptions (docstrings) for major functions and classes.
* Add comments as necessary when changing the code to help other developers understand.
* As each tool should be runnable independently, please be careful that modifications to one tool do not unintentionally affect others.

## Commit Message Rules

* Write concise and clear commit messages (Reference: [Conventional Commits](https://www.conventionalcommits.org/)).
* Summarize the changes in the commit message title (mentioning the relevant tool if possible) and provide detailed descriptions in the body.

## Pull Request Procedure

1.  Create a branch in the forked repository and write the code.
2.  Commit the changes.
3.  Push the changes to your remote repository (your fork).
4.  Create a Pull Request from your fork to the original repository on the **[[Your Project Repository Name] Pull Requests page](https://github.com/[Your GitHub Username]/[Your Project Repository Name]/pulls)**.
5.  Review the pull request content and clearly explain the problem it solves or the feature it adds.
6.  Reflect the reviewer's feedback and finalize the pull request.

## Inquiries

* For inquiries about this tool suite, please use the **[[Your Project Repository Name] Issues page](https://github.com/[Your GitHub Username]/[Your Project Repository Name]/issues)**.

## License

This tool suite is distributed under the CC BY-NC-SA 4.0 license.