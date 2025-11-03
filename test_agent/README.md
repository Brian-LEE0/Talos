# AI Agent Test Suite

이 디렉토리는 AI 에이전트의 자율적인 파일 읽기 및 검색 기능을 테스트하기 위한 샘플 프로젝트입니다.

## 프로젝트 구조

```
test_agent/
├── main.py                    # 메인 엔트리 포인트
├── config.py                  # 설정 파일
└── utils/
    ├── __init__.py           # 유틸리티 패키지 초기화
    ├── data_processor.py     # 데이터 처리 로직
    └── file_handler.py       # 파일 I/O 처리
```

## 프로젝트 설명

이것은 CSV 파일을 읽고 처리하는 데이터 처리 애플리케이션입니다.

### 주요 기능:
- CSV 파일 읽기
- 데이터 유효성 검증
- 데이터 변환 및 계산
- 결과를 CSV로 저장

### 모듈별 역할:

**main.py**
- 애플리케이션의 진입점
- 전체 프로세스 오케스트레이션
- config와 utils 모듈 사용

**config.py**
- 데이터베이스 URL
- 재시도 횟수 및 타임아웃 설정
- 입출력 디렉토리 경로

**utils/data_processor.py**
- `validate_input()`: 입력 데이터 검증
- `process_data()`: 데이터 변환 및 계산
- `aggregate_results()`: 데이터 집계

**utils/file_handler.py**
- `read_csv_file()`: CSV 파일 읽기
- `write_results()`: 결과 CSV 저장
- `list_input_files()`: 입력 파일 목록 조회

## AI 에이전트 테스트 시나리오

### 테스트 목적
AI가 main.py만 보고 전체 애플리케이션을 이해하려면:
1. import 문을 보고 관련 모듈을 찾아야 함
2. 각 모듈을 읽어서 함수 구현을 이해해야 함
3. config 파일을 읽어서 설정값을 파악해야 함

### 기대되는 AI의 동작

1. **초기 분석**
   - main.py를 읽고 import 문 확인
   - `from utils.data_processor import ...` 발견
   - `from utils.file_handler import ...` 발견
   - `from config import ...` 발견

2. **자율적 파일 검색**
   ```xml
   <search_files pattern="*.py" directory="test_agent/utils"></search_files>
   ```
   → data_processor.py, file_handler.py 발견

3. **관련 파일 읽기**
   ```xml
   <read_file path="test_agent/config.py"></read_file>
   <read_file path="test_agent/utils/data_processor.py"></read_file>
   <read_file path="test_agent/utils/file_handler.py"></read_file>
   ```

4. **통합 분석**
   - 모든 파일의 내용을 종합하여 완전한 분석 제공
   - 함수별 상세 설명
   - 데이터 플로우 다이어그램
   - CSV 파일로 정리된 문서 생성

## 테스트 방법

### 1. Streamlit GUI 사용

1. Streamlit 앱 실행:
   ```bash
   uv run streamlit run app.py --server.port 8504
   ```

2. "Upload Task File" 섹션에서 `agent_test_tasks.json` 업로드

3. 생성된 태스크 중 하나를 선택하고 "Run Task" 클릭

4. 결과 확인:
   - `workspaces/<task_name>/outputs/task.log` - 에이전트의 반복 과정
   - `workspaces/<task_name>/outputs/result.txt` - 최종 분석 결과
   - `workspaces/<task_name>/outputs/*.csv` - 생성된 CSV 파일들

### 2. 로그 확인

태스크 실행 중 로그를 확인하면 AI의 사고 과정을 볼 수 있습니다:

```
2025-11-03 23:45:12 - INFO - Agent iteration 1/5
2025-11-03 23:45:13 - INFO - Searching files: pattern=*.py, directory=test_agent
2025-11-03 23:45:14 - INFO - Agent iteration 2/5  
2025-11-03 23:45:14 - INFO - Reading file: test_agent/config.py
2025-11-03 23:45:15 - INFO - Reading file: test_agent/utils/data_processor.py
2025-11-03 23:45:16 - INFO - Agent iteration 3/5
2025-11-03 23:45:16 - INFO - Reading file: test_agent/utils/file_handler.py
2025-11-03 23:45:18 - INFO - Agent completed in 3 iterations
```

## 3가지 테스트 태스크

### 테스트 1: Basic Analysis (agent_test_basic)
- **목표**: AI가 기본적인 파일 읽기를 수행하는지 확인
- **기대 결과**: main.py의 import를 따라가며 관련 파일들을 읽음

### 테스트 2: Detailed Function Analysis (agent_test_detailed)
- **목표**: AI가 모든 함수를 찾아 분석하고 CSV로 정리하는지 확인
- **기대 결과**: 
  - `functions_inventory.csv` 생성
  - `configuration.csv` 생성
  - 완전한 함수 문서

### 테스트 3: File Search and Analysis (agent_test_search)
- **목표**: AI가 프로젝트 구조를 파악하고 체계적으로 분석하는지 확인
- **기대 결과**:
  - `project_structure.csv` 생성
  - `import_dependencies.csv` 생성
  - 프로젝트 구조 다이어그램

## 성공 기준

✅ AI가 main.py만 주어졌을 때 스스로:
1. import된 모듈을 찾아서 읽는다
2. 파일 검색을 통해 관련 파일들을 발견한다
3. 모든 관련 파일을 읽어 완전한 분석을 제공한다
4. 요청된 CSV 파일들을 생성한다
5. 3-5회 반복 안에 완료한다

## 문제 해결

### AI가 파일을 찾지 못하는 경우
- 로그에서 `workspace_dir` 확인
- 상대 경로가 올바른지 확인

### AI가 도구를 사용하지 않는 경우
- 프롬프트가 충분히 복잡한지 확인
- 시스템 인스트럭션이 제대로 전달되는지 확인

### 반복 횟수가 너무 많은 경우
- `max_iterations` 파라미터 조정
- AI 모델의 temperature 조정 (더 낮게)
