<div align="center">
  <img src="generated-image.png" alt="Talos AI Task Management" width="600"/>
  
  # Talos
  
  ### AI-Powered Task Orchestration & Autonomous Agent System
  
  [![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
  
  *An intelligent task orchestrator that reads prompts, autonomously discovers files, and executes complex workflows using conversational LLM agents.*
</div>

---

## 🌟 Features

### 🤖 **Autonomous AI Agent**
- **Self-Directed File Discovery**: AI autonomously searches and reads files when needed
- **Multi-Strategy Path Resolution**: Intelligent file location across workspace and project root
- **Tool-Based Execution**: Uses `<read_file>` and `<search_files>` XML-style tools
- **Iterative Problem Solving**: Up to 100 agent iterations for complex tasks

### 📋 **Advanced Task Management**
- **Three Execution Modes**:
  - **Single**: Individual task execution with full context
  - **Parallel**: CSV-driven batch processing with concurrent execution
  - **Sequential**: Ordered task chains with dependency management
- **Task-Specific Logging**: Dedicated log files per task in output directories
- **Comprehensive LLM I/O Logging**: Full transparency of AI input/output

### 📊 **Data Generation**
- **CSV File Creation**: AI can generate multiple CSV files per task using `<create_csv>` tags
- **Structured Output**: Automatic extraction and saving of structured data
- **Korean Language Support**: Results and outputs in Korean

### 🔍 **Context Management**
- **@ Mention System**: Reference files with `@filename.ext` or `@directory/`
- **Dual Resolution Strategy**: 
  - Workspace-relative for task outputs
  - Project root-relative for shared resources
- **Automatic Context XML**: Generated context files for each execution

### 🎨 **Modern Web Interface**
- **Streamlit-Based GUI**: Beautiful, responsive task management interface
- **Real-Time Monitoring**: Live task status and progress tracking
- **File Upload**: JSON task definition import
- **Multi-Language**: i18n support for Korean/English

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- Google Cloud Vertex AI credentials
- UV package manager (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Brian-LEE0/Talos.git
   cd Talos
   ```

2. **Install dependencies**
   ```bash
   uv pip install -r requirements.txt
   ```
   
   Or with pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Vertex AI credentials**
   
   Place your `vertex-ai-credentials.json` in the project root:
   ```json
   {
     "project_id": "your-project-id",
     "type": "service_account",
     ...
   }
   ```

4. **Run the application**
   ```bash
   streamlit run app.py --server.port 8504
   ```
   
   Or with UV:
   ```bash
   uv run streamlit run app.py --server.port 8504
   ```

---

## 📖 Usage

### 1. Create Task Definitions (JSON)

```json
[
  {
    "name": "Code Analysis Task",
    "type": "single",
    "description": "Analyze Python code structure",
    "prompt": "@test_agent/main.py\n\nAnalyze this file and create documentation including:\n1. Function inventory\n2. Import dependencies\n3. Code structure\n\n모든 응답은 한국어로 작성해주세요.",
    "priority": 1
  }
]
```

### 2. Launch Web Interface

Navigate to `http://localhost:8504` in your browser.

### 3. Upload & Execute

- Upload your JSON task file
- Monitor real-time execution
- View results in `workspaces/{task_name}/outputs/`

### 4. Check Outputs

Each task generates:
- `result.txt` - AI response (CSV tags removed)
- `context.xml` - Referenced files context
- `task.log` - Complete execution log with LLM I/O
- `*.csv` - Generated CSV files (if any)

---

## 🏗️ Project Structure

```
Talos/
├── app.py                          # Streamlit web interface
├── talos/
│   ├── core/
│   │   ├── task_manager.py        # Task orchestration & execution
│   │   ├── vertex_ai_client.py    # AI agent with tool use
│   │   ├── context_manager.py     # @ mention & context handling
│   │   ├── file_manager.py        # File operations
│   │   └── parallel_executor.py   # CSV-based parallel execution
│   ├── logger.py                  # Task-specific logging
│   └── i18n.py                    # Internationalization
├── workspaces/                     # Task execution outputs (gitignored)
│   └── {task_name}_{timestamp}/
│       └── outputs/
│           ├── result.txt
│           ├── context.xml
│           ├── task.log
│           └── *.csv
├── test_agent/                     # Sample test project
│   ├── main.py
│   ├── config.py
│   └── utils/
├── agent_test_tasks.json          # Example task definitions
└── generated-image.png            # Talos signature image
```

---

## 🔧 Configuration

### AI Model Settings

Edit `talos/core/vertex_ai_client.py`:
```python
self.model_name = "gemini-2.5-flash-lite"  # Change model
max_iterations = 100                        # Agent iteration limit
temperature = 0.7                           # Creativity control
```

### Task Manager Settings

Edit `talos/core/task_manager.py`:
```python
base_workspace_dir = "workspaces"  # Change output directory
```

---

## 🤝 AI Agent Tools

The AI agent has access to these tools during execution:

### `<read_file>`
```xml
<read_file path="test_agent/main.py"></read_file>
```
Reads file contents with 4-strategy path resolution.

### `<search_files>`
```xml
<search_files pattern="*.py" directory="test_agent"></search_files>
```
Searches for files matching patterns in directories.

### `<create_csv>`
```xml
<create_csv filename="results.csv">
Name,Value
Item1,100
Item2,200
</create_csv>
```
Creates CSV files from AI-generated content.

---

## 📝 Example Task: Autonomous Analysis

```json
{
  "name": "AI Agent Test - File Search and Analysis",
  "type": "single",
  "description": "AI autonomously finds and analyzes files",
  "prompt": "test_agent 폴더 안에 있는 모든 파이썬 파일을 찾아서 분석하고, 각 파일의 목적과 주요 함수를 정리한 CSV 파일을 생성해주세요.\n\n모든 응답은 한국어로 작성해주세요.",
  "priority": 0
}
```

**What happens:**
1. AI searches for `*.py` files in `test_agent/`
2. Reads discovered files autonomously
3. Analyzes code structure
4. Generates CSV with findings
5. Writes Korean documentation

---

## 🎯 Use Cases

- **Code Analysis**: Autonomous code review and documentation
- **Batch Data Processing**: Parallel CSV-driven workflows
- **Documentation Generation**: Auto-generate docs from codebases
- **File Organization**: Intelligent file discovery and categorization
- **Multi-File Tasks**: Complex operations spanning multiple files

---

## 🐛 Troubleshooting

### Issue: Context.xml is empty
**Solution**: Use `@` mentions in your prompt:
```
@test_agent/main.py

Analyze this file...
```

### Issue: Files not found
**Check:**
1. File exists in project root or workspace
2. Path in prompt is correct
3. Review `task.log` for attempted paths

### Issue: AI doesn't use tools
**Solution**: Be explicit in prompt:
```
"Search for Python files and read them to complete this task..."
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Google Cloud Vertex AI** - LLM backend
- **Streamlit** - Web interface framework
- **Python Community** - Amazing ecosystem

---

<div align="center">
  
  ### Built with ❤️ for autonomous AI task orchestration
  
  **[Report Bug](https://github.com/Brian-LEE0/Talos/issues)** · **[Request Feature](https://github.com/Brian-LEE0/Talos/issues)**
  
</div>
