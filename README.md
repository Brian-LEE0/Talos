<div align="center">
  <img src="generated-image.png" alt="Talos AI Task Management" width="600"/>
  
  # Talos
  
  ### AI-Powered Task Orchestration & Autonomous Agent System
  
  [![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
  [![Version](https://img.shields.io/badge/Version-0.1.0-green.svg)](https://github.com/Brian-LEE0/Talos)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
  
  *An intelligent task orchestrator that reads prompts, autonomously discovers files, and executes complex workflows using conversational LLM agents.*
</div>

---

## 📸 Screenshots

<div align="center">

### Dashboard Overview
<img src="docs/screenshots/dashboard.png" alt="Talos Dashboard" width="800"/>
*System status, task summary, and recent activity at a glance*

### Task Management Interface
<img src="docs/screenshots/tasks.png" alt="Task Management" width="800"/>
*Create, edit, and manage tasks with intuitive UI*

</div>

---

## 🌟 Features

### 🤖 **Autonomous AI Agent**
- **Self-Directed File Discovery**: AI autonomously searches and reads files when needed
- **Multi-Strategy Path Resolution**: Intelligent file location across workspace and project root
- **Advanced Tool System**: 9 powerful tools for file operations and task management
- **Iterative Problem Solving**: Up to 200 configurable agent iterations for complex tasks
- **Explicit Task Completion**: `<finish>` tool for controlled iteration termination
- **Real-time Context Updates**: Automatic context refresh when files are modified

### 📋 **Advanced Task Management**
- **Three Execution Modes**:
  - **Single**: Individual task execution with full context
  - **Parallel**: CSV-driven batch processing with concurrent execution
  - **Sequential**: Ordered task chains with dependency management
- **Task Editing**: Edit task parameters, prompts, and configuration after creation
- **Task-Specific Logging**: Dedicated log files per task in output directories
- **Comprehensive LLM I/O Logging**: Full transparency of AI input/output
- **Smart Task Sorting**: Newest tasks appear first for easy access
- **Per-Task Model Configuration**: Choose different Vertex AI models and regions for each task

### 🎯 **File Operations**
- **Create Files**: Generate new files with custom content
- **Update Files**: Line-based file editing (specify start/end line numbers)
- **Delete Files**: Remove files from workspace
- **Read Files**: Multi-strategy path resolution for file access
- **Search Files**: Pattern-based file discovery across directories
- **Auto Context Sync**: Modified files automatically update in context manager

### 📊 **Model Management**
- **Dynamic Model Discovery**: Automatically fetch available Gemini models from Vertex AI API
- **Model Caching**: Smart 1-hour cache to minimize API calls
- **Manual Refresh**: Force update model list with UI button
- **Future-Proof**: Automatically detects new models (gemini-3.0, gemini-2.5-ultra, etc.)
- **Multi-Region Support**: Choose from 13+ Google Cloud regions
- **Per-Task Configuration**: Set model and region individually for each task

### 🔍 **Context Management**
- **@ Mention System**: Reference files with `@filename.ext` or `@directory/`
- **Visual Mention Detection**: Highlighted mentions with file existence validation
- **Color-Coded Badges**: Green (✅) for existing files, Yellow (⚠️) for missing files
- **Dual Resolution Strategy**: 
  - Workspace-relative for task outputs
  - Project root-relative for shared resources
- **Automatic Context XML**: Generated context files for each execution

### 🎨 **Modern Web Interface**
- **Streamlit-Based GUI**: Beautiful, responsive task management interface
- **Real-Time Monitoring**: Live task status and progress tracking
- **Task Editor**: Inline task editing with all parameters
- **File Upload**: JSON task definition import
- **Multi-Language**: i18n support (English, Korean, Japanese, Chinese)
- **Interactive Mention Preview**: Live preview of @mentions with syntax highlighting
- **Model Refresh**: One-click model list update from UI
- **Type-Safe Architecture**: Pydantic BaseModel for manager type safety and better IDE support

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
    "prompt": "@test_agent/main.py\n\nAnalyze this file and create documentation including:\n1. Function inventory\n2. Import dependencies\n3. Code structure\n\nPlease respond in Korean.",
    "priority": 1,
    "temperature": 0.7,
    "max_tokens": 32000,
    "max_iterations": 100,
    "model_name": "gemini-2.5-flash-lite",
    "location": "us-central1"
  }
]
```

**New Task Parameters:**
- `temperature`: Creativity control (0.0-1.0, default: 0.7)
- `max_tokens`: Maximum output tokens (1000-100000, default: 32000)
- `max_iterations`: Agent iteration limit (1-200, default: 100)
- `model_name`: Vertex AI model to use (default: "gemini-2.5-flash-lite")
- `location`: Google Cloud region (default: "us-central1")
- `top_p`: Nucleus sampling (0.0-1.0, default: 1.0)
- `top_k`: Top-K sampling (1-100, default: 40)

### 2. Launch Web Interface

Navigate to `http://localhost:8504` in your browser.

**Web Interface Features:**
- **Dashboard**: System status, task summary, recent activity
- **Tasks**: Create, edit, run, and manage tasks with type-safe operations
- **Files**: Browse and manage workspace files
- **Context**: Manage file context with @mention processing
- **Monitor**: Performance metrics and system monitoring
- **Language Support**: Switch between English, Korean, Japanese, and Chinese (中文)

### 3. Create Tasks via UI

1. Navigate to **Tasks** tab
2. Click **"🔄 Refresh Models"** to get latest available models (optional)
3. Fill in task details:
   - Name and description
   - Task type (Single/Parallel/Sequential)
   - Priority (0-10)
   - Prompt with @mentions for file references
   - LLM parameters (temperature, max tokens, iterations)
   - Model configuration (model name and region)
4. Click **"Create Task"**

**@Mention System:**
Type `@` followed by a file path in your prompt. The UI will:
- Highlight mentions in green (✅) if file exists
- Show warning in yellow (⚠️) if file doesn't exist
- Display live preview with color-coded mentions

### 4. Edit Tasks (New!)

Click the **"✏️ Edit"** button on any task to modify:
- Task name, description, and priority
- Prompt content
- Temperature and token limits
- Max iterations
- Model name and region
- All LLM parameters

**Note**: Only tasks in PENDING or FAILED status can be edited.

### 5. Execute & Monitor

- Click **"Run"** to execute a task
- Monitor real-time status updates
- View live logs in task details
- Cancel running tasks if needed

### 6. Check Outputs

Each task generates:
- `result.txt` - AI response (cleaned)
- `context.xml` - Referenced files context
- `task.log` - Complete execution log with LLM I/O
- `*.csv` - Generated CSV files (if any)
- Custom files created by AI agent

Output location: `workspaces/{task_name}_{timestamp}/outputs/`

---

## 🏗️ Project Structure

```
Talos/
├── app.py                          # Streamlit web interface (Type-safe with Pydantic)
├── talos/
│   ├── core/
│   │   ├── task_manager.py        # Task orchestration & execution
│   │   ├── vertex_ai_client.py    # AI agent with tool use
│   │   ├── context_manager.py     # @ mention & context handling
│   │   ├── file_manager.py        # File operations
│   │   └── parallel_executor.py   # CSV-based parallel execution
│   ├── i18n/
│   │   ├── en.json                # English translations
│   │   ├── ko.json                # Korean translations
│   │   ├── ja.json                # Japanese translations
│   │   └── zh.json                # Chinese translations (新增)
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
├── requirements.txt               # Python dependencies
└── generated-image.png            # Talos signature image
```

---

## 🔧 Configuration

### AI Model Settings

**Available Models** (Auto-detected from Vertex AI):
- `gemini-2.0-flash-exp` - Latest experimental Flash model
- `gemini-2.5-flash-lite` - Fast, cost-effective (default)
- `gemini-1.5-pro-002` - Most capable production model
- `gemini-1.5-flash-002` - Balanced performance
- `gemini-1.5-pro-001` - Previous pro version
- And more...

**Note**: The system automatically fetches available models from Vertex AI API. New models (like gemini-3.0) will appear automatically!

**Available Regions**:
- `us-central1` (Iowa, USA) - Default
- `us-east4` (Virginia, USA)
- `asia-northeast1` (Tokyo, Japan)
- `asia-northeast3` (Seoul, South Korea)
- `europe-west1` (Belgium)
- And 8 more regions...

**Model Configuration**:
```python
# Per-task configuration (via UI or JSON)
model_name = "gemini-2.5-flash-lite"
location = "us-central1"
temperature = 0.7
max_tokens = 32000
max_iterations = 100
```

**Cache Settings** (in `vertex_ai_client.py`):
```python
_model_cache_ttl = 3600  # Cache models for 1 hour
```

### Task Manager Settings

Edit `talos/core/task_manager.py`:
```python
base_workspace_dir = "workspaces"  # Change output directory
```

---

## 🤝 AI Agent Tools

The AI agent has access to these tools during execution:

### 1. `<read_file>`
```xml
<read_file path="test_agent/main.py"></read_file>
```
Reads file contents with 4-strategy path resolution:
- Relative to workspace directory
- Absolute path
- Relative to current working directory
- Relative to project root

### 2. `<search_files>`
```xml
<search_files pattern="*.py" directory="test_agent"></search_files>
```
Searches for files matching patterns in directories. Supports recursive search and returns up to 50 results.

### 3. `<create_file>`
```xml
<create_file path="output/result.txt">
File content goes here...
</create_file>
```
Creates a new file with specified content. Automatically creates parent directories and updates context manager.

### 4. `<update_file>` ⭐ New!
```xml
<update_file path="test.py" start_line="10" end_line="15">
Updated content for lines 10-15...
Each line will be replaced.
</update_file>
```
Updates specific line ranges in existing files (1-indexed line numbers). Context is automatically refreshed after update.

### 5. `<delete_file>` ⭐ New!
```xml
<delete_file path="temp.txt"></delete_file>
```
Deletes a file from the workspace and removes it from context manager.

### 6. `<create_task>` ⭐ New!
```xml
<create_task name="Sub Task" description="Process data" type="single" priority="5">
Task prompt here...
</create_task>
```
Creates a new sub-task for complex workflows. Useful for breaking down large tasks into smaller, manageable pieces.

### 7. `<finish>` ⭐ New!
```xml
<finish>
Task completed successfully. All files have been processed and outputs generated.
</finish>
```
Explicitly marks the task as complete and stops agent iteration. Use this when the task is done to avoid unnecessary iterations.

**Benefits**:
- Saves API costs by stopping early
- Provides clear completion message
- Prevents over-processing

### 8. `<create_csv>` (Legacy Support)
```xml
<create_csv filename="results.csv">
Name,Value
Item1,100
Item2,200
</create_csv>
```
Creates CSV files from AI-generated content. Still supported for backward compatibility, but consider using `<create_file>` for more flexibility.

---

## 📝 Example Tasks

### Example 1: Autonomous Analysis with Auto-Completion
```json
{
  "name": "Code Analysis with Finish",
  "type": "single",
  "description": "Analyze Python files and explicitly finish",
  "prompt": "Find and analyze all Python files in the test_agent folder and generate documentation.\n\nWhen the task is complete, use the <finish> tag to signal completion.\n\nPlease respond in Korean.",
  "priority": 0,
  "max_iterations": 50,
  "model_name": "gemini-1.5-pro-002"
}
```

**What happens:**
1. AI searches for `*.py` files in `test_agent/`
2. Reads discovered files autonomously
3. Analyzes code structure
4. Generates documentation
5. Uses `<finish>` to signal completion
6. Stops iteration early (saves API costs!)

### Example 2: Multi-File Processing with Line Updates
```json
{
  "name": "Update Multiple Files",
  "type": "single", 
  "description": "Update configuration across multiple files",
  "prompt": "@config/settings.py\n@config/database.py\n\nChange the DEBUG setting to False and update the logging level to INFO in these files.\n\nModify only the relevant lines in each file.",
  "temperature": 0.3,
  "model_name": "gemini-2.5-flash-lite",
  "location": "asia-northeast3"
}
```

**Features demonstrated:**
- Multiple @mentions for context
- Line-based file updates
- Low temperature for precision
- Korean region for lower latency

### Example 3: Task Decomposition
```json
{
  "name": "Complex Analysis Pipeline",
  "type": "single",
  "description": "Break down analysis into sub-tasks",
  "prompt": "Analyze the large codebase and create the following sub-tasks:\n\n1. Dependency analysis task\n2. Code complexity analysis task\n3. Security inspection task\n\nCreate each task using <create_task>, and when all tasks are created, terminate with <finish>.",
  "max_iterations": 20
}
```

**Advanced features:**
- Automatic task decomposition
- Sub-task creation with `<create_task>`
- Explicit completion with `<finish>`

---

## 🎯 Use Cases

- **Code Analysis & Documentation**: Autonomous code review with multi-file analysis
- **Batch Data Processing**: Parallel CSV-driven workflows with concurrent execution
- **Configuration Updates**: Line-based updates across multiple configuration files
- **File Organization**: Intelligent file discovery, categorization, and restructuring
- **Multi-File Refactoring**: Complex operations spanning multiple files with context awareness
- **Task Decomposition**: Break down large tasks into manageable sub-tasks automatically
- **API Cost Optimization**: Use `<finish>` to stop iteration early and save costs
- **Multi-Region Deployment**: Leverage different regions for latency optimization
- **Model Comparison**: Test different models (Flash vs Pro) for quality/speed tradeoffs

---

## 🐛 Troubleshooting

### Issue: "Model not available" error
**Solution**: 
1. Click **"🔄 Refresh Models"** in the UI to update model list
2. Check your Vertex AI API permissions
3. Try a different region if model is region-specific
4. Use fallback model: `gemini-2.5-flash-lite`

### Issue: Context.xml is empty
**Solution**: Use `@` mentions in your prompt:
```
@test_agent/main.py
@config/settings.py

Analyze these files...
```
The UI will show color-coded badges for validation.

### Issue: Files not found
**Check:**
1. File exists in project root or workspace
2. Path in prompt is correct (check green ✅ badges in UI)
3. Review `task.log` for attempted paths
4. Try absolute paths if relative paths fail

### Issue: AI doesn't use tools
**Solution**: Be explicit in prompt:
```
"Search for Python files using <search_files> and read them with <read_file> to complete this task..."
```

### Issue: Task doesn't stop iterating
**Solution**: 
1. Instruct AI to use `<finish>` tag when done
2. Lower `max_iterations` parameter
3. Be more specific about completion criteria

### Issue: "Can't use st.button() in st.form()" error
**Fixed**: Model refresh button is now outside the form. Update to latest version.

### Issue: File update failed
**Check:**
1. Line numbers are valid (1-indexed)
2. `end_line` is not greater than file length
3. File permissions allow writing
4. File exists (use `<read_file>` first to verify)

### Issue: Cache not refreshing
**Solution**:
1. Wait 1 hour for automatic cache expiry
2. Use **"🔄 Refresh Models"** button for immediate refresh
3. Restart application to clear all caches

---

## 🆕 What's New

### Version 0.1.0 (Current)

**🎨 UI Enhancements:**
- ✏️ **Task Editing**: Edit tasks after creation with inline forms
- 🔄 **Model Refresh**: One-click model list update from UI
- 📎 **@Mention Highlighting**: Visual preview with color-coded file existence validation
- 📊 **Enhanced Task Details**: Model and region information in task cards
- 🌐 **Chinese Language Support**: Added Simplified Chinese (中文) translation
- 🔒 **Type-Safe Architecture**: Pydantic BaseModel for SystemManagers with full type hints

**🤖 Agent Improvements:**
- 🎯 **Finish Tool**: Explicitly mark task completion and stop iteration
- 📝 **File Update Tool**: Line-based file editing with 1-indexed line numbers
- 🗑️ **File Delete Tool**: Remove files from workspace
- 🔄 **Auto Context Sync**: Modified files automatically refresh in context
- 🏗️ **Task Creation Tool**: Create sub-tasks for complex workflows

**⚙️ Configuration:**
- 🌍 **Per-Task Models**: Set different models and regions for each task
- 📋 **Dynamic Model Discovery**: Auto-fetch available models from Vertex AI API
- 💾 **Smart Caching**: 1-hour model cache to minimize API calls
- 🔮 **Future-Proof**: Automatically detects new models (gemini-3.0, etc.)

**🎛️ Task Parameters:**
- Temperature, max tokens, max iterations
- Top-P and Top-K sampling controls
- Model name and region selection
- Priority-based execution

**📈 Performance:**
- Up to 200 configurable iterations (vs 100 before)
- Newest-first task sorting
- Optimized file operations with multi-strategy resolution

**🔧 Code Quality:**
- Type hints on all manager parameters (`managers: SystemManagers`)
- Better IDE autocomplete and type checking
- Cleaner attribute access (`managers.task_manager` vs `managers['task_manager']`)
- Pydantic validation for manager initialization

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
