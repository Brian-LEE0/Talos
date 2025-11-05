"""
Talos Web Application
Streamlit-based web interface for AI-powered task management system
Version: 0.1.0
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
import json
import time
import nest_asyncio
from pydantic import BaseModel, ConfigDict
from typing import Optional

nest_asyncio.apply()

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from talos.core.task_manager import get_task_manager, TaskType, TaskStatus
from talos.core.context_manager import get_context_manager
from talos.core.file_manager import get_file_manager
from talos.core.vertex_ai_client import get_vertex_client
from talos.core.parallel_executor import get_parallel_executor
from talos.i18n import get_i18n, t
from talos.logger import logger

# Page configuration
st.set_page_config(
    page_title="Talos - AI Task Management",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SystemManagers(BaseModel):
    """System managers container using Pydantic BaseModel"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    task_manager: object
    context_manager: object
    file_manager: object
    parallel_executor: object
    vertex_client: object

# Initialize managers
@st.cache_resource
def initialize_managers() -> SystemManagers:
    """Initialize all system managers"""
    return SystemManagers(
        task_manager=get_task_manager(),
        context_manager=get_context_manager(),
        file_manager=get_file_manager(),
        parallel_executor=get_parallel_executor(),
        vertex_client=get_vertex_client()
    )

def initialize_i18n():
    """Initialize internationalization"""
    if 'language' not in st.session_state:
        st.session_state.language = 'en'
    return get_i18n(st.session_state.language)

def language_selector():
    """Language selection sidebar"""
    languages = {
        'en': '🇺🇸 English',
        'ko': '🇰🇷 한국어', 
        'ja': '🇯🇵 日本語',
        'zh': '🇨🇳 中文'
    }
    
    selected = st.sidebar.selectbox(
        "Language / 언어 / 言語 / 语言",
        options=list(languages.keys()),
        format_func=lambda x: languages[x],
        index=list(languages.keys()).index(st.session_state.get('language', 'en'))
    )
    
    if selected != st.session_state.get('language', 'en'):
        st.session_state.language = selected
        logger.info(f"Language changed to {selected}")
        st.rerun()

def main():
    """Main application"""
    # Initialize i18n
    i18n = initialize_i18n()
    
    # Language selector
    language_selector()
    
    # Initialize managers
    managers = initialize_managers()
    
    # Main title
    st.title(t('app.title'))
    st.markdown(t('app.description'))
    
    # Version info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Talos v0.1.0**")
    st.sidebar.caption("AI Task Orchestration System")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    tab_names = [
        t('ui.tabs.dashboard'),
        t('ui.tabs.tasks'),
        t('ui.tabs.files'),
        t('ui.tabs.context'),
        t('ui.tabs.monitor')
    ]
    
    selected_tab = st.sidebar.radio("Navigation", tab_names, label_visibility="collapsed")
    
    # Tab content
    if selected_tab == t('ui.tabs.dashboard'):
        show_dashboard(managers)
    elif selected_tab == t('ui.tabs.tasks'):
        show_tasks(managers)
    elif selected_tab == t('ui.tabs.files'):
        show_files(managers)
    elif selected_tab == t('ui.tabs.context'):
        show_context(managers)
    elif selected_tab == t('ui.tabs.monitor'):
        show_monitor(managers)

def highlight_mentions(text: str) -> str:
    """
    Highlight @mentions in text with HTML/CSS formatting.
    
    Args:
        text: Input text with potential @mentions
        
    Returns:
        HTML formatted text with highlighted mentions
    """
    import re
    
    def replace_mention(match):
        mention = match.group(0)
        file_path = mention[1:]  # Remove @ prefix
        
        # Check if file exists
        if os.path.exists(file_path):
            # Green background for existing files
            return f'<span style="background-color: #d4edda; color: #155724; padding: 2px 6px; border-radius: 3px; font-weight: 600;">{mention}</span>'
        else:
            # Yellow background for non-existing files
            return f'<span style="background-color: #fff3cd; color: #856404; padding: 2px 6px; border-radius: 3px; font-weight: 600;">{mention}</span>'
    
    # Replace all @mentions with highlighted version
    highlighted = re.sub(r'@[\w\-\.\/\\]+', replace_mention, text)
    
    # Preserve line breaks
    highlighted = highlighted.replace('\n', '<br>')
    
    return highlighted

def show_dashboard(managers: SystemManagers):
    """Dashboard tab"""
    st.header(t('ui.dashboard.title'))
    
    # Credentials & Settings Section
    with st.expander("⚙️ Vertex AI Configuration", expanded=False):
        st.markdown("### 🔑 Credentials Management")
        
        # Show current credentials info
        creds_info = managers.vertex_client.get_credentials_info()
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info(f"**Project ID:** {creds_info.get('project_id', 'Not set')}")
            st.info(f"**Credentials File:** `{creds_info.get('credentials_path', 'N/A')}`")
        with col_info2:
            status = "✅ Found" if creds_info.get('credentials_exists') else "❌ Not Found"
            st.info(f"**Status:** {status}")
        
        st.markdown("---")
        
        # Upload new credentials
        st.markdown("### 📤 Upload New Credentials")
        uploaded_creds = st.file_uploader(
            "Upload service account JSON file",
            type=['json'],
            help="Upload your Google Cloud service account credentials JSON file",
            key="creds_uploader"
        )
        
        if uploaded_creds is not None:
            try:
                creds_content = uploaded_creds.read().decode('utf-8')
                
                # Show preview
                creds_data = json.loads(creds_content)
                st.success(f"✅ Valid credentials for project: **{creds_data.get('project_id', 'Unknown')}**")
                
                # Save button
                if st.button("💾 Save and Apply Credentials", type="primary"):
                    if managers.vertex_client.update_credentials(creds_content):
                        st.success("🎉 Credentials updated successfully! Reinitializing client...")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("❌ Failed to update credentials. Check logs for details.")
                        
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON file. Please upload a valid service account credentials file.")
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
        
        st.markdown("---")
        
        # Default settings
        st.markdown("### 🎛️ Default Settings")
        
        settings_col1, settings_col2 = st.columns(2)
        
        with settings_col1:
            available_locations = managers.vertex_client.list_available_locations()
            current_location = creds_info.get('location', 'us-central1')
            
            new_location = st.selectbox(
                "Default Region",
                options=available_locations,
                index=available_locations.index(current_location) if current_location in available_locations else 0,
                help="Default Google Cloud region for API calls",
                key="default_location"
            )
        
        with settings_col2:
            available_models = managers.vertex_client.list_available_models()
            current_model = creds_info.get('model_name', 'gemini-2.5-flash-lite')
            
            new_model = st.selectbox(
                "Default Model",
                options=available_models,
                index=available_models.index(current_model) if current_model in available_models else 1,
                help="Default Vertex AI model for new tasks",
                key="default_model"
            )
        
        if st.button("💾 Update Default Settings"):
            if managers.vertex_client.update_default_settings(location=new_location, model_name=new_model):
                st.success("✅ Default settings updated successfully!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Failed to update settings. Check logs for details.")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(t('ui.dashboard.system_status'))
        
        # AI Status
        try:
            ai_client = get_vertex_client()
            ai_info = ai_client.get_model_info()
            
            st.success(f"**{t('ui.dashboard.ai_status')}:** {t('ui.dashboard.connected')}")
            st.write(f"**{t('ui.dashboard.model')}:** {ai_info.get('model_name', 'Unknown')}")
            st.write(f"**{t('ui.dashboard.project')}:** {ai_info.get('project_id', 'Unknown')}")
        except:
            st.error(f"**{t('ui.dashboard.ai_status')}:** {t('ui.dashboard.disconnected')}")
        
        # Task Summary
        st.subheader(t('ui.dashboard.task_summary'))
        tasks = managers.task_manager.list_tasks()
        
        if tasks:
            task_counts = {
                'pending': len([t for t in tasks if t.status == TaskStatus.PENDING]),
                'running': len([t for t in tasks if t.status == TaskStatus.RUNNING]),
                'completed': len([t for t in tasks if t.status == TaskStatus.COMPLETED]),
                'failed': len([t for t in tasks if t.status == TaskStatus.FAILED])
            }
            
            st.metric(t('ui.dashboard.total_tasks'), len(tasks))
            
            col_p, col_r, col_c, col_f = st.columns(4)
            with col_p:
                st.metric(t('ui.dashboard.pending'), task_counts['pending'])
            with col_r:
                st.metric(t('ui.dashboard.running'), task_counts['running'])
            with col_c:
                st.metric(t('ui.dashboard.completed'), task_counts['completed'])
            with col_f:
                st.metric(t('ui.dashboard.failed'), task_counts['failed'])
        else:
            st.info(t('ui.tasks.no_tasks'))
    
    with col2:
        st.subheader(t('ui.dashboard.context_info'))
        context_info = managers.context_manager.get_context_info()
        
        st.metric(t('ui.dashboard.files_loaded'), context_info['file_count'])
        st.metric(t('ui.dashboard.total_tokens'), f"{context_info['total_tokens']:,}")
        
        # Recent Activity
        st.subheader(t('ui.dashboard.recent_activity'))
        if tasks:
            recent_tasks = sorted(tasks, key=lambda x: x.created_time, reverse=True)[:5]
            for task in recent_tasks:
                status_emoji = {
                    TaskStatus.PENDING: '⏸️',
                    TaskStatus.RUNNING: '🔄',
                    TaskStatus.COMPLETED: '✅',
                    TaskStatus.FAILED: '❌',
                    TaskStatus.CANCELLED: '⏹️'
                }.get(task.status, '❓')
                
                st.write(f"{status_emoji} {task.name} - {task.status.value}")
        else:
            st.info(t('ui.dashboard.no_activity'))

def show_tasks(managers: SystemManagers):
    """Tasks tab"""
    st.header(t('ui.tasks.title'))
    
    # Load tasks from file
    with st.expander(t('ui.tasks.load_from_file'), expanded=False):
        uploaded_task_file = st.file_uploader(
            t('ui.tasks.upload_json'),
            type=['json']
        )
        if uploaded_task_file is not None:
            # Process the file only if it's a new upload
            if 'processed_file_id' not in st.session_state or st.session_state.processed_file_id != uploaded_task_file.file_id:
                try:
                    # To use the path, we need to save it temporarily
                    file_path = f"temp_{uploaded_task_file.name}"
                    with open(file_path, "wb") as f:
                        f.write(uploaded_task_file.getbuffer())

                    tasks_loaded = managers.task_manager.load_tasks_from_file(file_path)
                    os.remove(file_path) # Clean up temp file

                    st.session_state.processed_file_id = uploaded_task_file.file_id
                    st.success(t('ui.tasks.load_success', count=len(tasks_loaded)))
                    # Wait a bit before rerunning to ensure user sees the message
                    time.sleep(2)
                    st.rerun()

                except Exception as e:
                    st.error(f"Error loading tasks: {e}")
                    # Reset on error to allow re-uploading the same file
                    if 'processed_file_id' in st.session_state:
                        del st.session_state['processed_file_id']

    # Task creation form
    with st.expander(t('ui.tasks.create'), expanded=True):
        # Model refresh button (outside form)
        refresh_col1, refresh_col2 = st.columns([6, 1])
        with refresh_col2:
            if st.button("🔄 Refresh Models", help="Refresh model list from Vertex AI API", key="refresh_models_create"):
                with st.spinner("Refreshing models..."):
                    try:
                        new_models = managers.vertex_client.refresh_model_cache()
                        st.success(f"Found {len(new_models)} models")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to refresh: {e}")
        
        with st.form("create_task"):
            col1, col2 = st.columns(2)
            
            with col1:
                task_name = st.text_input(t('ui.tasks.name'), placeholder=t('ui.tasks.name_placeholder'))
                task_description = st.text_area(t('ui.tasks.description'), placeholder=t('ui.tasks.description_placeholder'))
            
            with col2:
                task_type = st.selectbox(
                    t('ui.tasks.type'),
                    options=[TaskType.SINGLE, TaskType.PARALLEL, TaskType.SEQUENTIAL],
                    format_func=lambda x: t(f'ui.tasks.{x.value}')
                )
                priority = st.slider(t('ui.tasks.priority'), 0, 10, 0)
            
            prompt = st.text_area(
                t('ui.tasks.prompt'),
                placeholder=t('ui.tasks.prompt_placeholder'),
                height=150,
                key="task_prompt_input"
            )
            
            # Detect and display mentions with visual preview
            if prompt:
                import re
                mentions = re.findall(r'@[\w\-\.\/\\]+', prompt)
                if mentions:
                    st.markdown("**📎 Detected Mentions:**")
                    
                    # Show highlighted preview
                    st.markdown("**Preview with highlights:**")
                    highlighted_prompt = highlight_mentions(prompt)
                    st.markdown(
                        f'<div style="background-color: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 4px solid #007bff; max-height: 150px; overflow-y: auto;">{highlighted_prompt}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Show mention status badges
                    st.markdown("**Mention Status:**")
                    mention_cols = st.columns(min(len(mentions), 5))
                    for idx, mention in enumerate(mentions):
                        with mention_cols[idx % 5]:
                            file_path = mention[1:]  # Remove @ prefix
                            if os.path.exists(file_path):
                                st.markdown(f'<span style="background-color: #d4edda; color: #155724; padding: 4px 8px; border-radius: 4px; font-size: 12px; display: inline-block; margin: 2px;">✅ {mention}</span>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<span style="background-color: #fff3cd; color: #856404; padding: 4px 8px; border-radius: 4px; font-size: 12px; display: inline-block; margin: 2px;">⚠️ {mention}</span>', unsafe_allow_html=True)
            
            # LLM Parameters
            st.markdown("**LLM Parameters**")
            llm_col1, llm_col2, llm_col3 = st.columns(3)
            
            with llm_col1:
                temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, 
                                      help="Controls randomness: 0=deterministic, 1=creative")
                max_tokens = st.number_input("Max Output Tokens", 1000, 100000, 32000, 1000,
                                            help="Maximum number of tokens in output")
            
            with llm_col2:
                top_p = st.slider("Top-P", 0.0, 1.0, 1.0, 0.05,
                                help="Nucleus sampling: considers top P probability mass")
                top_k = st.number_input("Top-K", 1, 100, 40, 1,
                                       help="Considers only top K most likely tokens")
            
            with llm_col3:
                max_iterations = st.number_input("Max Agent Iterations", 1, 200, 100, 10,
                                                help="Maximum iterations for agent tool use")
            
            # Model Configuration
            st.markdown("**Model Configuration**")
            model_col1, model_col2 = st.columns(2)
            
            # Get available models and locations from vertex client
            available_models = managers.vertex_client.list_available_models()
            available_locations = managers.vertex_client.list_available_locations()
            
            with model_col1:
                model_name = st.selectbox("Model", 
                                         options=available_models,
                                         index=1,  # Default to gemini-2.5-flash-lite
                                         help="Vertex AI model to use for this task")
            
            with model_col2:
                location = st.selectbox("Region",
                                       options=available_locations,
                                       index=0,  # Default to us-central1
                                       help="Google Cloud region for API calls")
            
            if st.form_submit_button(t('ui.tasks.create_button')):
                if task_name and prompt:
                    try:
                        task = managers.task_manager.create_task(
                            name=task_name,
                            prompt=prompt,
                            description=task_description,
                            task_type=task_type,
                            priority=priority,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            top_k=top_k,
                            max_iterations=max_iterations,
                            model_name=model_name,
                            location=location
                        )
                        
                        st.success(t('ui.tasks.created_success'))
                        st.info(t('ui.tasks.created_id', task_id=task.task_id))
                        st.info(t('ui.tasks.created_workspace', workspace=task.workspace_dir))
                        
                    except Exception as e:
                        st.error(f"Error creating task: {e}")
                else:
                    st.error("Please provide task name and prompt")
    
    # Task list
    st.subheader(t('ui.tasks.list'))
    tasks = managers.task_manager.list_tasks()
    
    if tasks:
        for task in tasks:
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 2])
                
                with col1:
                    status_emoji = {
                        TaskStatus.PENDING: '⏸️',
                        TaskStatus.RUNNING: '🔄', 
                        TaskStatus.COMPLETED: '✅',
                        TaskStatus.FAILED: '❌',
                        TaskStatus.CANCELLED: '⏹️'
                    }.get(task.status, '❓')
                    
                    st.write(f"**{status_emoji} {task.name}**")
                    st.write(f"ID: `{task.task_id[:8]}...`")
                    if task.description:
                        st.write(f"Description: {task.description}")
                
                with col2:
                    st.write(f"**{t('ui.tasks.status')}:** {task.status.value}")
                    st.write(f"**{t('ui.tasks.type')}:** {task.task_type.value}")
                    st.write(f"**{t('ui.tasks.priority')}:** {task.priority}")
                    st.write(f"**LLM Temp:** {task.temperature}")
                
                with col3:
                    st.write(f"**{t('ui.tasks.actions')}:**")
                    
                    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
                    
                    # Check if task is editable (only PENDING or FAILED)
                    is_editable = task.status in [TaskStatus.PENDING, TaskStatus.FAILED]
                    
                    with action_col1:
                        if st.button("▶️ Run", key=f"run_{task.task_id}", use_container_width=True):
                            # Show execution container
                            st.session_state[f'running_{task.task_id}'] = True
                            st.rerun()
                    
                    with action_col2:
                        if st.button("✏️ Edit", key=f"edit_{task.task_id}", use_container_width=True, disabled=not is_editable):
                            if is_editable:
                                st.session_state[f'editing_{task.task_id}'] = True
                                st.rerun()
                    
                    with action_col3:
                        if st.button("⏹️ Cancel", key=f"cancel_{task.task_id}", use_container_width=True):
                            if managers.task_manager.cancel_task(task.task_id):
                                st.success(t('ui.tasks.cancel_success'))
                                st.rerun()
                    
                    with action_col4:
                        if st.button("🗑️ Delete", key=f"delete_{task.task_id}", use_container_width=True):
                            if managers.task_manager.delete_task(task.task_id, delete_workspace=True):
                                st.success(t('ui.tasks.delete_success'))
                                st.rerun()
                
                # Real-time execution streaming
                if st.session_state.get(f'running_{task.task_id}', False):
                    st.markdown("---")
                    
                    with st.status("🔄 Executing task...", expanded=True) as status:
                        st.write("� Task execution started...")
                        
                        # Create placeholders
                        progress_text = st.empty()
                        output_container = st.container()
                        
                        # Execute task in background
                        import threading
                        import queue
                        
                        output_queue = queue.Queue()
                        result_container = {'result': None, 'done': False}
                        
                        def execute_and_monitor():
                            """Execute task and monitor logs"""
                            try:
                                # Start execution
                                result = managers.task_manager.execute_task(task.task_id)
                                result_container['result'] = result
                                result_container['done'] = True
                            except Exception as e:
                                logger.error(f"Execution error: {e}", exc_info=True)
                                result_container['result'] = False
                                result_container['done'] = True
                        
                        # Start execution thread
                        exec_thread = threading.Thread(target=execute_and_monitor, daemon=True)
                        exec_thread.start()
                        
                        # Monitor log file
                        if task.output_dir:
                            log_file = os.path.join(task.output_dir, "task.log")
                            last_position = 0
                            iteration_count = 0
                            all_outputs = []
                            
                            import time
                            while not result_container['done']:
                                if os.path.exists(log_file):
                                    try:
                                        with open(log_file, 'r', encoding='utf-8') as f:
                                            f.seek(last_position)
                                            new_content = f.read()
                                            last_position = f.tell()
                                            
                                            if new_content:
                                                # Extract iteration info
                                                if "AGENT ITERATION" in new_content:
                                                    import re
                                                    iter_match = re.search(r'AGENT ITERATION (\d+)/(\d+)', new_content)
                                                    if iter_match:
                                                        iteration_count = int(iter_match.group(1))
                                                        max_iter = int(iter_match.group(2))
                                                        progress_text.info(f"🔄 Iteration {iteration_count}/{max_iter}")
                                                
                                                # Extract LLM output
                                                if "LLM OUTPUT - START" in new_content:
                                                    lines = new_content.split('\n')
                                                    output_lines = []
                                                    capturing = False
                                                    
                                                    for line in lines:
                                                        if "LLM OUTPUT - START" in line:
                                                            capturing = True
                                                            continue
                                                        elif "LLM OUTPUT - END" in line:
                                                            capturing = False
                                                            if output_lines:
                                                                all_outputs.append('\n'.join(output_lines))
                                                            output_lines = []
                                                            continue
                                                        elif capturing:
                                                            if not line.startswith('=') and line.strip():
                                                                if not line.startswith('Response:'):
                                                                    output_lines.append(line)
                                                    
                                                    # Display latest outputs
                                                    if all_outputs:
                                                        with output_container:
                                                            for idx, output in enumerate(all_outputs[-3:], 1):  # Show last 3 outputs
                                                                with st.expander(f"💬 Response {len(all_outputs) - 3 + idx}", expanded=(idx == len(all_outputs[-3:]))):
                                                                    st.code(output, language=None)
                                    except Exception as e:
                                        logger.debug(f"Log reading error: {e}")
                                
                                time.sleep(0.3)  # Check every 300ms
                            
                            # Wait for thread to finish
                            exec_thread.join(timeout=5)
                        
                        # Update status
                        st.session_state[f'running_{task.task_id}'] = False
                        
                        if result_container['result']:
                            status.update(label="✅ Task completed!", state="complete", expanded=False)
                            st.success("Task execution finished successfully!")
                        else:
                            status.update(label="❌ Task failed!", state="error", expanded=True)
                            st.error("Task execution encountered an error. Check the logs for details.")
                        
                        # Auto-refresh after delay
                        time.sleep(3)
                        st.rerun()
                
                # Edit form (if editing)
                if st.session_state.get(f'editing_{task.task_id}', False):
                    with st.form(f"edit_form_{task.task_id}"):
                        st.markdown("### 📝 Edit Task")
                        
                        edit_col1, edit_col2 = st.columns(2)
                        
                        with edit_col1:
                            new_name = st.text_input("Task Name", value=task.name, key=f"edit_name_{task.task_id}")
                            new_description = st.text_area("Description", value=task.description, key=f"edit_desc_{task.task_id}")
                            new_priority = st.slider("Priority", 0, 10, task.priority, key=f"edit_priority_{task.task_id}")
                        
                        with edit_col2:
                            new_temp = st.slider("Temperature", 0.0, 1.0, task.temperature, 0.1, key=f"edit_temp_{task.task_id}")
                            new_max_tokens = st.number_input("Max Tokens", 1000, 100000, task.max_tokens, 1000, key=f"edit_tokens_{task.task_id}")
                            new_max_iter = st.number_input("Max Iterations", 1, 200, task.max_iterations, 10, key=f"edit_iter_{task.task_id}")
                        
                        # Model Configuration
                        st.markdown("**Model Configuration**")
                        model_edit_col1, model_edit_col2 = st.columns(2)
                        
                        # Get available models and locations
                        available_models = managers.vertex_client.list_available_models()
                        available_locations = managers.vertex_client.list_available_locations()
                        
                        with model_edit_col1:
                            # Find current model index, fallback to default if not found
                            try:
                                current_model_index = available_models.index(task.model_name)
                            except (ValueError, AttributeError):
                                current_model_index = 1  # Default to gemini-2.5-flash-lite
                            
                            new_model = st.selectbox("Model", 
                                                    options=available_models,
                                                    index=current_model_index,
                                                    key=f"edit_model_{task.task_id}")
                        
                        with model_edit_col2:
                            # Find current location index, fallback to default if not found
                            try:
                                current_location_index = available_locations.index(task.location)
                            except (ValueError, AttributeError):
                                current_location_index = 0  # Default to us-central1
                            
                            new_location = st.selectbox("Region",
                                                       options=available_locations,
                                                       index=current_location_index,
                                                       key=f"edit_location_{task.task_id}")
                        
                        new_prompt = st.text_area("Prompt", value=task.prompt, height=200, key=f"edit_prompt_{task.task_id}")
                        
                        # Detect and display mentions in edit form
                        if new_prompt:
                            import re
                            mentions = re.findall(r'@[\w\-\.\/\\]+', new_prompt)
                            if mentions:
                                st.markdown("**📎 Detected Mentions:**")
                                
                                # Show highlighted preview
                                st.markdown("**Preview with highlights:**")
                                highlighted_prompt = highlight_mentions(new_prompt)
                                st.markdown(
                                    f'<div style="background-color: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 4px solid #007bff; max-height: 120px; overflow-y: auto;">{highlighted_prompt}</div>',
                                    unsafe_allow_html=True
                                )
                                
                                # Show mention status badges
                                st.markdown("**Mention Status:**")
                                mention_cols = st.columns(min(len(mentions), 5))
                                for idx, mention in enumerate(mentions):
                                    with mention_cols[idx % 5]:
                                        file_path = mention[1:]  # Remove @ prefix
                                        if os.path.exists(file_path):
                                            st.markdown(f'<span style="background-color: #d4edda; color: #155724; padding: 4px 8px; border-radius: 4px; font-size: 12px; display: inline-block; margin: 2px;">✅ {mention}</span>', unsafe_allow_html=True)
                                        else:
                                            st.markdown(f'<span style="background-color: #fff3cd; color: #856404; padding: 4px 8px; border-radius: 4px; font-size: 12px; display: inline-block; margin: 2px;">⚠️ {mention}</span>', unsafe_allow_html=True)
                        
                        form_col1, form_col2 = st.columns(2)
                        with form_col1:
                            if st.form_submit_button("💾 Save Changes"):
                                if managers.task_manager.update_task(
                                    task.task_id,
                                    name=new_name,
                                    description=new_description,
                                    prompt=new_prompt,
                                    priority=new_priority,
                                    temperature=new_temp,
                                    max_tokens=new_max_tokens,
                                    max_iterations=new_max_iter,
                                    model_name=new_model,
                                    location=new_location
                                ):
                                    st.success("Task updated successfully!")
                                    st.session_state[f'editing_{task.task_id}'] = False
                                    st.rerun()
                                else:
                                    st.error("Failed to update task. Check if task is in editable state (PENDING or FAILED).")
                        
                        with form_col2:
                            if st.form_submit_button("❌ Cancel"):
                                st.session_state[f'editing_{task.task_id}'] = False
                                st.rerun()
                
                # Task details expander
                with st.expander(f"📊 Details & Outputs - {task.name}", expanded=False):
                    detail_tab1, detail_tab2, detail_tab3 = st.tabs(["⚙️ Configuration", "📄 Logs & Files", "📈 Steps"])
                    
                    with detail_tab1:
                        st.markdown("**LLM Parameters:**")
                        param_col1, param_col2 = st.columns(2)
                        with param_col1:
                            st.write(f"• Temperature: `{task.temperature}`")
                            st.write(f"• Max Tokens: `{task.max_tokens}`")
                            st.write(f"• Max Iterations: `{task.max_iterations}`")
                        with param_col2:
                            st.write(f"• Top-P: `{task.top_p}`")
                            st.write(f"• Top-K: `{task.top_k}`")
                        
                        st.markdown("**Model Configuration:**")
                        model_info_col1, model_info_col2 = st.columns(2)
                        with model_info_col1:
                            st.write(f"• Model: `{task.model_name}`")
                        with model_info_col2:
                            st.write(f"• Region: `{task.location}`")
                        
                        st.markdown("**Directories:**")
                        st.write(f"• Workspace: `{task.workspace_dir}`")
                        st.write(f"• Output: `{task.output_dir}`")
                    
                    with detail_tab2:
                        # Log file viewer
                        if task.output_dir and os.path.exists(task.output_dir):
                            log_file = os.path.join(task.output_dir, "task.log")
                            if os.path.exists(log_file):
                                st.markdown("**📝 Task Log:**")
                                with open(log_file, 'r', encoding='utf-8') as f:
                                    log_content = f.read()
                                st.text_area("Log Content", log_content, height=300, key=f"log_{task.task_id}")
                                
                                # Download button
                                st.download_button(
                                    "⬇️ Download Log",
                                    log_content,
                                    file_name=f"{task.name}_task.log",
                                    mime="text/plain",
                                    key=f"dl_log_{task.task_id}"
                                )
                            
                            # Output files
                            st.markdown("**📁 Output Files:**")
                            output_files = []
                            if os.path.exists(task.output_dir):
                                for file in os.listdir(task.output_dir):
                                    file_path = os.path.join(task.output_dir, file)
                                    if os.path.isfile(file_path):
                                        output_files.append((file, file_path))
                            
                            if output_files:
                                for filename, filepath in output_files:
                                    file_col1, file_col2 = st.columns([3, 1])
                                    with file_col1:
                                        st.write(f"📄 `{filename}`")
                                    with file_col2:
                                        try:
                                            with open(filepath, 'r', encoding='utf-8') as f:
                                                file_content = f.read()
                                            st.download_button(
                                                "⬇️",
                                                file_content,
                                                file_name=filename,
                                                key=f"dl_{task.task_id}_{filename}"
                                            )
                                        except:
                                            st.write("Binary file")
                                
                                # File viewer
                                selected_file = st.selectbox(
                                    "View file content:",
                                    options=[f[0] for f in output_files],
                                    key=f"view_{task.task_id}"
                                )
                                if selected_file:
                                    file_path = next(fp for fn, fp in output_files if fn == selected_file)
                                    try:
                                        with open(file_path, 'r', encoding='utf-8') as f:
                                            content = f.read()
                                        st.text_area(f"Content of {selected_file}", content, height=200, key=f"content_{task.task_id}_{selected_file}")
                                    except Exception as e:
                                        st.error(f"Cannot read file: {e}")
                            else:
                                st.info("No output files yet")
                        else:
                            st.info("No output directory found")
                    
                    with detail_tab3:
                        # Task steps
                        if task.steps:
                            for step in task.steps:
                                step_emoji = {
                                    TaskStatus.PENDING: '⏸️',
                                    TaskStatus.RUNNING: '🔄',
                                    TaskStatus.COMPLETED: '✅',
                                    TaskStatus.FAILED: '❌'
                                }.get(step.status, '❓')
                                
                                step_col1, step_col2 = st.columns([3, 1])
                                with step_col1:
                                    st.write(f"{step_emoji} **{step.description}**")
                                with step_col2:
                                    if step.start_time and step.end_time:
                                        duration = step.end_time - step.start_time
                                        st.write(f"⏱️ {duration:.2f}s")
                        else:
                            st.info("No steps available")
                
                st.divider()
    else:
        st.info(t('ui.tasks.no_tasks'))

def show_files(managers: SystemManagers):
    """Files tab"""
    st.header(t('ui.files.title'))
    
    # File operations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(t('ui.files.new_file')):
            st.session_state.show_create_file = True
    
    with col2:
        uploaded_file = st.file_uploader(t('ui.files.upload'))
        if uploaded_file:
            try:
                content = uploaded_file.read().decode('utf-8')
                managers.file_manager.write_file(uploaded_file.name, content)
                st.success(f"File uploaded: {uploaded_file.name}")
                st.rerun()
            except Exception as e:
                st.error(f"Upload failed: {e}")
    
    with col3:
        if st.button(t('ui.files.refresh')):
            st.rerun()
    
    # File creation dialog
    if st.session_state.get('show_create_file'):
        with st.form("create_file_form"):
            st.subheader(t('ui.files.create_file'))
            filename = st.text_input(t('ui.files.filename'))
            content = st.text_area(t('ui.files.file_content'), height=200)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button(t('ui.files.create')):
                    if filename:
                        try:
                            managers.file_manager.write_file(filename, content or "")
                            st.success(t('ui.files.create_success'))
                            st.session_state.show_create_file = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Creation failed: {e}")
            
            with col2:
                if st.form_submit_button(t('ui.files.cancel')):
                    st.session_state.show_create_file = False
                    st.rerun()
    
    # File browser
    current_dir = Path.cwd()
    st.write(f"**{t('ui.files.current_directory')}:** {current_dir}")
    
    # File search
    search_term = st.text_input(t('ui.files.search_placeholder'))
    
    # List files
    try:
        files = []
        for item in current_dir.iterdir():
            if item.is_file() and not item.name.startswith('.'):
                if not search_term or search_term.lower() in item.name.lower():
                    files.append({
                        'name': item.name,
                        'size': item.stat().st_size,
                        'modified': datetime.fromtimestamp(item.stat().st_mtime),
                        'path': str(item)
                    })
        
        if files:
            df = pd.DataFrame(files)
            
            # File table
            for _, file_info in df.iterrows():
                col1, col2, col3, col4 = st.columns([3, 1, 2, 2])
                
                with col1:
                    st.write(f"📄 **{file_info['name']}**")
                
                with col2:
                    st.write(f"{file_info['size']} bytes")
                
                with col3:
                    st.write(file_info['modified'].strftime("%Y-%m-%d %H:%M"))
                
                with col4:
                    if st.button(t('ui.files.view'), key=f"view_{file_info['name']}"):
                        try:
                            content = managers.file_manager.read_file(file_info['path'])
                            st.text_area(f"Content of {file_info['name']}", content, height=300)
                        except Exception as e:
                            st.error(f"Error reading file: {e}")
        else:
            st.info("No files found")
            
    except Exception as e:
        st.error(f"Error listing files: {e}")

def show_context(managers: SystemManagers):
    """Context tab"""
    st.header(t('ui.context.title'))
    
    # Mention processing
    with st.expander(t('ui.context.process_mentions'), expanded=True):
        st.info(t('ui.context.mention_help'))
        
        mention_text = st.text_area(
            t('ui.context.text_input'),
            placeholder="Example: Please analyze @README.md and @config.json"
        )
        
        if st.button(t('ui.context.process')):
            if mention_text:
                try:
                    successful, failed = managers.context_manager.process_mentions(mention_text)
                    
                    if successful:
                        st.success(t('ui.context.mentions_processed'))
                        st.write(f"**{t('ui.context.successful_mentions')}:**")
                        for path in successful:
                            st.write(f"✅ {path}")
                        logger.info(f"Successfully processed mentions: {successful}")
                    
                    if failed:
                        st.write(f"**{t('ui.context.failed_mentions')}:**")
                        for path in failed:
                            st.write(f"❌ {path}")
                        logger.warning(f"Failed to process mentions: {failed}")
                            
                except Exception as e:
                    st.error(f"Error processing mentions: {e}")
                    logger.error(f"Error processing mentions: {e}", exc_info=True)
    
    # Current context
    st.subheader(t('ui.context.current_context'))
    context_info = managers.context_manager.get_context_info()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(t('ui.context.file_count'), context_info['file_count'])
    with col2:
        st.metric(t('ui.context.token_count'), f"{context_info['total_tokens']:,}")
    
    # Context files list
    context_files = managers.context_manager.get_context_files()
    
    if context_files:
        st.subheader(t('ui.context.context_files'))
        
        for file_path in context_files:
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"📄 {file_path}")
            
            with col2:
                if st.button(t('ui.context.view_content'), key=f"view_ctx_{file_path}"):
                    try:
                        content = managers.file_manager.read_file(file_path)
                        st.text_area(f"Content", content[:1000] + "..." if len(content) > 1000 else content)
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            with col3:
                if st.button(t('ui.context.remove'), key=f"remove_ctx_{file_path}"):
                    try:
                        managers.context_manager.remove_file(file_path)
                        st.success(f"Removed {file_path}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.info(t('ui.context.no_files'))
    
    # Clear context
    if st.button(t('ui.context.clear_context')):
        managers.context_manager.clear_context()
        st.success(t('ui.context.context_cleared'))
        logger.info("Context cleared by user.")
        st.rerun()

def show_monitor(managers: SystemManagers):
    """Monitor tab"""
    st.header(t('ui.monitor.title'))
    
    # System metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(t('ui.monitor.worker_settings'))
        
        current_workers = st.session_state.get('max_workers', 4)
        max_workers = st.slider(t('ui.monitor.max_workers'), 1, 16, current_workers)
        
        if st.button(t('ui.monitor.update_workers')):
            st.session_state.max_workers = max_workers
            st.success(t('ui.monitor.workers_updated'))
            logger.info(f"Max workers updated to {max_workers}")
        
        st.metric(t('ui.monitor.current_workers'), max_workers)
    
    with col2:
        st.subheader(t('ui.monitor.performance'))
        
        # Task distribution chart
        tasks = managers.task_manager.list_tasks()
        if tasks:
            task_status_counts = {}
            for task in tasks:
                status = task.status.value
                task_status_counts[status] = task_status_counts.get(status, 0) + 1
            
            fig = px.pie(
                values=list(task_status_counts.values()),
                names=list(task_status_counts.keys()),
                title=t('ui.monitor.task_distribution')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No task data available")
    
    # Auto-refresh
    auto_refresh = st.checkbox(t('ui.monitor.auto_refresh'))
    if auto_refresh:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    logger.info("Talos application started.")
    main()
