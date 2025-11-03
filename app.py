"""
Talos Web Application
Streamlit-based web interface for AI-powered task management system
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

# Initialize managers
@st.cache_resource
def initialize_managers():
    """Initialize all system managers"""
    return {
        'task_manager': get_task_manager(),
        'context_manager': get_context_manager(),
        'file_manager': get_file_manager(),
        'parallel_executor': get_parallel_executor()
    }

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
        'ja': '🇯🇵 日本語'
    }
    
    selected = st.sidebar.selectbox(
        "Language / 언어 / 言語",
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

def show_dashboard(managers):
    """Dashboard tab"""
    st.header(t('ui.dashboard.title'))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(t('ui.dashboard.system_status'))
        
        # AI Status
        try:
            ai_client = get_ai_client(use_mock=False)
            ai_info = ai_client.get_model_info()
            
            st.success(f"**{t('ui.dashboard.ai_status')}:** {t('ui.dashboard.connected')}")
            st.write(f"**{t('ui.dashboard.model')}:** {ai_info.get('model_name', 'Unknown')}")
            st.write(f"**{t('ui.dashboard.project')}:** {ai_info.get('project_id', 'Unknown')}")
        except:
            st.error(f"**{t('ui.dashboard.ai_status')}:** {t('ui.dashboard.disconnected')}")
        
        # Task Summary
        st.subheader(t('ui.dashboard.task_summary'))
        tasks = managers['task_manager'].list_tasks()
        
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
        context_info = managers['context_manager'].get_context_info()
        
        st.metric(t('ui.dashboard.files_loaded'), context_info['file_count'])
        st.metric(t('ui.dashboard.total_tokens'), f"{context_info['total_tokens']:,}")
        
        # Recent Activity
        st.subheader(t('ui.dashboard.recent_activity'))
        if tasks:
            recent_tasks = sorted(tasks, key=lambda x: x.created_at, reverse=True)[:5]
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

def show_tasks(managers):
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

                    tasks_loaded = managers['task_manager'].load_tasks_from_file(file_path)
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
                height=150
            )
            
            if st.form_submit_button(t('ui.tasks.create_button')):
                if task_name and prompt:
                    try:
                        task = managers['task_manager'].create_task(
                            name=task_name,
                            prompt=prompt,
                            description=task_description,
                            task_type=task_type,
                            priority=priority
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
    tasks = managers['task_manager'].list_tasks()
    
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
                
                with col3:
                    st.write(f"**{t('ui.tasks.actions')}:**")
                    
                    action_col1, action_col2, action_col3 = st.columns(3)
                    
                    with action_col1:
                        if st.button(t('ui.tasks.run'), key=f"run_{task.task_id}"):
                            if managers['task_manager'].execute_task(task.task_id):
                                st.success(t('ui.tasks.run_success'))
                                st.rerun()
                            else:
                                st.error(t('ui.tasks.run_failed'))
                    
                    with action_col2:
                        if st.button(t('ui.tasks.cancel'), key=f"cancel_{task.task_id}"):
                            if managers['task_manager'].cancel_task(task.task_id):
                                st.success(t('ui.tasks.cancel_success'))
                                st.rerun()
                    
                    with action_col3:
                        if st.button(t('ui.tasks.delete'), key=f"delete_{task.task_id}"):
                            if managers['task_manager'].delete_task(task.task_id, delete_workspace=True):
                                st.success(t('ui.tasks.delete_success'))
                                st.rerun()
                
                st.divider()
    else:
        st.info(t('ui.tasks.no_tasks'))

def show_files(managers):
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
                managers['file_manager'].write_file(uploaded_file.name, content)
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
                            managers['file_manager'].write_file(filename, content or "")
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
                            content = managers['file_manager'].read_file(file_info['path'])
                            st.text_area(f"Content of {file_info['name']}", content, height=300)
                        except Exception as e:
                            st.error(f"Error reading file: {e}")
        else:
            st.info("No files found")
            
    except Exception as e:
        st.error(f"Error listing files: {e}")

def show_context(managers):
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
                    successful, failed = managers['context_manager'].process_mentions(mention_text)
                    
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
    context_info = managers['context_manager'].get_context_info()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(t('ui.context.file_count'), context_info['file_count'])
    with col2:
        st.metric(t('ui.context.token_count'), f"{context_info['total_tokens']:,}")
    
    # Context files list
    context_files = managers['context_manager'].get_context_files()
    
    if context_files:
        st.subheader(t('ui.context.context_files'))
        
        for file_path in context_files:
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"📄 {file_path}")
            
            with col2:
                if st.button(t('ui.context.view_content'), key=f"view_ctx_{file_path}"):
                    try:
                        content = managers['file_manager'].read_file(file_path)
                        st.text_area(f"Content", content[:1000] + "..." if len(content) > 1000 else content)
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            with col3:
                if st.button(t('ui.context.remove'), key=f"remove_ctx_{file_path}"):
                    try:
                        managers['context_manager'].remove_file(file_path)
                        st.success(f"Removed {file_path}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.info(t('ui.context.no_files'))
    
    # Clear context
    if st.button(t('ui.context.clear_context')):
        managers['context_manager'].clear_context()
        st.success(t('ui.context.context_cleared'))
        logger.info("Context cleared by user.")
        st.rerun()

def show_monitor(managers):
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
        tasks = managers['task_manager'].list_tasks()
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