"""
Talos Main Entry Point
"""

import os
import sys
import click
from pathlib import Path

# Add Talos module path
sys.path.insert(0, str(Path(__file__).parent))

from talos.core.task_manager import get_task_manager, TaskType
from talos.core.context_manager import get_context_manager
from talos.core.vertex_ai_client import get_vertex_client
from talos.i18n import get_i18n, t


@click.group()
def cli():
    """Talos - AI-powered file management and task execution system"""
    pass


@cli.command()
@click.option('--host', default='localhost', help=t('cli.options.host'))
@click.option('--port', default=8501, help=t('cli.options.port'))
def web(host, port):
    """Launch web UI"""
    import streamlit.web.cli as stcli
    import sys
    
    # Run Streamlit app
    sys.argv = ["streamlit", "run", "app.py", "--server.address", host, "--server.port", str(port)]
    sys.exit(stcli.main())


@cli.command()
@click.argument('name')
@click.argument('prompt')
@click.option('--description', default=None, help='Task description')
@click.option('--type', 'task_type', default='single', type=click.Choice(['single', 'parallel', 'sequential']), help='Task type')
@click.option('--priority', default=0, help='Priority')
def create(name, prompt, description, task_type, priority):
    """Create a task."""
    tm = get_task_manager()
    
    type_map = {
        'single': TaskType.SINGLE,
        'parallel': TaskType.PARALLEL,
        'sequential': TaskType.SEQUENTIAL
    }
    
    task = tm.create_task(
        name=name,
        prompt=prompt,
        description=description,
        task_type=type_map[task_type],
        priority=priority
    )
    
    click.echo(f"✅ Task created successfully: {task.task_id}")
    click.echo(f"   Name: {task.name}")
    click.echo(f"   Workspace: {task.workspace_dir}")


@cli.command(name="list")
def list_tasks_command():
    """List all tasks."""
    task_manager = get_task_manager()
    tasks = task_manager.list_tasks()
    
    if not tasks:
        click.echo("No tasks created yet.")
        return
    
    click.echo(f"Total {len(tasks)} tasks:")
    click.echo("-" * 80)
    
    for task in tasks:
        status_emoji = {
            'pending': '⏸️',
            'running': '🔄',
            'completed': '✅',
            'failed': '❌',
            'cancelled': '⏹️'
        }.get(task.status.value, '❓')
        
        click.echo(f"{status_emoji} {task.name} ({task.task_id})")
        click.echo(f"   Status: {task.status.value} | Type: {task.task_type.value} | Priority: {task.priority}")
        click.echo(f"   Workspace: {task.workspace_dir}")
        if task.description:
            click.echo(f"   Description: {task.description}")
        click.echo()


@cli.command()
@click.argument('task_id')
def run(task_id):
    """Run a task."""
    tm = get_task_manager()
    
    task = tm.get_task(task_id)
    if not task:
        click.echo(f"❌ Task not found: {task_id}")
        return
    
    click.echo(f"🚀 Starting task: {task.name}")
    
    success = tm.execute_task(task_id)
    
    if success:
        click.echo(f"✅ Task completed: {task.name}")
        
        # Print results
        status = tm.get_task_status(task_id)
        if status and status.get('result'):
            click.echo("\n📄 Execution result:")
            click.echo(status['result'][:500] + "..." if len(status['result']) > 500 else status['result'])
        
        if status and status.get('output_files'):
            click.echo(f"\n📁 Output files: {len(status['output_files'])} files")
            for file_path in status['output_files'][:5]:  # Show up to 5 files
                click.echo(f"   - {file_path}")
    else:
        click.echo(f"❌ Task execution failed: {task.name}")
        
        status = tm.get_task_status(task_id)
        if status and status.get('error'):
            click.echo(f"Error: {status['error']}")


@cli.command()
def run_all():
    """Run all pending tasks."""
    tm = get_task_manager()
    
    results = tm.execute_all_pending_tasks()
    
    if not results:
        click.echo("No tasks to run.")
        return
    
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    click.echo(f"📊 Execution complete: {success_count}/{total_count} succeeded")
    
    for task_id, success in results.items():
        task = tm.get_task(task_id)
        status_emoji = "✅" if success else "❌"
        click.echo(f"{status_emoji} {task.name if task else task_id}")


@cli.command()
@click.argument('task_id')
def cancel(task_id):
    """Cancel a task."""
    tm = get_task_manager()
    
    success = tm.cancel_task(task_id)
    
    if success:
        click.echo(f"✅ Task cancellation complete: {task_id}")
    else:
        click.echo(f"❌ Task cancellation failed: {task_id}")


@cli.command()
@click.argument('task_id')
@click.option('--delete-workspace', is_flag=True, help='Delete workspace as well')
def delete(task_id, delete_workspace):
    """Delete a task."""
    tm = get_task_manager()
    
    task = tm.get_task(task_id)
    if not task:
        click.echo(f"❌ Task not found: {task_id}")
        return
    
    success = tm.delete_task(task_id, delete_workspace)
    
    if success:
        click.echo(f"✅ Task deletion complete: {task.name}")
        if delete_workspace:
            click.echo("   Workspace has also been deleted.")
    else:
        click.echo(f"❌ Task deletion failed: {task_id}")


@cli.command()
@click.argument('text')
def mention(text):
    """Mention processing"""
    cm = get_context_manager()
    
    click.echo("🔍 Processing mentions...")
    
    successful, failed = cm.process_mentions(text)
    
    if successful:
        click.echo("✅ Successful mentions:")
        for path in successful:
            click.echo(f"   + {path}")
    
    if failed:
        click.echo("❌ Failed mentions:")
        for path in failed:
            click.echo(f"   - {path}")
    
    # Print context information
    info = cm.get_context_info()
    click.echo(f"\n📚 Current context: {info['file_count']} files, {info['total_tokens']:,} tokens")


@cli.command()
def status():
    """Check system status"""
    click.echo("🔧 Talos System Status")
    click.echo("=" * 50)
    
    # AI client status
    try:
        ai_client = get_vertex_client()
        ai_info = ai_client.get_model_info()
        click.echo(f"✅ AI: {ai_info.get('model_name', 'Unknown')} ({ai_info.get('project_id', 'Unknown')})")
    except Exception as e:
        click.echo(f"❌ AI connection failed: {str(e)}")
    
    # Task manager status
    tm = get_task_manager()
    tasks = tm.list_tasks()
    pending_count = len([t for t in tasks if t.status.value == 'pending'])
    running_count = len([t for t in tasks if t.status.value == 'running'])
    completed_count = len([t for t in tasks if t.status.value == 'completed'])
    
    click.echo(f"📋 Tasks: Total {len(tasks)} (Pending {pending_count}, Running {running_count}, Completed {completed_count})")
    
    # Context status
    cm = get_context_manager()
    context_info = cm.get_context_info()
    click.echo(f"📚 Context: {context_info['file_count']} files, {context_info['total_tokens']:,} tokens")


@cli.command()
def version():
    """Version information"""
    from talos import __version__
    click.echo(f"Talos v{__version__}")
    click.echo("AI-based file management and task execution system")


def main():
    """Main function."""
    cli()


if __name__ == "__main__":
    main()