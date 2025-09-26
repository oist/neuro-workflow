"""
Independent script export utilities for NeuroWorkflow.

This module provides functions to export workflow scripts based on execution
sequence dictionaries, supporting both Python (.py) and Jupyter notebook (.ipynb) formats.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional


def export_workflow_scripts(
    execution_sequence: Dict[str, Any],
    output_dir: str = "./output",
    export_python: bool = True,
    export_notebook: bool = True,
    filename_base: str = "workflow_script",
    deduplicate_imports: bool = True,
    add_metadata: bool = True
) -> Dict[str, str]:
    """
    Independent function to export workflow scripts based on execution sequence dictionary.
    
    Args:
        execution_sequence: Dictionary with execution sequence from WorkflowBuilder
        output_dir: Directory for output files
        export_python: Whether to export Python script
        export_notebook: Whether to export Jupyter notebook
        filename_base: Base filename for exports
        deduplicate_imports: Remove duplicate imports
        add_metadata: Add workflow metadata as comments
    
    Returns:
        Dictionary with paths to exported files
    """
    
    workflow_name = execution_sequence['workflow_name']
    sequence = execution_sequence['execution_sequence']
    
    print(f"Processing workflow: {workflow_name}")
    print(f"Total nodes in execution order: {len(sequence)}")
    
    # Collect script fragments in execution order
    script_fragments = []
    notebook_fragments = []
    
    for entry in sequence:
        node_name = entry['node_name']
        node_instance = entry['node_instance']
        
        print(f"Processing node: {node_name} (has_python: {entry['has_python_script']}, has_notebook: {entry['has_notebook_cell']})")
        
        # Collect Python script
        if entry['has_python_script']:
            python_script = node_instance._output_ports['python_script'].value
            if python_script and python_script.strip():
                script_fragments.append({
                    'node_name': node_name,
                    'node_type': entry['node_type'],
                    'execution_order': entry['execution_order'],
                    'script': python_script
                })
                print(f"  → Collected Python script ({len(python_script)} chars)")
        
        # Collect notebook cell (use python_script for code, notebook_cell for markdown)
        if entry['has_notebook_cell']:
            notebook_cell = node_instance._output_ports['notebook_cell'].value
            python_script = node_instance._output_ports['python_script'].value if entry['has_python_script'] else ""
            
            if notebook_cell and notebook_cell.strip():
                notebook_fragments.append({
                    'node_name': node_name,
                    'node_type': entry['node_type'],
                    'execution_order': entry['execution_order'],
                    'cell': notebook_cell,
                    'python_script': python_script  # Include full python script for code sections
                })
                print(f"  → Collected notebook cell ({len(notebook_cell)} chars) + python script ({len(python_script)} chars)")
    
    print(f"\nSummary:")
    print(f"  Python script fragments: {len(script_fragments)}")
    print(f"  Notebook cell fragments: {len(notebook_fragments)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    exported_files = {}
    
    # Export Python script
    if export_python and script_fragments:
        python_file = os.path.join(output_dir, f"{filename_base}.py")
        merged_python = _merge_python_scripts(
            script_fragments, execution_sequence, deduplicate_imports, add_metadata
        )
        
        with open(python_file, 'w') as f:
            f.write(merged_python)
        
        exported_files['python_script'] = python_file
        print(f"  Python script exported to: {python_file}")
    
    # Export Jupyter notebook
    if export_notebook and notebook_fragments:
        notebook_file = os.path.join(output_dir, f"{filename_base}.ipynb")
        notebook_json = _create_jupyter_notebook(
            notebook_fragments, execution_sequence, add_metadata
        )
        
        with open(notebook_file, 'w') as f:
            json.dump(notebook_json, f, indent=2)
        
        exported_files['jupyter_notebook'] = notebook_file
        print(f"  Jupyter notebook exported to: {notebook_file}")
    
    if not exported_files:
        print("   No script fragments found to export")
    
    return exported_files


def export_workflow_scripts_direct(
    workflow_instance,
    output_dir: str = "./output",
    **kwargs
) -> Dict[str, str]:
    """
    Direct export using WorkflowBuilder or Workflow instance.
    
    Args:
        workflow_instance: WorkflowBuilder or Workflow instance
        output_dir: Directory for output files
        **kwargs: Additional arguments passed to export_workflow_scripts
        
    Returns:
        Dictionary with paths to exported files
    """
    # Check if it's a WorkflowBuilder or Workflow instance
    if hasattr(workflow_instance, 'get_execution_sequence'):
        execution_sequence = workflow_instance.get_execution_sequence()
    else:
        raise ValueError("workflow_instance must have a get_execution_sequence() method")
    
    return export_workflow_scripts(execution_sequence, output_dir, **kwargs)


def _merge_python_scripts(
    fragments: List[Dict], 
    execution_sequence: Dict[str, Any], 
    deduplicate_imports: bool, 
    add_metadata: bool
) -> str:
    """Merge Python script fragments into a single script."""
    
    sections = []
    
    # Add header metadata
    if add_metadata:
        sections.extend([
            f"# Workflow Script: {execution_sequence['workflow_name']}",
            f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Original execution: {execution_sequence['execution_timestamp']}",
            f"# Execution order: {[f['node_name'] for f in fragments]}",
            f"# Total nodes: {len(fragments)}",
            "",
            "# This script was automatically generated from NeuroWorkflow execution",
            "# Each section corresponds to a node that was executed in the workflow",
            ""
        ])
    
    # Collect and deduplicate imports
    if deduplicate_imports:
        all_imports = set()
        script_bodies = []
        
        for fragment in fragments:
            lines = fragment['script'].split('\n')
            imports = [line for line in lines if line.strip().startswith(('import ', 'from '))]
            body_lines = [line for line in lines if not line.strip().startswith(('import ', 'from '))]
            
            all_imports.update([imp for imp in imports if imp.strip()])
            
            body = '\n'.join(body_lines).strip()
            if body:
                script_bodies.append({
                    'node_name': fragment['node_name'],
                    'node_type': fragment['node_type'],
                    'execution_order': fragment['execution_order'],
                    'body': body
                })
        
        # Add deduplicated imports
        if all_imports:
            sections.append("# === IMPORTS ===")
            sections.extend(sorted(all_imports))
            sections.append("")
        
        # Add script bodies in execution order
        for body_info in script_bodies:
            sections.extend([
                f"# === {body_info['node_name']} ({body_info['node_type']}) ===",
                f"# Execution order: {body_info['execution_order']}",
                body_info['body'],
                ""
            ])
    else:
        # Simple concatenation without import deduplication
        for fragment in fragments:
            sections.extend([
                f"# === {fragment['node_name']} ({fragment['node_type']}) ===",
                f"# Execution order: {fragment['execution_order']}",
                fragment['script'],
                ""
            ])
    
    return '\n'.join(sections)


def _create_jupyter_notebook(
    fragments: List[Dict], 
    execution_sequence: Dict[str, Any], 
    add_metadata: bool
) -> Dict:
    """Create Jupyter notebook JSON structure."""
    
    cells = []
    
    # Add metadata cell
    if add_metadata:
        execution_order_list = '\n'.join([
            f"{i+1}. **{f['node_name']}** ({f['node_type']})" 
            for i, f in enumerate(fragments)
        ])
        
        metadata_markdown = f"""# Workflow: {execution_sequence['workflow_name']}

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Original execution:** {execution_sequence['execution_timestamp']}  
**Total nodes:** {len(fragments)}  

## Execution Order:
{execution_order_list}

---

*This notebook was automatically generated from NeuroWorkflow execution.  
Each cell below corresponds to a node that was executed in the workflow.*
"""
        
        # Format metadata markdown with proper \n endings
        metadata_lines = metadata_markdown.split('\n')
        formatted_metadata = []
        for i, line in enumerate(metadata_lines):
            if i < len(metadata_lines) - 1:  # Not the last line
                formatted_metadata.append(line + '\n')
            else:  # Last line - no \n
                formatted_metadata.append(line)
        
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": formatted_metadata
        })
    
    # Add cells in execution order (parse markdown and code sections)
    for fragment in fragments:
        parsed_cells = _parse_notebook_cell_content(fragment)
        cells.extend(parsed_cells)
    
    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            },
            "workflow_metadata": {
                "workflow_name": execution_sequence['workflow_name'],
                "generation_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "original_execution": execution_sequence['execution_timestamp'],
                "total_nodes": len(fragments)
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook


def _parse_notebook_cell_content(fragment: Dict) -> List[Dict]:
    """
    Parse notebook cell content that may contain both markdown and code sections.
    
    Args:
        fragment: Dictionary with node information and cell content
        
    Returns:
        List of notebook cells (markdown and/or code)
    """
    
    cell_content = fragment['cell']
    python_script = fragment.get('python_script', '')
    node_name = fragment['node_name']
    node_type = fragment['node_type']
    execution_order = fragment['execution_order']
    
    # Check if content contains markdown sections
    if '# Markdown Cell' in cell_content and '```markdown' in cell_content:
        # Content has both markdown and code sections - parse them
        return _parse_mixed_content(cell_content, python_script, node_name, node_type, execution_order)
    else:
        # Content is pure code - use full python script if available
        code_content = python_script if python_script else cell_content
        return _create_code_cell(code_content, node_name, node_type, execution_order)


def _parse_mixed_content(content: str, python_script: str, node_name: str, node_type: str, execution_order: int) -> List[Dict]:
    """
    Parse content that contains both markdown and code sections.
    
    Expected format:
    # Markdown Cell
    ```markdown
    ## Title
    Content...
    ```
    
    # Code Cell  
    ```python
    import nest
    # code...
    ```
    """
    
    cells = []
    sections = content.split('# Markdown Cell')
    
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
            
        if section.startswith('```markdown'):
            # Extract markdown content
            markdown_content = _extract_markdown_content(section)
            if markdown_content:
                # Format markdown lines with proper \n endings
                markdown_lines = markdown_content.split('\n')
                formatted_markdown = []
                for i, line in enumerate(markdown_lines):
                    if i < len(markdown_lines) - 1:  # Not the last line
                        formatted_markdown.append(line + '\n')
                    else:  # Last line - no \n
                        formatted_markdown.append(line)
                
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {
                        "tags": [f"node-{node_name}", f"order-{execution_order}", "markdown"]
                    },
                    "source": formatted_markdown
                })
        
        # Look for code sections in this part
        if '# Code Cell' in section and '```python' in section:
            # Use full python_script instead of simplified notebook code
            if python_script:
                code_content = python_script
            else:
                code_content = _extract_code_content(section)
            
            if code_content:
                # Add header comment to code with proper line separation
                header_comment = f"# {node_name} ({node_type}) - Execution order: {execution_order}"
                full_code = f"{header_comment}\n\n{code_content}"
                
                # Ensure proper line splitting - filter out empty lines at start/end
                lines = full_code.split('\n')
                while lines and not lines[0].strip():
                    lines.pop(0)
                while lines and not lines[-1].strip():
                    lines.pop()
                
                # Add \n to each line except the last one (Jupyter notebook format requirement)
                formatted_lines = []
                for i, line in enumerate(lines):
                    if i < len(lines) - 1:  # Not the last line
                        formatted_lines.append(line + '\n')
                    else:  # Last line - no \n
                        formatted_lines.append(line)
                
                cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {
                        "tags": [f"node-{node_name}", f"order-{execution_order}", "code"]
                    },
                    "outputs": [],
                    "source": formatted_lines
                })
    
    # If no cells were parsed, treat entire content as code
    if not cells:
        code_content = python_script if python_script else content
        return _create_code_cell(code_content, node_name, node_type, execution_order)
    
    return cells


def _create_code_cell(content: str, node_name: str, node_type: str, execution_order: int) -> List[Dict]:
    """Create a single code cell from content."""
    
    # Add header comment with proper line separation
    header_comment = f"# {node_name} ({node_type}) - Execution order: {execution_order}"
    full_code = f"{header_comment}\n\n{content}"
    
    # Ensure proper line splitting - filter out empty lines at start/end
    lines = full_code.split('\n')
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    # Add \n to each line except the last one (Jupyter notebook format requirement)
    formatted_lines = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:  # Not the last line
            formatted_lines.append(line + '\n')
        else:  # Last line - no \n
            formatted_lines.append(line)
    
    return [{
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "tags": [f"node-{node_name}", f"order-{execution_order}", "code"]
        },
        "outputs": [],
        "source": formatted_lines
    }]


def _extract_markdown_content(section: str) -> str:
    """Extract markdown content from ```markdown ... ``` block."""
    
    lines = section.split('\n')
    markdown_lines = []
    in_markdown = False
    
    for line in lines:
        if line.strip() == '```markdown':
            in_markdown = True
            continue
        elif line.strip() == '```' and in_markdown:
            break
        elif in_markdown:
            # Fix font size: change ## to ### for smaller headers
            if line.strip().startswith('## '):
                line = line.replace('## ', '### ', 1)
            markdown_lines.append(line)
    
    return '\n'.join(markdown_lines).strip()


def _extract_code_content(section: str) -> str:
    """Extract Python code content from ```python ... ``` block."""
    
    lines = section.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        if line.strip() == '```python':
            in_code = True
            continue
        elif line.strip() == '```' and in_code:
            break
        elif in_code:
            code_lines.append(line)
    
    return '\n'.join(code_lines).strip()