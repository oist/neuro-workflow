#!/usr/bin/env python3
"""
Detailed examples of export_workflow_scripts() function usage.

This script shows all possible arguments and what the return value contains.
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from neuroworkflow.core.workflow import WorkflowBuilder
from neuroworkflow.utils.script_exporter import export_workflow_scripts, export_workflow_scripts_direct

# Import the updated nodes from check directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'check'))
from SNNbuilder_SingleNeuron import SNNbuilder_SingleNeuron
from SNNbuilder_Population import SNNbuilder_Population


def create_sample_workflow():
    """Create a sample workflow for demonstration."""
    
    # Create workflow
    workflow_builder = WorkflowBuilder("Epilepsy Simulation Model")
    
    # Create and configure nodes
    neuron = SNNbuilder_SingleNeuron("excitatory_neuron")
    population = SNNbuilder_Population("cortical_population")
    
    neuron.configure(
        name='Excitatory Pyramidal Neuron',
        execution_mode='both',
        script_format='both'
    )
    
    population.configure(
        name='Cortical Excitatory Population',
        population_size=500,
        execution_mode='both',
        script_format='both'
    )
    
    # Build workflow
    workflow_builder.add_node(neuron)
    workflow_builder.add_node(population)
    workflow_builder.connect(neuron.name, "nest_model_name", population.name, "nest_model_name")
    workflow_builder.connect(neuron.name, "nest_properties", population.name, "nest_properties")
    
    # Execute workflow
    workflow_ready = workflow_builder.build()
    workflow_ready.execute()
    
    return workflow_ready


def example_1_basic_usage():
    """Example 1: Basic usage with minimal arguments."""
    
    print("=== Example 1: Basic Usage ===")
    
    workflow = create_sample_workflow()
    execution_sequence = workflow.get_execution_sequence()
    
    # BASIC USAGE - Only required argument
    files = export_workflow_scripts(
        execution_sequence=execution_sequence
    )
    
    print("Arguments used:")
    print("  execution_sequence: <execution_sequence_dict>")
    print("  (all other arguments use defaults)")
    print()
    print("Return value (files variable):")
    for key, value in files.items():
        print(f"  {key}: {value}")
    print()
    
    # Check if files exist
    print("File verification:")
    for file_type, file_path in files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"   {file_path} ({size} bytes)")
        else:
            print(f"   {file_path} (not found)")
    print()


def example_2_custom_arguments():
    """Example 2: Custom arguments with specific settings."""
    
    print("=== Example 2: Custom Arguments ===")
    
    workflow = create_sample_workflow()
    execution_sequence = workflow.get_execution_sequence()
    
    # CUSTOM USAGE - All arguments specified
    files = export_workflow_scripts(
        execution_sequence=execution_sequence,
        output_dir="./custom_epilepsy_output",           # Custom directory
        export_python=True,                              # Export Python script
        export_notebook=True,                            # Export Jupyter notebook
        filename_base="epilepsy_model_v2",               # Custom filename base
        deduplicate_imports=True,                        # Remove duplicate imports
        add_metadata=True                                # Add workflow metadata
    )
    
    print("Arguments used:")
    print("  execution_sequence: <execution_sequence_dict>")
    print("  output_dir: './custom_epilepsy_output'")
    print("  export_python: True")
    print("  export_notebook: True") 
    print("  filename_base: 'epilepsy_model_v2'")
    print("  deduplicate_imports: True")
    print("  add_metadata: True")
    print()
    print("Return value (files variable):")
    for key, value in files.items():
        print(f"  {key}: {value}")
    print()
    
    # Check if files exist
    print("File verification:")
    for file_type, file_path in files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"   {file_path} ({size} bytes)")
        else:
            print(f"   {file_path} (not found)")
    print()


def example_3_python_only():
    """Example 3: Export only Python script, no notebook."""
    
    print("=== Example 3: Python Script Only ===")
    
    workflow = create_sample_workflow()
    execution_sequence = workflow.get_execution_sequence()
    
    # PYTHON ONLY - No Jupyter notebook
    files = export_workflow_scripts(
        execution_sequence=execution_sequence,
        output_dir="./python_only_output",
        export_python=True,                              # Export Python script
        export_notebook=False,                           # NO Jupyter notebook
        filename_base="simulation_script",
        deduplicate_imports=False,                       # Keep all imports as-is
        add_metadata=False                               # Clean script without metadata
    )
    
    print("Arguments used:")
    print("  execution_sequence: <execution_sequence_dict>")
    print("  output_dir: './python_only_output'")
    print("  export_python: True")
    print("  export_notebook: False  # NO NOTEBOOK!")
    print("  filename_base: 'simulation_script'")
    print("  deduplicate_imports: False")
    print("  add_metadata: False")
    print()
    print("Return value (files variable):")
    for key, value in files.items():
        print(f"  {key}: {value}")
    print()
    
    # Check if files exist
    print("File verification:")
    for file_type, file_path in files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"   {file_path} ({size} bytes)")
        else:
            print(f"   {file_path} (not found)")
    print()


def example_4_notebook_only():
    """Example 4: Export only Jupyter notebook, no Python script."""
    
    print("=== Example 4: Jupyter Notebook Only ===")
    
    workflow = create_sample_workflow()
    execution_sequence = workflow.get_execution_sequence()
    
    # NOTEBOOK ONLY - No Python script
    files = export_workflow_scripts(
        execution_sequence=execution_sequence,
        output_dir="./notebook_only_output",
        export_python=False,                             # NO Python script
        export_notebook=True,                            # Export Jupyter notebook
        filename_base="epilepsy_analysis",
        add_metadata=True                                # Include metadata in notebook
    )
    
    print("Arguments used:")
    print("  execution_sequence: <execution_sequence_dict>")
    print("  output_dir: './notebook_only_output'")
    print("  export_python: False  # NO PYTHON SCRIPT!")
    print("  export_notebook: True")
    print("  filename_base: 'epilepsy_analysis'")
    print("  add_metadata: True")
    print()
    print("Return value (files variable):")
    for key, value in files.items():
        print(f"  {key}: {value}")
    print()
    
    # Check if files exist
    print("File verification:")
    for file_type, file_path in files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"   {file_path} ({size} bytes)")
        else:
            print(f"   {file_path} (not found)")
    print()


def example_5_direct_workflow_method():
    """Example 5: Using export_workflow_scripts_direct() method."""
    
    print("=== Example 5: Direct Workflow Method ===")
    
    workflow = create_sample_workflow()
    
    # DIRECT METHOD - Pass workflow instance directly
    files = export_workflow_scripts_direct(
        workflow_instance=workflow,                      # Pass workflow directly!
        output_dir="./direct_method_output",
        filename_base="direct_export",
        export_python=True,
        export_notebook=True
    )
    
    print("Arguments used:")
    print("  workflow_instance: <workflow_object>  # Direct workflow instance!")
    print("  output_dir: './direct_method_output'")
    print("  filename_base: 'direct_export'")
    print("  export_python: True")
    print("  export_notebook: True")
    print()
    print("Return value (files variable):")
    for key, value in files.items():
        print(f"  {key}: {value}")
    print()
    
    # Check if files exist
    print("File verification:")
    for file_type, file_path in files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"   {file_path} ({size} bytes)")
        else:
            print(f"   {file_path} (not found)")
    print()


def explain_return_value():
    """Explain what the return value (files variable) contains."""
    
    print("=== What is the 'files' Variable? ===")
    print()
    print("The 'files' variable is a Python dictionary that contains:")
    print("  - Keys: Type of exported file ('python_script', 'jupyter_notebook')")
    print("  - Values: Full file path where the file was saved")
    print()
    print("Example return values:")
    print()
    print("1. Both Python and Notebook exported:")
    print("   files = {")
    print("       'python_script': './output/workflow_script.py',")
    print("       'jupyter_notebook': './output/workflow_script.ipynb'")
    print("   }")
    print()
    print("2. Only Python script exported:")
    print("   files = {")
    print("       'python_script': './output/simulation_script.py'")
    print("   }")
    print()
    print("3. Only Jupyter notebook exported:")
    print("   files = {")
    print("       'jupyter_notebook': './output/analysis.ipynb'")
    print("   }")
    print()
    print("4. No files exported (no script content found):")
    print("   files = {}")
    print()
    print("How to use the return value:")
    print()
    print("# Check what was exported")
    print("if 'python_script' in files:")
    print("    print(f'Python script saved to: {files[\"python_script\"]}')") 
    print()
    print("if 'jupyter_notebook' in files:")
    print("    print(f'Notebook saved to: {files[\"jupyter_notebook\"]}')") 
    print()
    print("# Get file paths for further processing")
    print("python_file = files.get('python_script')")
    print("notebook_file = files.get('jupyter_notebook')")
    print()
    print("# Verify files exist")
    print("import os")
    print("for file_type, file_path in files.items():")
    print("    if os.path.exists(file_path):")
    print("        print(f'{file_type}: {file_path} (exists)')")
    print("    else:")
    print("        print(f'{file_type}: {file_path} (missing!)')")
    print()


def explain_file_storage():
    """Explain how files are automatically stored."""
    
    print("=== How Are Files Automatically Stored? ===")
    print()
    print("YES, files are automatically stored to disk!")
    print()
    print("The export_workflow_scripts() function:")
    print("1. Creates the output directory if it doesn't exist")
    print("2. Generates the script content from workflow execution")
    print("3. Writes files to disk automatically")
    print("4. Returns the file paths in the 'files' dictionary")
    print()
    print("File naming pattern:")
    print("  Python script: {output_dir}/{filename_base}.py")
    print("  Jupyter notebook: {output_dir}/{filename_base}.ipynb")
    print()
    print("Examples:")
    print("  output_dir='./results', filename_base='simulation'")
    print("  → './results/simulation.py'")
    print("  → './results/simulation.ipynb'")
    print()
    print("  output_dir='/home/user/models', filename_base='epilepsy_v2'")
    print("  → '/home/user/models/epilepsy_v2.py'")
    print("  → '/home/user/models/epilepsy_v2.ipynb'")
    print()
    print("The function handles:")
    print("  Directory creation (os.makedirs)")
    print("  File writing (with open())")
    print("  Error handling (permissions, disk space, etc.)")
    print("  Path resolution (relative/absolute paths)")
    print()


if __name__ == "__main__":
    try:
        print("🔬 Export Function Examples and Explanations\n")
        
        example_1_basic_usage()
        example_2_custom_arguments()
        example_3_python_only()
        example_4_notebook_only()
        example_5_direct_workflow_method()
        
        explain_return_value()
        explain_file_storage()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)