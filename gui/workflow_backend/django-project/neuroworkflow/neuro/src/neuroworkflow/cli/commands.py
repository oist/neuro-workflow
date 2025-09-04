"""
Command-line interface for NeuroWorkflow.

This module provides a command-line interface for running workflows.
"""

import argparse
import importlib.util
import sys
import json
from pathlib import Path

from neuroworkflow.core.workflow import Workflow


def run_workflow(args):
    """Run a workflow from a Python file.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Load the workflow module
    spec = importlib.util.spec_from_file_location("workflow_module", args.workflow_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Find the workflow object
    workflow = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, Workflow):
            workflow = attr
            break
    
    if workflow is None:
        print(f"No workflow found in {args.workflow_file}")
        return 1
        
    # Execute the workflow
    print(f"Executing workflow: {workflow.name}")
    success = workflow.execute()
    
    return 0 if success else 1


def info_workflow(args):
    """Get information about a workflow from a Python file.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Load the workflow module
    spec = importlib.util.spec_from_file_location("workflow_module", args.workflow_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Find the workflow object
    workflow = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, Workflow):
            workflow = attr
            break
    
    if workflow is None:
        print(f"No workflow found in {args.workflow_file}")
        return 1
        
    # Get workflow information
    info = workflow.get_info()
    
    # Print or save information
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"Workflow information saved to {args.output}")
    else:
        print(json.dumps(info, indent=2))
    
    return 0


def main():
    """Main entry point for the CLI.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(description="NeuroWorkflow CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run a workflow")
    run_parser.add_argument("workflow_file", help="Python file containing the workflow")
    run_parser.set_defaults(func=run_workflow)
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get information about a workflow")
    info_parser.add_argument("workflow_file", help="Python file containing the workflow")
    info_parser.add_argument("-o", "--output", help="Output file for workflow information (JSON)")
    info_parser.set_defaults(func=info_workflow)
    
    # Parse args and dispatch
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 1
        
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())