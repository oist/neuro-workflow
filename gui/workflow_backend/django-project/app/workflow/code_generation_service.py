import re
import os
import ast
import json
import keyword
import textwrap
from pathlib import Path
from django.conf import settings
from .models import FlowProject, FlowNode, FlowEdge
from .path_utils import code_file_path, notebook_file_path, projects_root
import logging
import traceback

logger = logging.getLogger(__name__)


class CodeGenerationService:
    """A service that generates Python code from workflows (with .ipynb conversion functionality)"""

    def __init__(self):
        self.code_dir = projects_root()

        # Predefined regular expression patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile a regular expression pattern to use"""
        self.patterns = {
            # Detect WorkflowBuilder import
            "workflow_builder_import": re.compile(
                r"^(from\s+neuroworkflow.core.workflow\s+import\s+WorkflowBuilder)$", re.MULTILINE
            ),
        }

    def get_code_file_path(self, project, *, create=False):
        """Get the generated Python file path for a project."""
        return code_file_path(project, create=create)

    def get_notebook_file_path(self, project, *, create=False):
        """Get the generated notebook file path for a project."""
        return notebook_file_path(project, create=create)

    def _convert_py_to_ipynb(self, project_id):
        """Convert Python files to Jupyter Notebook"""
        try:
            # Get Project by Id
            project = FlowProject.objects.get(id=project_id)
            # Get file path
            code_file = self.get_code_file_path(project)
            notebook_file = self.get_notebook_file_path(project, create=code_file.parent.name == str(project.id))

            if not code_file.exists():
                logger.error(f"Python file does not exist: {code_file}")
                return False

            with open(code_file, "r", encoding="utf-8") as f:
                py_content = f.read()

            # Convert Python code to notebook format
            notebook_content = self._create_notebook_from_python(py_content)

            # Save to notebook file
            with open(notebook_file, "w", encoding="utf-8") as f:
                json.dump(notebook_content, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully converted to notebook: {notebook_file}")
            return True

        except Exception as e:
            logger.error(f"Error converting to notebook: {e}")
            logger.error(traceback.format_exc())
            return False

    def _create_notebook_from_python(self, py_content):
        """Create notebook structure from Python code"""
        # Split your Python code into appropriate cells
        cells = self._split_python_into_cells(py_content)

        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "codemirror_mode": {"name": "ipython", "version": 3},
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.0",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        return notebook

    def _split_python_into_cells(self, py_content):
        """Split your Python code into appropriate cells"""
        lines = py_content.split("\n")
        cells = []
        current_cell = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # Treat comment blocks as markdown cells
            if line.strip().startswith('"""') and len(line.strip()) > 3:
                # Save current cell
                if current_cell:
                    cell = self._create_code_cell("\n".join(current_cell))
                    if cell:
                        cells.append(cell)
                    current_cell = []

                docstring_content = []
                docstring_content.append(line.strip()[3:])  # Remove the first ""
                i += 1

                while i < len(lines) and not lines[i].strip().endswith('"""'):
                    docstring_content.append(lines[i])
                    i += 1

                if i < len(lines):
                    # Remove the last ""
                    last_line = lines[i].rstrip()
                    if last_line.endswith('"""'):
                        last_line = last_line[:-3]
                    if last_line:
                        docstring_content.append(last_line)

                markdown_content = "\n".join(docstring_content).strip()
                if markdown_content:
                    cell = self._create_markdown_cell(markdown_content)
                    if cell:
                        cells.append(cell)

            # Combine imports into one cell
            elif line.strip().startswith(
                ("import ", "from ")
            ) or line.strip().startswith("sys.path"):
                if current_cell and not any(
                    l.strip().startswith(("import ", "from ", "sys.path"))
                    for l in current_cell
                ):
                    cell = self._create_code_cell("\n".join(current_cell))
                    if cell:
                        cells.append(cell)
                    current_cell = []
                current_cell.append(line)

            # Start of function definition - whole function in one cell
            elif line.strip().startswith("def "):
                # Save current cell
                if current_cell:
                    cell = self._create_code_cell("\n".join(current_cell))
                    if cell:
                        cells.append(cell)
                    current_cell = []

                # Loading the entire function
                function_lines = [line]
                i += 1

                # Read the entire function (determined by indentation)
                while i < len(lines):
                    next_line = lines[i]
                    # Blank lines are included
                    if next_line.strip() == "":
                        function_lines.append(next_line)
                    # Indented lines or comments within functions
                    elif next_line.startswith("    ") or next_line.startswith("\t"):
                        function_lines.append(next_line)
                    # A new function definition, class definition, or top-level code begins
                    elif next_line.strip().startswith(
                        ("def ", "class ", "if __name__")
                    ) or (next_line.strip() and not next_line.startswith((" ", "\t"))):
                        # End of function, return index
                        i -= 1
                        break
                    else:
                        function_lines.append(next_line)
                    i += 1

                # Create function cell
                cell = self._create_code_cell("\n".join(function_lines))
                if cell:
                    cells.append(cell)

            # Starting class definition
            elif line.strip().startswith("class "):
                # Save current cell
                if current_cell:
                    cell = self._create_code_cell("\n".join(current_cell))
                    if cell:
                        cells.append(cell)
                    current_cell = []
                current_cell.append(line)

            # main execution part - if __name__ == "__main__": from to the end
            elif line.strip() == 'if __name__ == "__main__":':
                # Save current cell
                if current_cell:
                    cell = self._create_code_cell("\n".join(current_cell))
                    if cell:
                        cells.append(cell)
                    current_cell = []

                # Save current cell
                main_lines = [line]
                i += 1

                # Read all the way to the end of the file
                while i < len(lines):
                    main_lines.append(lines[i])
                    i += 1

                # Create the main execution cell.
                # Replace sys.exit(main()) with main() so the notebook cell
                # does not raise SystemExit(0) when executed in a Jupyter kernel.
                main_source = "\n".join(main_lines).replace(
                    "sys.exit(main())", "main()"
                )
                cell = self._create_code_cell(main_source)
                if cell:
                    cells.append(cell)
                break  # Create the main execution cell

            else:
                current_cell.append(line)

            i += 1

        # Add the rest of the code
        if current_cell:
            cell = self._create_code_cell("\n".join(current_cell))
            if cell:
                cells.append(cell)

        return cells

    def _create_code_cell(self, source_code):
        """Create code cell"""
        # Exclude empty code
        if not source_code.strip():
            return None

        # To preserve line breaks, store each line as an element in an array and add a newline character at the end.
        lines = source_code.split("\n")
        # Add a newline character to all lines except the last one
        source_lines = []
        for i, line in enumerate(lines):
            if i < len(lines) - 1:  # except the last line
                source_lines.append(line + "\n")
            else:  # last line
                source_lines.append(line)

        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source_lines,
        }

    def _create_markdown_cell(self, markdown_text):
        """Create a Markdown cell"""
        # To preserve line breaks, store each line as an element in an array and add a newline character at the end.
        lines = markdown_text.split("\n")
        # Add a newline character to all lines except the last one
        source_lines = []
        for i, line in enumerate(lines):
            if i < len(lines) - 1:  # except the last line
                source_lines.append(line + "\n")
            else:  # last line
                source_lines.append(line)

        return {"cell_type": "markdown", "metadata": {}, "source": source_lines}

    def _create_base_template(self, project):
        """Create a basic template (with section comments)"""
        context_obj = getattr(project, "workflow_context", {}) or {}
        context_block = json.dumps(context_obj, indent=4)
        context_block_indented = textwrap.indent(context_block, " " * 8)
        if context_block_indented.startswith(" " * 8):
            context_block_indented = context_block_indented[8:]
        workflow_context_literal = context_block_indented
        project_name = project.name or "workflow"
        return f'''#!/usr/bin/env python3
"""
{project.description if project.description else f"Generated workflow for project: {project.name}"}
"""
import sys
import os
import numpy as np

# Add paths for JupyterLab environment
sys.path.append('../../')

from neuroworkflow.core.workflow import WorkflowBuilder

def main():
    """Run a simple neural simulation workflow."""

    # workflow_builder_import

    # workflow_builder creation
    workflow_builder = WorkflowBuilder(
        "{project_name}",
        context={workflow_context_literal}
    )

    # Create nodes
     
    # workflow_builder_ready

    # Print workflow information
    print(workflow)

    # Execute workflow
    print("\\nExecuting workflow...")
    success = workflow.execute()
    
    if success:
        print("Workflow execution completed successfully!")
    else:
        print("Workflow execution failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''

    def _generate_import_statement(self, category, class_name):
        """Dynamically generate import statements from class names"""
        try:
            # Class Name Validation
            if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", class_name):
                logger.warning(f"Invalid class name format: {class_name}")
                return None

            # Import all nodes from nodes (dynamically generated)
            # nodes/{ClassName}.py import {ClassName} from
            catregory_folder = category.replace("/", "")
            return f"from nodes.{catregory_folder}.{class_name} import {class_name}"

        except Exception as e:
            logger.error(f"Error generating import statement for {class_name}: {e}")
            return None

    def _generate_node_code_block(self, node, node_no, instance_name):
        """Dynamically generate code blocks for nodes (category-based)"""
        label = node.data.get("label", "").strip()

        category = (
            node.data.get("nodeType", "")
            or getattr(node, "node_type", "")
            or node.data.get("category", "")
        ).strip()

        node_id = node.id

        logger.info(
            f"DEBUG: Generating code block for node {node_id} - label: '{label}', category: '{category}', node_data: {node.data}"
        )

        if not label:
            var_name = self._sanitize_variable_name(node_id, "node")
            logger.info(f"DEBUG: No label provided for node {node_id}")
            return f"""    # Node with no label (ID: {node_id})
        {var_name} = None  # TODO: Add implementation"""

        # Generate code for all categories
        category_lower = category.lower()
        logger.info(
            f"DEBUG: Generating code for node {node_id} - category '{category}' (normalized: '{category_lower}')"
        )

        #####
        var_name = instance_name
        if label == var_name:
            var_name = self._generate_variable_name_by_category(
                label, node_id, category, node_no
            )
        """
        var_name = self._generate_variable_name_by_category(
            label, node_id, category_lower, node_no
        )
        """
        #constructor_arg = self._generate_constructor_arg_by_category(
        #    label, category_lower
        #)
        configure_block = self._generate_configure_block_by_category(
            label, category_lower, node.data
        )

        #code_block = f"""    {var_name} = {label}("{constructor_arg}")"""
        code_block = f"""    {var_name} = {label}("{var_name}")"""

        if configure_block:
            code_block += f"""
    {var_name}.configure(
{configure_block}
    )\n"""
        else:
            code_block += f"""
    {var_name}.configure()\n"""

        logger.info(f"DEBUG: Generated code block for node {node_id}:\n{code_block}")
        return code_block

    def _sanitize_variable_name(self, node_id, prefix):
        """Converting node IDs to valid variable names"""
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", str(node_id))
        if sanitized and sanitized[0].isdigit():
            sanitized = f"{prefix}_{sanitized}"
        elif not sanitized:
            sanitized = prefix
        if keyword.iskeyword(sanitized):
            sanitized = f"{prefix}_{sanitized}"
        return sanitized

    def _generate_variable_name_by_category(self, class_name, node_id, category, node_no):
        """Generate variable names based on category (short names)"""
        # Extract only the numeric part from node_id (use the first numeric part)
        import re

        match = re.search(r"\d+", node_id)
        if match:
            # First 6 digits (or as is if the total is less than 6 digits)
            short_id = match.group()[:6]
        else:
            # If there is no number, the first 8 characters of the node_id
            short_id = node_id.replace("calc_", "").replace("_", "")[:8]

        node_no_zero = str(node_no).zfill(3)

        return f"instance_{class_name}_{node_no_zero}"
        
    """ Passing Node to the constructor (unused?)"""
    def _generate_constructor_arg_by_category(self, class_name, category):
        """Generate constructor arguments based on category"""
        if category == "network":
            return "SonataNetworkBuilder"
        elif category == "simulation":
            return "SonataNetworkSimulation"
        elif category == "analysis":
            return "SpikeAnalyzer"
        else:
            # For other categories, use the category name as is
            return category.capitalize()

    def _generate_configure_block_by_category(self, class_name, category, node_data):
        """Generate configure blocks based on category (including parameter changes)"""
        return self._generate_generic_configure_block(class_name, node_data)

    def _generate_generic_configure_block(self, class_name, node_data):
        """Generate configure blocks for other categories (only changed parameters)"""
        # Get only changed parameters
        modified_params = self._get_modified_parameters_for_configure(node_data, {})

        # Format the configure block
        config_lines = []
        for key, value in modified_params.items():
            # Sanitize parameter key to avoid Python keyword collisions
            safe_key = f"{key}_" if keyword.iskeyword(key) else key
            if isinstance(value, str):
                trimmed = value.strip()
                # Lambda expressions must be emitted bare — repr() would quote them
                # into a string literal, making them uncallable at runtime.
                if trimmed.startswith("lambda "):
                    config_lines.append(f"            {safe_key}={trimmed}")
                    continue
                # If the string looks like a Python dict/list literal (e.g. it was
                # saved with single quotes before frontend validation was in place),
                # recover it with ast.literal_eval so the generated code gets a real
                # dict/list rather than a quoted string.
                if trimmed.startswith(('{', '[')):
                    try:
                        parsed = ast.literal_eval(value)
                        config_lines.append(f"            {safe_key}={parsed!r}")
                        continue
                    except (ValueError, SyntaxError):
                        pass
                config_lines.append(f'            {safe_key}={repr(value)}')
            else:
                # int, float, bool, list, dict all render correctly via f-string.
                # Note: bool must be checked before int since bool subclasses int,
                # but f"{True}" already produces "True" so the else branch handles both.
                config_lines.append(f"            {safe_key}={value}")

        return ",\n".join(config_lines)

    def _get_modified_parameters_for_configure(self, node_data, default_params):
        """
        Gets parameter change information for a node and returns only those that have changed
        Converting categorically changed parameters to the appropriate configure settings
        """
        # A dictionary that stores only the changed parameters
        modified_params_only = {}

        # Get the current values ​​from the schema.parameters
        schema = node_data.get("schema", {})
        parameters = schema.get("parameters", {})

        logger.info(f"DEBUG: Processing parameters for configure block: {parameters}")

        # Get modification information from parameter_modifications
        modifications = node_data.get("parameter_modifications", {})
        logger.info(f"DEBUG: parameter_modifications: {modifications}")

        # Process only changed parameters
        for param_key, param_info in parameters.items():
            current_value = param_info.get("default_value")  # Get the current value from default_value
            logger.info(f"DEBUG: Processing parameter '{param_key}' with value '{current_value}'")

            # Get the parameter name that maps to a configuration setting
            config_key = self._map_parameter_to_config_key(param_key)
            logger.info(f"DEBUG: Parameter '{param_key}' mapped to config_key '{config_key}'")

            # Check only when default_value is changed
            if param_key in modifications:
                modification_info = modifications[param_key]
                # Compatible with new data structures
                original_value = modification_info.get("field_modifications", {}).get("default_value_original", "")
                is_modified = modification_info.get("is_modified", False)
                # Check if default_value is different from original value
                has_default_value_changed = (current_value != original_value) and is_modified
                logger.info(f"DEBUG: Parameter '{param_key}' default_value changed: {original_value} -> {current_value} (changed: {has_default_value_changed}, is_modified: {is_modified})")
            else:
                has_default_value_changed = False
                logger.info(f"DEBUG: Parameter '{param_key}' not found in modifications")

            # Add only parameters whose default_value has changed
            if (param_key in modifications and
                has_default_value_changed and
                config_key and current_value is not None):

                # perform type conversion
                converted_value = self._convert_parameter_value(current_value, config_key)
                modified_params_only[config_key] = converted_value

                modification_info = modifications[param_key]
                original_value = modification_info.get("field_modifications", {}).get("default_value_original", "")
                logger.info(f"DEBUG: Added parameter with default_value change '{param_key}' -> '{config_key}': {original_value} -> {current_value}")

        # Additional checks to ensure all items in parameter_modifications are processed
        # More detailed logging to resolve recognition issue after the third one
        logger.info(f"DEBUG: Starting additional check for all modifications: {len(modifications)} items")
        for i, (param_key, modification_info) in enumerate(modifications.items()):
            logger.info(f"DEBUG: Processing modification {i+1}/{len(modifications)}: {param_key}")

            if param_key not in parameters:
                logger.warning(f"DEBUG: Parameter '{param_key}' found in modifications but not in schema.parameters")
                continue

            param_info = parameters[param_key]
            current_value = param_info.get("default_value")
            # Compatible with new data structures
            original_value = modification_info.get("field_modifications", {}).get("default_value_original", "")
            is_modified = modification_info.get("is_modified", False)

            # Get the parameter name that maps to a configuration setting
            config_key = self._map_parameter_to_config_key(param_key)
            logger.info(f"DEBUG: Modification {i+1} - '{param_key}' -> '{config_key}': {original_value} -> {current_value} (is_modified: {is_modified})")

            # More lenient mutation detection (comparison after type conversion)
            current_converted = self._convert_parameter_value(current_value, config_key)
            original_converted = self._convert_parameter_value(original_value, config_key)

            has_change = (current_converted != original_converted) and is_modified
            already_processed = (config_key in modified_params_only)

            logger.info(f"DEBUG: Modification {i+1} - has_change: {has_change}, already_processed: {already_processed}")

            # Add any parameter changes that have not yet been processed
            if (config_key not in modified_params_only and
                has_change and
                config_key and current_value is not None):

                # perform type conversion
                converted_value = self._convert_parameter_value(current_value, config_key)
                modified_params_only[config_key] = converted_value

                logger.info(f"DEBUG: Added modification {i+1} '{param_key}' -> '{config_key}': {original_value} -> {current_value}")
            else:
                logger.info(f"DEBUG: Skipped modification {i+1} '{param_key}' (already processed or no change)")

        logger.info(f"DEBUG: Found {len(modified_params_only)} modified parameters for configure block")
        return modified_params_only

    def _map_parameter_to_config_key(self, parameter_key):
        """
        Use the parameter name as is (no mapping required)
        """
        # Returns the parameter name as is
        return parameter_key

    def _convert_parameter_value(self, value, config_key):
        """
        Converts parameter values ​​to the appropriate type (supports arrays and numbers)
        """
        # Processing arrays
        if isinstance(value, list):
            # If it is a list, convert it to Python array notation
            return value

        # Processing arrays coming from JSON as strings
        if isinstance(value, str) and value.strip().startswith('[') and value.strip().endswith(']'):
            try:
                import json
                parsed_value = json.loads(value)
                if isinstance(parsed_value, list):
                    return parsed_value
            except (json.JSONDecodeError, ValueError):
                logger.warning(f"Could not parse array string '{value}' for {config_key}, keeping as string")
                return value

        # Numerical parameters
        numeric_keys = ["simulation_time", "record_n_neurons", "hdf5_hyperslab_size"]

        # Automatically determines whether it is a number
        if config_key in numeric_keys or self._is_numeric_value(value):
            try:
                # convert to float or int
                if config_key == "simulation_time" or '.' in str(value):
                    return float(value)
                else:
                    return int(value)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert '{value}' to number for {config_key}, keeping as string")
                return value

        # String parameters (returned as is)
        return value

    def _is_numeric_value(self, value):
        """Determine if a value is a number"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def _extract_handle_name(self, handle_string):
        """
        Extract the port name from handle string.
        New format: {nodeId}::{fieldName}::{handleType}::{portType}
        Legacy format: {nodeId}-{fieldName}-{handleType}-{portType}
        """
        if not handle_string:
            return ""
        try:
            # New format (::)
            if '::' in handle_string:
                parts = handle_string.split('::')
                if len(parts) >= 2:
                    return parts[1]  # fieldName
                return handle_string

            # Legacy format (-)
            # {nodeId}-{fieldName}-{handleType}-{portType}
            # fieldName may contain '-', so join everything between nodeId and handleType
            parts = handle_string.split('-')
            if len(parts) >= 4 and parts[-2] in ('input', 'output'):
                return '-'.join(parts[1:-2])
            elif len(parts) >= 2:
                return parts[1]
            return handle_string
        except Exception as e:
            logger.warning(f"Could not extract handle name from '{handle_string}': {e}")
            return handle_string


    def _get_categories(self):
        """Get the category directory as a dictionary"""
        sub_directories = {}
        nodes_path = Path(settings.MEDIA_ROOT)
        for item in os.listdir(nodes_path):
            itemlarge = item.capitalize()
            if item == 'io':
                itemlarge = 'I/O'
            sub_directories[item] = itemlarge
        return sub_directories


    def _get_section_name_from_category(self, category):
        """Get section name from category"""
        """
        category_to_section = {
            "analysis": "Analysis",
            "io": "I/O",
            "network": "Network",
            "optimization": "Optimization",
            "simulation": "Simulation",
            "stimulus": "Stimulus",
        }
        """
        category_to_section = self._get_categories()
        return category_to_section.get(category.lower(), "Analysis")

    def _get_builder_name_from_label(self, label, node_id, category=None):
        """Get the name in the Builder from the label and category (same rules as constructor arguments)"""
        if category == "network":
            return "SonataNetworkBuilder"
        elif category == "simulation":
            return "SonataNetworkSimulation"
        elif category == "analysis":
            return "SpikeAnalyzer"
        else:
            # For other categories, use the category name as is
            return category.capitalize()

    def _build_workflow_commands_from_json(self, nodes_data, edges_data):
        """Generate workflow commands from node and edge information"""
        commands = []

        # Create a mapping from node ID to variable name and BuilderName
        node_id_to_var = {}
        node_id_to_builder = {}

        # Node number assignment
        node_no = 1

        # First, collect information on all nodes.
        for node_data in nodes_data:
            node_id = node_data.get("id", "")
            label = node_data.get("data", {}).get("label", "")

            # Fixed the method to get category
            category = (
                node_data.get("data", {}).get("nodeType", "")
                or node_data.get("type", "")
                or node_data.get("data", {}).get("category", "")
            ).lower()

            # Generate variable_name and builder_name for all categories
            class_name = node_data.get("data", {}).get("label", "")
            var_name = node_data.get("data", {}).get("instanceName", "")
            if class_name == var_name:
                var_name = self._generate_variable_name_by_category(
                    label, node_id, category, node_no
                )
            """
            builder_name = self._get_builder_name_from_label(
                label, node_id, category
            )
            """
            builder_name = var_name

            node_id_to_var[node_id] = var_name
            node_id_to_builder[node_id] = builder_name
            node_no += 1

        # Generate add_node command (for all nodes)
        for node_id in node_id_to_var:
            var_name = node_id_to_var[node_id]
            commands.append(f"    workflow_builder.add_node({var_name})")

        # Generate connect commands (for each edge)
        for edge_data in edges_data:
            source_id = edge_data.get("source", "")
            target_id = edge_data.get("target", "")

            # Get port information from sourceHandle and targetHandle
            source_handle_raw = edge_data.get("sourceHandle", "")
            target_handle_raw = edge_data.get("targetHandle", "")

            # Extract necessary parts from handle names (Example: calc_xxx-sonata_net-output-object -> sonata_net)
            source_handle = self._extract_handle_name(source_handle_raw)
            target_handle = self._extract_handle_name(target_handle_raw)

            if source_id in node_id_to_var and target_id in node_id_to_var:
                # Get builder name
                source_builder = node_id_to_builder.get(source_id, f"Node_{source_id}")
                target_builder = node_id_to_builder.get(target_id, f"Node_{target_id}")

                # Generate connect command
                commands.append(
                    f'    workflow_builder.connect("{source_builder}", "{source_handle}", '
                    f'"{target_builder}", "{target_handle}")'
                )

        # Add build() at the end
        commands.append("    workflow = workflow_builder.build()")

        return commands, node_id_to_var

    # Workflow Code Generator
    def generate_code_from_flow_data(self, project_id, project_name, nodes_data, edges_data):
        """React Flow New method for bulk code generation from JSON data"""
        try:
            logger.info(
                f"=== Starting batch code generation from flow data for project {project_id} ==="
            )
            logger.info(
                f"Processing {len(nodes_data)} nodes and {len(edges_data)} edges"
            )

            # Create a basic template for your project
            project = FlowProject.objects.get(id=project_id)
            base_code = self._create_base_template(project)

            # Organize nodes by category
            nodes_by_category = {}
            node_imports = set()

            logger.info(
                f"DEBUG: Processing {len(nodes_data)} nodes for NEW ARCHITECTURE"
            )

            # Assigning node numbers
            node_no = 1
            for i, node_data in enumerate(nodes_data):
                logger.info(f"DEBUG: Node {i+1}: {node_data}")

                # Retrieve node information from the actual database and include parameter change information
                node_id = node_data.get("id", "")
                instance_name = node_data.get("data", "").get("instanceName", "")
                try:
                    # Get actual nodes from DB (including parameter change information)
                    db_node = FlowNode.objects.get(id=node_id, project_id=project_id)
                    # The DB data contains parameter change information, so use that.
                    enhanced_node_data = db_node.data.copy()
                    # Location information etc. is obtained from JSON
                    enhanced_node_data.update(node_data.get("data", {}))
                    # Restore parameter_modifications from DB — the frontend React Flow state
                    # only updates schema on parameter change, so its parameter_modifications
                    # is stale. The DB is always the canonical source for this field.
                    enhanced_node_data['parameter_modifications'] = db_node.data.get('parameter_modifications', {})
                    enhanced_node_data['has_parameter_modifications'] = db_node.data.get('has_parameter_modifications', False)

                    logger.info(f"DEBUG: Enhanced node data with parameter modifications: {enhanced_node_data}")
                except FlowNode.DoesNotExist:
                    logger.warning(f"Node {node_id} not found in DB, using JSON data only")
                    enhanced_node_data = node_data.get("data", {})

                # Create a temporary FlowNode object (including parameter change information)
                temp_node = type(
                    "TempNode",
                    (),
                    {
                        "id": node_id,
                        "data": enhanced_node_data,
                        "position_x": node_data.get("position", {}).get("x", 0),
                        "position_y": node_data.get("position", {}).get("y", 0),
                        "node_type": node_data.get(
                            "type", "default"
                        ),  # Pass the type field
                    },
                )()

                # Generate a code block for a node
                code_block = self._generate_node_code_block(temp_node, node_no, instance_name)                
                logger.info(f"DEBUG: Generated code block: '{code_block}'")
                # Node number count
                node_no += 1

                if code_block and code_block.strip():
                    # Fixed the method to get category
                    category = (
                        temp_node.data.get("nodeType", "")
                        or temp_node.node_type
                        or temp_node.data.get("category", "")
                    ).lower()

                    if category not in nodes_by_category:
                        nodes_by_category[category] = []
                    nodes_by_category[category].append(
                        {"node": temp_node, "code_block": code_block}
                    )
                    logger.info(f"DEBUG: Added to {category} category")

                # Collect required imports (dynamically generated)
                label = temp_node.data.get("label", "").strip()
                if label:
                    import_statement = self._generate_import_statement(category, label)
                    if import_statement:
                        node_imports.add(import_statement)
                        logger.info(f"DEBUG: Added import: {import_statement}")

            # Add import statement
            updated_code = base_code
            logger.info(f"DEBUG: Adding {len(node_imports)} imports")
            for import_line in node_imports:
                if import_line not in updated_code:
                    # Add after WorkflowBuilder import
                    match = self.patterns["workflow_builder_import"].search(
                        updated_code
                    )
                    if match:
                        updated_code = updated_code.replace(
                            match.group(0), f"{match.group(0)}\n{import_line}"
                        )
                        logger.info(f"DEBUG: Added import: {import_line}")

            # Insert code blocks into sections by category
            logger.info(f"DEBUG: Categories found: {list(nodes_by_category.keys())}")

            # Create a node for each category (create all categories in one section)
            section_codes = ""
            for category, node_list in nodes_by_category.items():
                section_name = self._get_section_name_from_category(category)

                # Combine code blocks in sections
                section_code_blocks = [
                    node_info["code_block"] for node_info in node_list
                ]
                section_code = "\n".join(section_code_blocks)
                section_codes += section_code

            # detect section
            section_pattern = re.compile(
                #rf"^(\s*)# {re.escape(section_name)} field\s*$", re.MULTILINE
                rf"# Create nodes", re.MULTILINE
            )
            match = section_pattern.search(updated_code)

            if match:
                insertion_point = match.end()
                logger.info(
                    f"DEBUG: Found '{section_name}' section at position {insertion_point}"
                )

                # Delete the existing code in the section and replace it with the new code
                # Search to the next section or workflow builder marker
                next_section_pattern = re.compile(
                    #r'^(\s*)# (Analysis|IO|Network|Optimization|Simulation|Stimulus|Test|Create workflow) field\s*$',
                    r'^(\s*)# workflow_builder_ready\s*$',
                    re.MULTILINE
                )

                # Find next section after insertion_point
                remaining_code = updated_code[insertion_point:]
                next_match = next_section_pattern.search(remaining_code)

                if next_match:
                    # If the next section is found, replace it up to that point
                    section_end = insertion_point + next_match.start()
                else:
                    # If the next section is not found, continue to the end of the file.
                    section_end = len(updated_code)

                # Replace section content (delete existing code and insert new code)
                before_section = updated_code[:insertion_point]
                after_section = updated_code[section_end:]
                updated_code = f"{before_section}\n{section_codes}\n{after_section}"
                logger.info(
                    f"DEBUG: Replaced section content with {len(section_code_blocks)} code blocks in '{section_name}' section"
                )
            else:
                logger.error(f"DEBUG: Could not find '{section_name}' section")
            


            # Generate Workflow Command
            logger.info(f"DEBUG: Building workflow commands")
            workflow_commands, node_id_to_var = self._build_workflow_commands_from_json(
                nodes_data, edges_data
            )
            logger.info(f"DEBUG: Generated {len(workflow_commands)} workflow commands")
            for command in workflow_commands:
                logger.info(f"DEBUG: Command: {command}")

            # Insert the command in the workflow builder marker
            workflow_section_pattern = re.compile(
                r'^(\s*)# workflow_builder_ready\s*$',
                re.MULTILINE,
            )
            match = workflow_section_pattern.search(updated_code)

            logger.info(f"DEBUG: !!! updated_code !!! {updated_code}")

            if match:
                insertion_point = match.end()
                logger.info(
                    f"DEBUG: Found workflow builder marker at position {insertion_point}"
                )

                if workflow_commands:
                    # insert command
                    before_commands = updated_code[:insertion_point]
                    after_commands = updated_code[insertion_point:]
                    commands_text = "\n" + "\n".join(workflow_commands) + "\n"
                    updated_code = before_commands + commands_text + after_commands
                    logger.info(
                        f"DEBUG: Inserted {len(workflow_commands)} workflow commands"
                    )
                else:
                    logger.info(f"DEBUG: No workflow commands to insert")
            else:
                logger.error(f"DEBUG: Could not find workflow builder marker")

            # save to file
            code_file = self.get_code_file_path(project, create=True)
            code_file.parent.mkdir(parents=True, exist_ok=True)

            with open(code_file, "w", encoding="utf-8") as f:
                f.write(updated_code)

            logger.info(f"DEBUG: Final generated code:\n{updated_code}")
            logger.info(f"Successfully saved generated code to: {code_file}")

            # Sidecar map used by the run stream to attribute output (figures)
            # to canvas nodes. Keyed by generated variable name; on duplicate
            # instanceNames last-wins (the generated code is broken then anyway).
            node_map = {
                "version": 1,
                "var_to_node": {
                    var_name: node_id for node_id, var_name in node_id_to_var.items()
                },
            }
            node_map_file = code_file.parent / "node_map.json"
            with open(node_map_file, "w", encoding="utf-8") as f:
                json.dump(node_map, f, indent=2)
            logger.info(f"Saved node map to: {node_map_file}")

            # Convert to Jupyter notebook
            notebook_success = self._convert_py_to_ipynb(project_id)
            if notebook_success:
                logger.info("Successfully converted to Jupyter notebook")
            else:
                logger.warning("Failed to convert to Jupyter notebook")

            logger.info("=== Batch code generation completed successfully ===")
            return True

        except Exception as e:
            logger.error(f"=== Critical error in batch code generation: {e} ===")
            logger.error(traceback.format_exc())
            return False
