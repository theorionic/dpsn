"""Prompt templates for fine-tuning datasets."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Set, Tuple
import json
import os
import re


@dataclass
class PromptTemplate:
    name: str
    instruction_template: str
    response_template: str
    system_template: Optional[str] = None

    def format(
        self,
        instruction: str,
        input_text: Optional[str] = None,
        output: Optional[str] = None,
    ) -> str:
        if input_text:
            prompt = self.instruction_template.format(
                instruction=instruction, input=input_text
            )
        else:
            prompt = self.instruction_template.format(instruction=instruction, input="")

        if output:
            result = self.response_template.format(prompt=prompt, output=output)
        else:
            prompt_end = self.response_template.split("{output}")[0]
            result = prompt_end.replace("{prompt}", prompt)

        return result

    def get_response_start(self) -> str:
        parts = self.response_template.split("{output}")
        if len(parts) >= 1:
            return parts[0].replace("{prompt}", "")
        return ""


ALPACA_TEMPLATE = PromptTemplate(
    name="alpaca",
    instruction_template="### Instruction:\n{instruction}\n\n### Input:\n{input}\n",
    response_template="{prompt}### Response:\n{output}",
)

ALPACA_NO_INPUT_TEMPLATE = PromptTemplate(
    name="alpaca_no_input",
    instruction_template="### Instruction:\n{instruction}\n",
    response_template="{prompt}### Response:\n{output}",
)

CHATML_TEMPLATE = PromptTemplate(
    name="chatml",
    instruction_template="<|im_start|>user\n{instruction}\n{input}<|im_end|>\n",
    response_template="{prompt}<|im_start|>assistant\n{output}<|im_end|>",
)

VICUNA_TEMPLATE = PromptTemplate(
    name="vicuna",
    instruction_template="USER: {instruction}\n{input}\n",
    response_template="{prompt}ASSISTANT: {output}",
)

LLAMA_TEMPLATE = PromptTemplate(
    name="llama",
    instruction_template="[INST] {instruction}\n{input} [/INST]",
    response_template="{prompt} {output}",
)

MISTRAL_TEMPLATE = PromptTemplate(
    name="mistral",
    instruction_template="[INST] {instruction}\n{input} [/INST]",
    response_template="{prompt} {output}",
)

SHAREGPT_TEMPLATE = PromptTemplate(
    name="sharegpt",
    instruction_template="USER: {instruction}\n{input}\n",
    response_template="{prompt}ASSISTANT: {output}",
)


BUILTIN_TEMPLATES: Dict[str, PromptTemplate] = {
    "alpaca": ALPACA_TEMPLATE,
    "alpaca_no_input": ALPACA_NO_INPUT_TEMPLATE,
    "chatml": CHATML_TEMPLATE,
    "vicuna": VICUNA_TEMPLATE,
    "llama": LLAMA_TEMPLATE,
    "mistral": MISTRAL_TEMPLATE,
    "sharegpt": SHAREGPT_TEMPLATE,
}


def get_template(name: str) -> PromptTemplate:
    if name not in BUILTIN_TEMPLATES:
        raise ValueError(
            f"Unknown template: {name}. Available: {list(BUILTIN_TEMPLATES.keys())}"
        )
    return BUILTIN_TEMPLATES[name]


def load_custom_template(path: str) -> PromptTemplate:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        return PromptTemplate(
            name=data.get("name", "custom"),
            instruction_template=data["instruction_template"],
            response_template=data["response_template"],
            system_template=data.get("system_template"),
        )
    elif ext in (".yaml", ".yml"):
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return PromptTemplate(
            name=data.get("name", "custom"),
            instruction_template=data["instruction_template"],
            response_template=data["response_template"],
            system_template=data.get("system_template"),
        )
    else:
        raise ValueError(f"Unsupported template file format: {ext}")


def format_conversation(
    messages: list[Dict[str, str]],
    template: PromptTemplate,
) -> str:
    formatted_parts = []

    for msg in messages:
        role = msg.get("role", msg.get("from", "user"))
        content = msg.get("content", msg.get("value", ""))

        if role in ("user", "human"):
            formatted_parts.append(f"USER: {content}\n")
        elif role in ("assistant", "gpt"):
            formatted_parts.append(f"ASSISTANT: {content}\n")
        elif role == "system":
            formatted_parts.append(f"SYSTEM: {content}\n")

    return "".join(formatted_parts)


# ================================================================================
# FLEXIBLE TEMPLATE SYSTEM - Auto-parse columns from template placeholders
# ================================================================================


def extract_placeholders(template: str) -> List[str]:
    """Extract all {placeholder} names from a template string.

    Args:
        template: Template string with {placeholder} patterns

    Returns:
        List of placeholder names (without braces)

    Example:
        >>> extract_placeholders("Q: {question}\\nA: {answer}")
        ['question', 'answer']
    """
    pattern = r"\{([^}]+)\}"
    return re.findall(pattern, template)


def validate_placeholders(
    template: str, available_columns: Set[str]
) -> Tuple[bool, List[str]]:
    """Validate that all template placeholders exist in dataset columns.

    Args:
        template: Template string with {placeholder} patterns
        available_columns: Set of column names from dataset

    Returns:
        Tuple of (is_valid, missing_placeholders)
    """
    placeholders = set(extract_placeholders(template))
    missing = placeholders - available_columns
    return len(missing) == 0, list(missing)


def build_column_mapping(template: str, available_columns: Set[str]) -> Dict[str, str]:
    """Build mapping from template placeholders to dataset columns.

    Auto-matches placeholders to columns. Returns dict like:
    {'question': 'question', 'answer': 'answer'}

    Args:
        template: Template string with {placeholder} patterns
        available_columns: Set of column names from dataset

    Returns:
        Dict mapping placeholder -> column_name

    Raises:
        ValueError: If a placeholder has no matching column
    """
    placeholders = extract_placeholders(template)
    mapping = {}

    for placeholder in placeholders:
        if placeholder in available_columns:
            mapping[placeholder] = placeholder
        else:
            # Try case-insensitive match
            for col in available_columns:
                if col.lower() == placeholder.lower():
                    mapping[placeholder] = col
                    break
            else:
                raise ValueError(
                    f"Template placeholder '{placeholder}' not found in dataset columns. "
                    f"Available: {sorted(available_columns)}"
                )

    return mapping


def format_with_columns(
    template: str,
    sample: Dict[str, Any],
    column_mapping: Optional[Dict[str, str]] = None,
) -> str:
    """Format template using column mapping to extract values from sample.

    Args:
        template: Template string with {placeholder} patterns
        sample: Dataset sample (dict with column values)
        column_mapping: Optional mapping from placeholder -> column_name.
                       If None, auto-detect from sample keys.

    Returns:
        Formatted string with placeholders replaced by values
    """
    if column_mapping is None:
        # Auto-detect mapping
        available_columns = set(sample.keys())
        column_mapping = build_column_mapping(template, available_columns)

    format_kwargs = {}
    for placeholder, column_name in column_mapping.items():
        value = sample.get(column_name, "")
        if value is None:
            value = ""
        format_kwargs[placeholder] = str(value)

    return template.format(**format_kwargs)


def detect_dataset_format(sample: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    """Auto-detect dataset format from a sample and return template name + column mapping.

    Detection order:
    1. ShareGPT format (has 'conversations' column)
    2. Alpaca format (has 'instruction', 'output' columns)
    3. Instruction format (has 'prompt'/'response' or 'question'/'answer')
    4. Raw text (has single text column)

    Args:
        sample: A sample from the dataset

    Returns:
        Tuple of (template_name, column_mapping)

    Example:
        >>> sample = {'question': 'What is 2+2?', 'answer': '4'}
        >>> detect_dataset_format(sample)
        ('alpaca', {'instruction': 'question', 'output': 'answer'})
    """
    columns = set(sample.keys())

    # 1. ShareGPT format: conversations column
    if "conversations" in columns:
        return "sharegpt", {}

    # 2. Alpaca format: instruction + output (exact match)
    if "instruction" in columns and "output" in columns:
        mapping = {"instruction": "instruction", "output": "output"}
        if "input" in columns:
            mapping["input"] = "input"
        return "alpaca", mapping

    # 3. Instruction format variants - detect common patterns
    instruction_cols = [
        "instruction",
        "prompt",
        "question",
        "query",
        "user",
        "user_input",
    ]
    output_cols = [
        "output",
        "response",
        "answer",
        "target",
        "assistant",
        "bot_response",
    ]

    detected_instruction = None
    detected_output = None

    for col in instruction_cols:
        if col in columns:
            detected_instruction = col
            break

    for col in output_cols:
        if col in columns:
            detected_output = col
            break

    if detected_instruction and detected_output:
        # Map detected columns to standard template placeholders
        mapping = {
            "instruction": detected_instruction,
            "output": detected_output,
        }
        if "input" in columns:
            mapping["input"] = "input"
        return "alpaca", mapping

    # 4. Raw text format: find any text column
    for col in ["text", "content", "passage", "document"]:
        if col in columns:
            return "raw_text", {"text": col}

    # No known format detected
    raise ValueError(
        f"Could not detect dataset format from columns: {sorted(columns)}. "
        f"Please specify --template explicitly with column names like: "
        f'--template "Instruction: {{instruction}}\\nResponse: {{output}}"'
    )


class FlexibleTemplate:
    """Flexible template that auto-maps placeholders to dataset columns.

    Unlike PromptTemplate which has fixed placeholders (instruction, input, output),
    FlexibleTemplate can use any placeholder names and auto-maps them to dataset columns.

    Example:
        >>> template = FlexibleTemplate("Q: {question}\\nA: {answer}")
        >>> sample = {'question': 'What is 2+2?', 'answer': '4', 'difficulty': 'easy'}
        >>> template.format(sample)
        'Q: What is 2+2?\\nA: 4'
    """

    def __init__(self, template_str: str, name: str = "flexible"):
        self.name = name
        self.template_str = template_str
        self.placeholders = extract_placeholders(template_str)
        self._column_mapping: Optional[Dict[str, str]] = None

    def detect_mapping(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """Detect column mapping from a sample."""
        available_columns = set(sample.keys())
        return build_column_mapping(self.template_str, available_columns)

    def format(
        self,
        sample: Optional[Dict[str, Any]] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        instruction: Optional[str] = None,
        input_text: Optional[str] = None,
        output: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Format the template with values from sample or keyword args.

        Supports two calling conventions:
        1. sample dict: template.format(sample_dict)
        2. kwargs: template.format(instruction=..., output=...) for PromptTemplate compatibility

        Args:
            sample: Dataset sample dict
            column_mapping: Optional explicit mapping. If None, uses auto-detected mapping.
            instruction: Instruction text (for PromptTemplate compatibility)
            input_text: Input text (for PromptTemplate compatibility)
            output: Output text (for PromptTemplate compatibility)
            **kwargs: Additional column values

        Returns:
            Formatted string
        """
        # If called with sample dict, use original behavior
        if sample is not None:
            if column_mapping is None:
                if self._column_mapping is None:
                    self._column_mapping = self.detect_mapping(sample)
                column_mapping = self._column_mapping
            return format_with_columns(self.template_str, sample, column_mapping)

        # If called with kwargs (PromptTemplate API), build format_kwargs from placeholders
        format_kwargs = {}
        for placeholder in self.placeholders:
            if placeholder == "instruction":
                format_kwargs[placeholder] = instruction or ""
            elif placeholder == "input":
                format_kwargs[placeholder] = input_text or ""
            elif placeholder == "input_text":
                format_kwargs[placeholder] = input_text or ""
            elif placeholder == "output":
                format_kwargs[placeholder] = output or ""
            elif placeholder in kwargs:
                format_kwargs[placeholder] = str(kwargs[placeholder])
            else:
                format_kwargs[placeholder] = ""

        return self.template_str.format(**format_kwargs)

    def format_with_output(
        self,
        sample: Dict[str, Any],
        output: Optional[str] = None,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> str:
        """Format template, optionally replacing the output placeholder.

        If output is provided, replaces the last placeholder with it.
        If output is None, returns template up to the output placeholder (for inference).

        Args:
            sample: Dataset sample dict
            output: Optional output text to include
            column_mapping: Optional explicit column mapping

        Returns:
            Formatted string
        """
        if column_mapping is None:
            if self._column_mapping is None:
                self._column_mapping = self.detect_mapping(sample)
            column_mapping = self._column_mapping

        # For flexible templates, we treat the last placeholder as "output"
        # Split template at last placeholder
        if not self.placeholders:
            return self.template_str

        last_placeholder = self.placeholders[-1]

        if output is not None:
            # Include the output in formatting
            format_kwargs = {}
            for placeholder in self.placeholders[:-1]:
                col_name = column_mapping.get(placeholder, placeholder)
                value = sample.get(col_name, "")
                if value is None:
                    value = ""
                format_kwargs[placeholder] = str(value)
            format_kwargs[last_placeholder] = output
            return self.template_str.format(**format_kwargs)
        else:
            # Return template up to output placeholder (for inference)
            format_kwargs = {}
            for placeholder in self.placeholders[:-1]:
                col_name = column_mapping.get(placeholder, placeholder)
                value = sample.get(col_name, "")
                if value is None:
                    value = ""
                format_kwargs[placeholder] = str(value)

            # Find the position of the last placeholder and truncate
            last_marker = "{" + last_placeholder + "}"
            template_up_to_output = self.template_str[
                : self.template_str.find(last_marker)
            ]
            return template_up_to_output.format(**format_kwargs)

    def get_output_column(self, sample: Dict[str, Any]) -> str:
        """Get the column name for the output (last placeholder)."""
        if self._column_mapping is None:
            self._column_mapping = self.detect_mapping(sample)
        last_placeholder = self.placeholders[-1] if self.placeholders else "output"
        return self._column_mapping.get(last_placeholder, last_placeholder)

    def __repr__(self) -> str:
        return f"FlexibleTemplate(name={self.name}, placeholders={self.placeholders})"


def parse_template_spec(template_spec: str) -> Tuple[str, Optional[str]]:
    """Parse a template specification string.

    Handles both:
    - Built-in template names: "alpaca", "chatml", etc.
    - Custom template strings: "Q: {question}\\nA: {answer}"

    Args:
        template_spec: Template name or custom template string

    Returns:
        Tuple of (template_type, template_content)
        - template_type: "builtin" or "custom"
        - template_content: Template name or custom string
    """
    # Check if it's a builtin template name
    if template_spec in BUILTIN_TEMPLATES:
        return "builtin", template_spec

    # Check if it contains placeholders (custom template)
    if "{" in template_spec and "}" in template_spec:
        return "custom", template_spec

    # Unknown - treat as potential builtin name
    return "unknown", template_spec
