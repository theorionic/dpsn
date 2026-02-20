"""Prompt templates for fine-tuning datasets."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import os


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
            prompt = self.instruction_template.format(instruction=instruction)

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
    instruction_template="<|user|>\n{instruction}\n{input}\n",
    response_template="{prompt}<|assistant|>\n{output}",
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
            formatted_parts.append(f"<|user|>\n{content}\n")
        elif role in ("assistant", "gpt"):
            formatted_parts.append(f"<|assistant|>\n{content}\n")
        elif role == "system":
            formatted_parts.append(f"<|system|>\n{content}\n")

    return "".join(formatted_parts)
