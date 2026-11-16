"""mini-swe-agent's text-based harness templates, verbatim.

Copied from mini-swe-agent 2.4.6 (MIT License, Copyright (c) 2025 Kilian A.
Lieret and Carlos E. Jimenez), ``minisweagent/config/mini_textbased.yaml``,
plus the action regex of its text-based models and the defaults of its agent
and local environment. Do not edit; re-copy to update.
"""

MINI_SWE_AGENT_VERSION = "2.4.6"

SYSTEM_TEMPLATE = "You are a helpful assistant that can interact with a computer.\n\nYour response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).\nInclude a THOUGHT section before your command where you explain your reasoning process.\nFormat your response as shown in <format_example>.\n\n<format_example>\nYour reasoning and analysis here. Explain why you want to perform the action.\n\n```mswea_bash_command\nyour_command_here\n```\n</format_example>\n\nFailure to follow these rules will cause your response to be rejected.\n"

INSTANCE_TEMPLATE = "Please solve this issue: {{task}}\n\nYou can execute bash commands and edit files to implement the necessary changes.\n\n## Recommended Workflow\n\nThis workflow should be done step-by-step so that you can iterate on your changes and any possible problems.\n\n1. Analyze the codebase by finding and reading relevant files\n2. Create a script to reproduce the issue\n3. Edit the source code to resolve the issue\n4. Verify your fix works by running your script again\n5. Test edge cases to ensure your fix is robust\n6. Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.\n   Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>\n\n## Important Rules\n\n1. Every response must contain exactly one action\n2. The action must be enclosed in triple backticks\n3. Directory or environment variable changes are not persistent. Every action is executed in a new subshell.\n   However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files\n\n<system_information>\n{{system}} {{release}} {{version}} {{machine}}\n</system_information>\n\n## Formatting your response\n\nHere is an example of a correct response:\n\n<example_response>\nTHOUGHT: I need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.\n\n```mswea_bash_command\nls -la\n```\n</example_response>\n\n## Useful command examples\n\n### Create a new file:\n\n```mswea_bash_command\ncat <<'EOF' > newfile.py\nimport numpy as np\nhello = \"world\"\nprint(hello)\nEOF\n```\n\n### Edit files with sed:\n\n{%- if system == \"Darwin\" -%}\n<important>\nYou are on MacOS. For all the below examples, you need to use `sed -i ''` instead of `sed -i`.\n</important>\n{%- endif -%}\n\n```mswea_bash_command\n# Replace all occurrences\nsed -i 's/old_string/new_string/g' filename.py\n\n# Replace only first occurrence\nsed -i 's/old_string/new_string/' filename.py\n\n# Replace first occurrence on line 1\nsed -i '1s/old_string/new_string/' filename.py\n\n# Replace all occurrences in lines 1-10\nsed -i '1,10s/old_string/new_string/g' filename.py\n```\n\n### View file content:\n\n```mswea_bash_command\n# View specific lines with numbers\nnl -ba filename.py | sed -n '10,20p'\n```\n\n### Any other command you want to run\n\n```mswea_bash_command\nanything\n```\n"

OBSERVATION_TEMPLATE = "{% if output.exception_info -%}\n<exception>{{output.exception_info}}</exception>\n{% endif -%}\n<returncode>{{output.returncode}}</returncode>\n{% if output.output | length < 10000 -%}\n<output>\n{{ output.output -}}\n</output>\n{%- else -%}\n<warning>\nThe output of your last command was too long.\nPlease try a different command that produces less output.\nIf you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.\nIf you're using grep or find and it produced too much output, you can use a more selective search pattern.\nIf you really need to see something from the full command's output, you can redirect output to a file and then search in that file.\n</warning>\n{%- set elided_chars = output.output | length - 10000 -%}\n<output_head>\n{{ output.output[:5000] }}\n</output_head>\n<elided_chars>\n{{ elided_chars }} characters elided\n</elided_chars>\n<output_tail>\n{{ output.output[-5000:] }}\n</output_tail>\n{%- endif -%}\n"

FORMAT_ERROR_TEMPLATE = '{% if finish_reason is defined and finish_reason in ["length", "tool_calls"] -%}\nYour previous response reached the output token limit (finish_reason={{ finish_reason }}) before you produced a complete action, so it was cut off. Respond more concisely and provide exactly one action in the required format. If you need to think more, do so briefly.\n{%- else -%}\nFormat error:\n\n<error>\n{{error}}\n</error>\n\nHere is general guidance on how to format your response:\n\nPlease always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions.\n\nPlease format your action in triple backticks as shown in <response_example>.\n\n<response_example>\nHere are some thoughts about why you want to perform the action.\n\n```mswea_bash_command\n<action>\n```\n</response_example>\n\nIf you have completed your assignment, please consult the first message about how to\nsubmit your solution (you will not be able to continue working on this task after that).\n{%- endif %}'

ENVIRONMENT_ENV = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}

# minisweagent.models.openrouter_textbased_model.TextbasedModelConfig.action_regex
ACTION_REGEX = r"```mswea_bash_command\s*\n(.*?)\n```"

# minisweagent.environments.local.LocalEnvironmentConfig.timeout
COMMAND_TIMEOUT_SEC = 30

# minisweagent.agents.default.AgentConfig.max_consecutive_format_errors
MAX_CONSECUTIVE_FORMAT_ERRORS = 3

SUBMIT_COMMAND_OUTPUT = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
