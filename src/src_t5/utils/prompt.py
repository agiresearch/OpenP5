import os
import re
from utils.utils import ReadLineFromFile

def load_prompt_template(path, task_list):
    """
    Load prompt template from the file. Keep training tasks only.
    Input:
    - path: The path for prompt template txt file.
    - task_list: A list of required tasks.
    Return:
    - prompt_templates: a dictionary of prompt templates. e.g., {task: {'seen': {'0': {'Input': template_input, 'Output': template_output}}}}
    
    """
    
    if not os.path.exists(path):
        raise FileNotFoundError
    prompt_info = ReadLineFromFile(path)
    prompt_templates = dict()
    for prompt in prompt_info:
        t = [sens.strip() for sens in prompt.split(';')]
        if t[0] not in task_list:
            continue
        if t[0] not in prompt_templates:
            prompt_templates[t[0]] = dict()
        if t[1] not in prompt_templates[t[0]]:
            prompt_templates[t[0]][t[1]] = dict()
        num = len(prompt_templates[t[0]][t[1]])
        prompt_templates[t[0]][t[1]][str(num)] = dict()
        prompt_templates[t[0]][t[1]][str(num)]['Input'] = t[2]
        prompt_templates[t[0]][t[1]][str(num)]['Output'] = t[3]
    return prompt_templates

def get_info_from_prompt(prompt_templates):
    """
    Extract the require information from the prompt templates.
    Input:
    - prompt_templates: a dictionary of prompt templates.
    Output:
    - info: a list of required information.
    """
    
    info = []
    for task in prompt_templates:
        for see in prompt_templates[task]:
            for i in prompt_templates[task][see]:
                info += re.findall(r'\{.*?\}', prompt_templates[task][see][i]['Input'])
                info += re.findall(r'\{.*?\}', prompt_templates[task][see][i]['Output'])
    info = [i[1:-1] for i in set(info)]
    return info

def check_task_prompt(prompt_templates, task_list):
    """
    Check if all tasks have prompt templates. Raise Error if training tasks have no prompt.
    Input:
    - prompt_templates: A dictionary of prompt templates.
    - task_list: A list of training tasks.
    """
    for task in task_list:
        assert task in prompt_templates, f"No prompt for {task} task"
    

