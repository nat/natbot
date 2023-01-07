"""Given an objective, the current webpage, the list of previous actions, and the current decided action, pick the command to give the controller."""

import re
from typing import Dict, List
import cohere

from weblm.basic_controller.utils import (MODEL, DialogueState, Prompt, construct_prompt, construct_state,
                                          gather_examples, search, shorten_prompt, truncate_left, user_prompt_2,
                                          user_prompt_3)


def _get_cmd_prediction(co: cohere.Client, action: str, prompt: str, chosen_element: str) -> str:
    if "type" in action:
        text = None
        while text is None:
            try:
                num_tokens = 20
                if len(co.tokenize(prompt)) > 2048 - num_tokens:
                    print(f"WARNING: truncating sequence of length {len(co.tokenize(prompt))}")
                    prompt = truncate_left(co.tokenize, prompt, action, chosen_element, limit=2048 - num_tokens)

                print(len(co.tokenize(prompt + action + chosen_element)))
                text = max(co.generate(prompt=prompt + action + chosen_element,
                                       model=MODEL,
                                       temperature=0.5,
                                       num_generations=5,
                                       max_tokens=num_tokens,
                                       stop_sequences=["\n"],
                                       return_likelihoods="GENERATION").generations,
                           key=lambda x: x.likelihood).text
            except cohere.error.CohereError as e:
                print(f"Cohere fucked up: {e}")
                continue
    else:
        text = ""

    return (action + chosen_element + text).strip()


def generate_command(co: cohere.Client,
                     step: str,
                     action: str,
                     cmd: str,
                     chosen_elements: List[Dict[str, str]],
                     objective: str,
                     url: str,
                     pruned_elements: List[str],
                     previous_commands: List[str],
                     response: str = None):
    state = construct_state(objective, url, pruned_elements, previous_commands)
    examples = gather_examples(co, state)
    prompt = construct_prompt(state, examples)

    if step == DialogueState.Command:
        if len(pruned_elements) == 1:
            chosen_element = " " + " ".join(pruned_elements[0].split(" ")[:2])
            chosen_elements = [{"id": chosen_element}]
        else:
            state = construct_state(objective, url, ["$elements"], previous_commands)
            prompt = construct_prompt(state, examples)

            state, prompt = shorten_prompt(objective, url, ["$elements"], previous_commands, examples, action)

            group_size = 20
            chosen_elements = self.choose_element(
                prompt + action + "{id}",
                list(map(lambda x: {
                    "id": " " + " ".join(x.split(" ")[:2]),
                    "elements": x
                }, pruned_elements)),
                group_size,
                topk=5)
            chosen_element = chosen_elements[0]["id"]

            state = construct_state(objective, url, pruned_elements, previous_commands)
            prompt = construct_prompt(state, examples)

            state, prompt = shorten_prompt(co, objective, url, pruned_elements, previous_commands, examples, action,
                                           chosen_element)

        cmd = _get_cmd_prediction(co, action, prompt, chosen_element)

        step = DialogueState.CommandFeedback
        other_options = "\n".join(f"\t({i+2}){action}{x['id']}" for i, x in enumerate(chosen_elements[1:]))
        return step, cmd, chosen_elements, Prompt(eval(f'f"""{user_prompt_2}"""'))
    elif step == DialogueState.CommandFeedback:
        if response == "examples":
            examples = "\n".join(examples)
            return step, cmd, chosen_elements, Prompt(f"Examples:\n{examples}\n\n"
                                                      "Please respond with 'y' or 's'")

        if re.match(r'\d+', response):
            chosen_element = chosen_elements[int(response) - 1]["id"]
            state, prompt = shorten_prompt(co, objective, url, pruned_elements, previous_commands, examples, action,
                                           chosen_element)
            cmd = _get_cmd_prediction(co, action, prompt, chosen_element)
            if "type" in action:
                return step, cmd, chosen_elements, Prompt(eval(f'f"""{user_prompt_3}"""'))
        elif response != "y" and response != "s":
            cmd = response

        cmd_pattern = r"(click|type) (link|button|input|select) [\d]+( \"\w+\")?"
        if not re.match(cmd_pattern, cmd):
            return step, cmd, chosen_elements, Prompt(
                f"Invalid command '{cmd}'. Must match regex '{cmd_pattern}'. Try again...")

    return step, cmd, chosen_elements, None