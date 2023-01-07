"""Misc. utility function"""

from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import itertools
import json
import math
from typing import Any, Dict, List, Tuple
import cohere
import numpy as np

MAX_SEQ_LEN = 2000
MAX_NUM_ELEMENTS = 50
TYPEABLE = ["input", "select"]
CLICKABLE = ["link", "button"]
MODEL = "xlarge"

prompt_template = """Given:
    (1) an objective that you are trying to achieve
    (2) the URL of your current web page
    (3) a simplified text description of what's visible in the browser window

Your commands are:
    click X - click on element X.
    type X "TEXT" - type the specified text into input X

Here are some examples:

$examples

Present state:
$state
Next Command:"""

state_template = """Objective: $objective
Current URL: $url
Current Browser Content:
------------------
$browser_content
------------------
Previous actions:
$previous_commands"""

user_prompt_end = ("\n\t(success) the goal is accomplished"
                   "\n\t(cancel) terminate the session"
                   "\nType a choice and then press enter:")
user_prompt_1 = ("Given web state:\n{state}"
                 "\n\nI have to choose between `clicking` and `typing` here."
                 "\n**I think I should{action}**"
                 "\n\t(y) proceed with this action"
                 "\n\t(n) do the other action" + user_prompt_end)
user_prompt_2 = ("Given state:\n{construct_state(objective, url, pruned_elements, previous_commands)}"
                 "\n\nSuggested command: {cmd}.\n\t(y) accept and continue"
                 "\n\t(s) save example, accept, and continue"
                 "\n{other_options}"
                 "\n\t(back) choose a different action"
                 "\n\t(enter a new command) type your own command to replace the model's suggestion" + user_prompt_end)
user_prompt_3 = ("Given state:\n{construct_state(objective, url, pruned_elements, previous_commands)}"
                 "\n\nSuggested command: {cmd}.\n\t(y) accept and continue"
                 "\n\t(s) save example, accept, and continue"
                 "\n\t(back) choose a different action"
                 "\n\t(enter a new command) type your own command to replace the model's suggestion" + user_prompt_end)


class DialogueState(Enum):
    Unset = None
    Action = "pick action"
    ActionFeedback = "action from feedback"
    Command = "suggest command"
    CommandFeedback = "command from feedback"


class Prompt:

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt

    def __str__(self) -> str:
        return self.prompt


class Command:

    def __init__(self, cmd: str) -> None:
        self.cmd = cmd

    def __str__(self) -> str:
        return self.cmd


def truncate_left(tokenize, prompt, *rest_of_prompt, limit=2048):
    i = 0
    chop_size = 5
    print(f"WARNING: truncating sequence of length {len(tokenize(prompt + ''.join(rest_of_prompt)))} to length {limit}")
    while len(tokenize(prompt + "".join(rest_of_prompt))) > limit:
        prompt = prompt[i * chop_size:]
        i += 1
    return prompt


def split_list_by_separators(l: List[Any], separator_sequences: List[List[Any]]) -> List[List[Any]]:
    """Split a list by a subsequence.
    
    split_list_by_separators(range(7), [[2, 3], [5]]) == [[0, 1], [4], [6]]
    """
    split_list: List[List[Any]] = []
    tmp_seq: List[Any] = []

    i = 0
    while i < len(l):
        item = l[i]
        # if this item may be part of one of the separator_sequences
        if any(item == x[0] for x in separator_sequences):
            for s in filter(lambda x: item == x[0], separator_sequences):
                # if we've found a matching subsequence
                if l[i:i + len(s)] == s:
                    if len(tmp_seq) != 0:
                        split_list.append(tmp_seq)
                    tmp_seq = []
                    i += len(s)
                    break
            else:
                i += 1
        else:
            tmp_seq.append(item)
            i += 1

    if len(tmp_seq) != 0:
        split_list.append(tmp_seq)

    return split_list


def search(co: cohere.Client, query: str, items: List[str], topk: int) -> List[str]:
    embedded_items = np.array(co.embed(texts=items, truncate="RIGHT").embeddings)
    embedded_query = np.array(co.embed(texts=[query], truncate="RIGHT").embeddings[0])
    scores = np.einsum("i,ji->j", embedded_query,
                       embedded_items) / (np.linalg.norm(embedded_query) * np.linalg.norm(embedded_items, axis=1))
    ind = np.argsort(scores)[-topk:]
    return np.flip(np.array(items)[ind], axis=0)


def construct_prev_cmds(previous_commands: List[str]) -> str:
    return "\n".join(f"{i+1}. {x}" for i, x in enumerate(previous_commands)) if previous_commands else "None"


def construct_state(objective: str, url: str, page_elements: List[str], previous_commands: List[str]) -> str:
    state = state_template
    state = state.replace("$objective", objective)
    state = state.replace("$url", url[:100])
    state = state.replace("$previous_commands", construct_prev_cmds(previous_commands))
    return state.replace("$browser_content", "\n".join(page_elements))


def construct_prompt(state: str, examples: List[str]) -> str:
    prompt = prompt_template
    prompt = prompt.replace("$examples", "\n\n".join(examples))
    return prompt.replace("$state", state)


def _fn(x):
    if len(x) == 3:
        option, prompt, self = x
        return_likelihoods = "ALL"
    elif len(x) == 4:
        option, prompt, co, return_likelihoods = x

    while True:
        try:
            if len(co.tokenize(prompt)) > 2048:
                prompt = truncate_left(self.co.tokenize, prompt)
            return (co.generate(prompt=prompt, max_tokens=0, model=MODEL,
                                return_likelihoods=return_likelihoods).generations[0].likelihood, option)
        except cohere.error.CohereError as e:
            print(f"Cohere fucked up: {e}")
            continue
        except ConnectionError as e:
            print(f"Connection error: {e}")
            continue


def choose(co: cohere.Client,
           template: str,
           options: List[Dict[str, str]],
           return_likelihoods: str = "ALL",
           topk: int = 1) -> List[Tuple[int, Dict[str, str]]]:
    """Choose the most likely continuation of `prompt` from a set of `options`.

    Args:
        template (str): a string template with keys that match the dictionaries in `options`
        options (List[Dict[str, str]]): the options to be chosen from

    Returns:
        str: the most likely option from `options`
    """
    num_options = len(options)
    with ThreadPoolExecutor(num_options) as pp:
        _lh = pp.map(
            _fn,
            zip(options, [template.format(**option) for option in options], [co] * num_options,
                [return_likelihoods] * num_options))
    return sorted(_lh, key=lambda x: x[0], reverse=True)[:topk]


def shorten_prompt(co: cohere.Client,
                   objective: str,
                   url: str,
                   elements: List[str],
                   previous_commands: List[str],
                   examples: List[str],
                   *rest_of_prompt,
                   target: int = MAX_SEQ_LEN):
    state = construct_state(objective, url, elements, previous_commands)
    prompt = construct_prompt(state, examples)

    tokenized_prompt = co.tokenize(prompt + "".join(rest_of_prompt))
    tokens = tokenized_prompt.token_strings

    split_tokens = split_list_by_separators(tokens,
                                            [['EX', 'AMP', 'LE'], ["Example"], ["Present", " state", ":", "\n"]])
    example_tokens = split_tokens[1:-1]
    length_of_examples = list(map(len, example_tokens))
    state_tokens = split_tokens[-1]
    state_tokens = list(
        itertools.chain.from_iterable(
            split_list_by_separators(state_tokens, [['----', '----', '----', '----', '--', '\n']])[1:-1]))
    state_tokens = split_list_by_separators(state_tokens, [["\n"]])
    length_of_elements = list(map(len, state_tokens))
    length_of_prompt = len(tokenized_prompt)

    def _fn(i, j):
        state = construct_state(objective, url, elements[:len(elements) - i], previous_commands)
        prompt = construct_prompt(state, examples[j:])

        return state, prompt

    MIN_EXAMPLES = 1
    i, j = (0, 0)
    while (length_of_prompt - sum(length_of_examples)) + sum(
            length_of_examples[j:]) > target and j < len(examples) - MIN_EXAMPLES:
        j += 1

    print(f"num examples: {len(examples) - j}")

    state, prompt = _fn(i, j)
    if len(co.tokenize(prompt + "".join(rest_of_prompt))) <= target:
        return state, prompt

    MIN_ELEMENTS = 7
    while (length_of_prompt - sum(length_of_examples[:j]) - sum(length_of_elements)) + sum(
            length_of_elements[:len(length_of_elements) - i]) > target and i < len(elements) - MIN_ELEMENTS:
        i += 1

    print(f"num elements: {len(length_of_elements) - i}")

    state, prompt = _fn(i, j)

    # last resort, start cutting off the bigging of the prompt
    if len(co.tokenize(prompt + "".join(rest_of_prompt))) > target:
        prompt = truncate_left(co.tokenize, prompt, *rest_of_prompt, limit=target)

    return state, prompt


def gather_examples(co: cohere.Client, state: str, topk: int = 5) -> List[str]:
    """Simple semantic search over a file of past interactions to find the most similar ones."""
    with open("examples.json", "r") as fd:
        history = json.load(fd)

    if len(history) == 0:
        return []

    embeds = [h["embedding"] for h in history]
    examples = [h["example"] for h in history]
    embeds = np.array(embeds)
    embedded_state = np.array(co.embed(texts=[state], truncate="RIGHT").embeddings[0])
    scores = np.einsum("i,ji->j", embedded_state,
                       embeds) / (np.linalg.norm(embedded_state) * np.linalg.norm(embeds, axis=1))
    ind = np.argsort(scores)[-topk:]
    examples = np.array(examples)[ind]

    states = []
    for i in ind:
        h = history[int(i)]
        if all(x in h for x in ["objective", "url", "elements", "previous_commands"]):
            states.append(
                construct_state(objective=h["objective"],
                                url=h["url"],
                                page_elements=h["elements"],
                                previous_commands=h["previous_commands"]))
        else:
            states.append(h["example"])

    return states


def choose_element(co: cohere.Client,
                   template: str,
                   options: List[Dict[str, str]],
                   group_size: int = 10,
                   topk: int = 1) -> List[Dict[str, str]]:
    """A hacky way of choosing the most likely option, while staying within sequence length constraints

        Algo:
        1. chunk `options` into groups of `group_size`
        2. within each group perform a self.choose to get the topk elements (we'll have num_groups*topk elements after this)
        3. flatten and repeat recursively until the number of options is down to topk

        Args:
            template (str): the prompt template with f-string style template tags 
            options (List[Dict[str, str]]): a list of dictionaries containing key-value replacements of the template tags
            group_size (int, optional): The size of each group of options to select from. Defaults to 10.
            topk (int, optional): The topk most likely options to return. Defaults to 1.

        Returns:
            List[Dict[str, str]]: The `topk` most likely elements in `options` according to the model
        """
    num_options = len(options)
    num_groups = int(math.ceil(num_options / group_size))

    if num_options == 0:
        raise Exception()

    choices = []
    for i in range(num_groups):
        group = options[i * group_size:(i + 1) * group_size]
        template_tmp = template.replace("elements", "\n".join(item["elements"] for item in group))
        options_tmp = [{"id": item["id"]} for item in group]

        choice = [x[1] for x in choose(co, template_tmp, options_tmp, topk=topk)]
        chosen_elements = []
        for x in choice:
            chosen_elements.append(list(filter(lambda y: y["id"] == x["id"], group))[0])
        choices.extend(chosen_elements)

    if len(choices) <= topk:
        return choices
    else:
        return choose_element(co, template, choices, group_size, topk)
