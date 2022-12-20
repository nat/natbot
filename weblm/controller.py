import csv
import heapq
import itertools
import json
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, DefaultDict, Dict, List, Tuple, Union

import cohere
import numpy as np
from requests.exceptions import ConnectionError

MAX_SEQ_LEN = 2000
MAX_NUM_ELEMENTS = 50
TYPEABLE = ["input", "select"]
CLICKABLE = ["link", "button"]
MODEL = "xlarge"

help_msg = """Welcome to WebLM!

The goal of this project is build a system that takes an objective from the user, and operates a browser to carry it out.

For example:
- book me a table for 2 at bar isabel next wednesday at 7pm
- i need a flight from SF to London on Oct 15th nonstop
- buy me more whitening toothpaste from amazon and send it to my apartment

WebLM learns to carry out tasks *by demonstration*. That means that you'll need to guide it and correct it when it goes astray. Over time, the more people who use it, the more tasks it's used for, WebLM will become better and better and rely less and less on user input.

To control the system:
- You can see what the model sees at each step by looking at the list of elements the model can interact with
- show: You can also see a picture of the browser window by typing `show`
- goto: You can go to a specific webpage by typing `goto www.yourwebpage.com`
- success: When the model has succeeded at the task you set out (or gotten close enough), you can teach the model by typing `success` and it will save it's actions to use in future interations
- cancel: If the model is failing or you made a catastrophic mistake you can type `cancel` to kill the session
- help: Type `help` to show this message

Everytime you use WebLM it will improve. If you want to contribute to the project and help us build join the discord (https://discord.com/invite/co-mmunity) or send an email to weblm@cohere.com"""

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

prioritization_template = """$examples
---
Here are the most relevant elements on the webpage (links, buttons, selects and inputs) to achieve the objective below:
Objective: $objective
URL: $url
Relevant elements:
{element}"""

priorit_tmp = ("Objective: {objective}"
               "\nURL: {url}"
               "\nRelevant elements:"
               "\n{elements}")
user_prompt_end = ("\n\t(success) the goal is accomplished"
                   "\n\t(cancel) terminate the session"
                   "\nType a choice and then press enter:")
user_prompt_1 = ("Given web state:\n{state}"
                 "\n\nI have to choose between `clicking` and `typing` here."
                 "\n**I think I should{action}**"
                 "\n\t(y) proceed with this action"
                 "\n\t(n) do the other action" + user_prompt_end)
user_prompt_2 = ("Given state:\n{self._construct_state(self.objective, url, pruned_elements, self.previous_commands)}"
                 "\n\nSuggested command: {cmd}.\n\t(y) accept and continue"
                 "\n\t(s) save example, accept, and continue"
                 "\n{other_options}"
                 "\n\t(back) choose a different action"
                 "\n\t(enter a new command) type your own command to replace the model's suggestion" + user_prompt_end)
user_prompt_3 = ("Given state:\n{self._construct_state(self.objective, url, pruned_elements, self.previous_commands)}"
                 "\n\nSuggested command: {self._cmd}.\n\t(y) accept and continue"
                 "\n\t(s) save example, accept, and continue"
                 "\n\t(back) choose a different action"
                 "\n\t(enter a new command) type your own command to replace the model's suggestion" + user_prompt_end)


def _fn(x):
    if len(x) == 3:
        option, prompt, self = x
        return_likelihoods = "ALL"
    elif len(x) == 4:
        option, prompt, self, return_likelihoods = x

    while True:
        try:
            if len(self.co.tokenize(prompt)) > 2048:
                prompt = truncate_left(self.co.tokenize, prompt)
            return (self.co.generate(prompt=prompt, max_tokens=0, model=MODEL,
                                     return_likelihoods=return_likelihoods).generations[0].likelihood, option)
        except cohere.error.CohereError as e:
            print(f"Cohere fucked up: {e}")
            continue
        except ConnectionError as e:
            print(f"Connection error: {e}")
            continue


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


class DialogueState(Enum):
    Unset = None
    Action = "pick action"
    ActionFeedback = "action from feedback"
    Command = "suggest command"
    CommandFeedback = "command from feedback"


class Controller:
    """A Cohere-powered controller that takes in a browser state and produces and action.

    The basic outline of this Controller's strategy is:
    1. receive page content from browser
    2. prioritise elements on page based on how relevant they are to the objective
    3. look up similar states from the past
    4. choose between clicking and typing
    5. choose what element to click or what element to type in
    """

    def __init__(self, co: cohere.Client, objective: str):
        """
        Args:
            co (cohere.Client): a Cohere Client
            objective (str): the objective to accomplish
        """
        self.co = co
        self.objective = objective
        self.previous_commands: List[str] = []
        self.moments: List[Tuple[str, str, str, List[str]]] = []
        self.user_responses: DefaultDict[str, int] = defaultdict(int)
        self.reset_state()

    def is_running(self):
        return self._step != DialogueState.Unset

    def reset_state(self):
        self._step = DialogueState.Unset
        self._action = None
        self._cmd = None
        self._chosen_elements: List[Dict[str, str]] = []
        self._prioritized_elements = None
        self._pruned_prioritized_elements = None
        self._prioritized_elements_hash = None
        self._page_elements = None
        self._error = None

    def success(self):
        for url, elements, command, previous_commands in self.moments:
            self._save_example(objective=self.objective,
                               url=url,
                               elements=elements,
                               command=command,
                               previous_commands=previous_commands)

    def choose(self,
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
                zip(options, [template.format(**option) for option in options], [self] * num_options,
                    [return_likelihoods] * num_options))
        return sorted(_lh, key=lambda x: x[0], reverse=True)[:topk]

    def choose_element(self,
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

            choice = [x[1] for x in self.choose(template_tmp, options_tmp, topk=topk)]
            chosen_elements = []
            for x in choice:
                chosen_elements.append(list(filter(lambda y: y["id"] == x["id"], group))[0])
            choices.extend(chosen_elements)

        if len(choices) <= topk:
            return choices
        else:
            return self.choose_element(template, choices, group_size, topk)

    def gather_examples(self, state: str, topk: int = 5) -> List[str]:
        """Simple semantic search over a file of past interactions to find the most similar ones."""
        with open("examples.json", "r") as fd:
            history = json.load(fd)

        if len(history) == 0:
            return []

        embeds = [h["embedding"] for h in history]
        examples = [h["example"] for h in history]
        embeds = np.array(embeds)
        embedded_state = np.array(self.co.embed(texts=[state], truncate="RIGHT").embeddings[0])
        scores = np.einsum("i,ji->j", embedded_state,
                           embeds) / (np.linalg.norm(embedded_state) * np.linalg.norm(embeds, axis=1))
        ind = np.argsort(scores)[-topk:]
        examples = np.array(examples)[ind]

        states = []
        for i in ind:
            h = history[int(i)]
            if all(x in h for x in ["objective", "url", "elements", "previous_commands"]):
                states.append(
                    self._construct_state(objective=h["objective"],
                                          url=h["url"],
                                          page_elements=h["elements"],
                                          previous_commands=h["previous_commands"]))
            else:
                states.append(h["example"])

        return states

    def gather_prioritisation_examples(self, state: str, topk: int = 6, num_elements: int = 3) -> List[str]:
        """Simple semantic search over a file of past interactions to find the most similar ones."""
        with open("examples.json", "r") as fd:
            history = json.load(fd)

        if len(history) == 0:
            return []

        embeds = [h["embedding"] for h in history]
        examples = [h["example"] for h in history]
        embeds = np.array(embeds)
        embedded_state = np.array(self.co.embed(texts=[state], truncate="RIGHT").embeddings[0])
        scores = np.einsum("i,ji->j", embedded_state,
                           embeds) / (np.linalg.norm(embedded_state) * np.linalg.norm(embeds, axis=1))
        ind = np.argsort(scores)[-topk:]
        examples = np.array(examples)[ind]

        prioritisation_examples = []
        for i, h in enumerate(history):
            if i in ind:
                if all(x in h for x in ["objective", "command", "url", "elements"]):
                    # make sure the element relevant to the next command is included
                    elements = h["elements"]
                    command_element = " ".join(h["command"].split()[1:3])
                    command_element = list(filter(lambda x: command_element in x, elements))
                    assert len(command_element) == 1
                    command_element = command_element[0]

                    if not command_element in elements[:num_elements]:
                        elements = [command_element] + elements[:-1]

                    elements = elements[:num_elements]

                    objective = h["objective"]
                    url = h["url"]
                    elements = '\n'.join(elements)
                    prioritisation_example = eval(f'f"""{priorit_tmp}"""')
                    prioritisation_examples.append(prioritisation_example)

        return prioritisation_examples

    def _construct_prev_cmds(self, previous_commands: List[str]) -> str:
        return "\n".join(f"{i+1}. {x}" for i, x in enumerate(previous_commands)) if previous_commands else "None"

    def _construct_state(self, objective: str, url: str, page_elements: List[str], previous_commands: List[str]) -> str:
        state = state_template
        state = state.replace("$objective", objective)
        state = state.replace("$url", url[:100])
        state = state.replace("$previous_commands", self._construct_prev_cmds(previous_commands))
        return state.replace("$browser_content", "\n".join(page_elements))

    def _construct_prompt(self, state: str, examples: List[str]) -> str:
        prompt = prompt_template
        prompt = prompt.replace("$examples", "\n\n".join(examples))
        return prompt.replace("$state", state)

    def _save_example(self, objective: str, url: str, elements: List[str], command: str, previous_commands: List[str]):
        state = self._construct_state(objective, url, elements[:MAX_NUM_ELEMENTS], previous_commands)
        example = ("Example:\n"
                   f"{state}\n"
                   f"Next Command: {command}\n"
                   "----")
        print(f"Example being saved:\n{example}")
        with open("examples.json", "r") as fd:
            history = json.load(fd)
            examples = [h["example"] for h in history]

        if example in examples:
            print("example already exists")
            return

        history.append({
            "example": example,
            "embedding": self.co.embed(texts=[example]).embeddings[0],
            "url": url,
            "elements": elements,
            "command": command,
            "previous_commands": previous_commands,
            "objective": objective,
        })

        with open("examples_tmp.json", "w") as fd:
            json.dump(history, fd)
        os.replace("examples_tmp.json", "examples.json")

    def _construct_responses(self):
        keys_to_save = ["y", "n", "s", "command", "success", "cancel"]
        responses_to_save = defaultdict(int)
        for key, value in self.user_responses.items():
            if key in keys_to_save:
                responses_to_save[key] = value
            elif key not in keys_to_save and key:
                responses_to_save["command"] += 1

        self.user_responses = responses_to_save
        print(f"Responses being saved:\n{dict(responses_to_save)}")

    def save_responses(self):
        keys_to_save = ["y", "n", "s", "command", "success", "cancel"]
        # Check if data file already exists
        responses_filepath = "responses.csv"
        if os.path.isfile(responses_filepath):
            print("File exists")
            with open(responses_filepath, "a+") as fd:
                wr = csv.writer(fd, quoting=csv.QUOTE_ALL)
                wr.writerow([self.user_responses[key] for key in keys_to_save])
        else:
            print("No data available")
            with open(responses_filepath, "w+") as fd:
                wr = csv.writer(fd, quoting=csv.QUOTE_ALL)
                wr.writerow(keys_to_save)
                wr.writerow([self.user_responses[key] for key in keys_to_save])

    def _shorten_prompt(self,
                        objective: str,
                        url: str,
                        elements: List[str],
                        previous_commands: List[str],
                        examples: List[str],
                        *rest_of_prompt,
                        target: int = MAX_SEQ_LEN):
        state = self._construct_state(objective, url, elements, previous_commands)
        prompt = self._construct_prompt(state, examples)

        tokenized_prompt = self.co.tokenize(prompt + "".join(rest_of_prompt))
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
            state = self._construct_state(objective, url, elements[:len(elements) - i], previous_commands)
            prompt = self._construct_prompt(state, examples[j:])

            return state, prompt

        MIN_EXAMPLES = 1
        i, j = (0, 0)
        while (length_of_prompt - sum(length_of_examples)) + sum(
                length_of_examples[j:]) > target and j < len(examples) - MIN_EXAMPLES:
            j += 1

        print(f"num examples: {len(examples) - j}")

        state, prompt = _fn(i, j)
        if len(self.co.tokenize(prompt + "".join(rest_of_prompt))) <= target:
            return state, prompt

        MIN_ELEMENTS = 7
        while (length_of_prompt - sum(length_of_examples[:j]) - sum(length_of_elements)) + sum(
                length_of_elements[:len(length_of_elements) - i]) > target and i < len(elements) - MIN_ELEMENTS:
            i += 1

        print(f"num elements: {len(length_of_elements) - i}")

        state, prompt = _fn(i, j)

        # last resort, start cutting off the bigging of the prompt
        if len(self.co.tokenize(prompt + "".join(rest_of_prompt))) > target:
            prompt = truncate_left(self.co.tokenize, prompt, *rest_of_prompt, limit=target)

        return state, prompt

    def _generate_prioritization(self, page_elements: List[str], url: str):
        state = self._construct_state(self.objective, url, page_elements, self.previous_commands)
        examples = self.gather_prioritisation_examples(state)

        prioritization = prioritization_template
        prioritization = prioritization.replace("$examples", "\n---\n".join(examples))
        prioritization = prioritization.replace("$objective", self.objective)
        prioritization = prioritization.replace("$url", url)

        self._prioritized_elements = self.choose(prioritization, [{
            "element": x
        } for x in page_elements],
                                                 topk=len(page_elements))
        self._prioritized_elements = [x[1]["element"] for x in self._prioritized_elements]
        self._prioritized_elements_hash = hash(frozenset(page_elements))
        self._pruned_prioritized_elements = self._prioritized_elements[:MAX_NUM_ELEMENTS]
        self._step = DialogueState.Action
        print(self._prioritized_elements)

    def pick_action(self, url: str, page_elements: List[str], response: str = None):
        # this strategy for action selection does not work very well, TODO improve this

        if self._step not in [DialogueState.Action, DialogueState.ActionFeedback]:
            return

        state = self._construct_state(self.objective, url, self._pruned_prioritized_elements, self.previous_commands)
        examples = self.gather_examples(state)
        prompt = self._construct_prompt(state, examples)

        if self._step == DialogueState.Action:
            action = " click"
            if any(y in x for y in TYPEABLE for x in page_elements):
                elements = list(
                    filter(lambda x: any(x.startswith(y) for y in CLICKABLE + TYPEABLE),
                           self._pruned_prioritized_elements))

                state, prompt = self._shorten_prompt(self.objective,
                                                     url,
                                                     elements,
                                                     self.previous_commands,
                                                     examples,
                                                     target=MAX_SEQ_LEN)

                action = self.choose(prompt + "{action}", [
                    {
                        "action": " click",
                    },
                    {
                        "action": " type",
                    },
                ], topk=2)

                # if the model is confident enough, just assume the suggested action is correct
                if (action[0][0] - action[1][0]) / -action[1][0] > 1.:
                    action = action[0][1]["action"]
                else:
                    action = action[0][1]["action"]
                    self._action = action
                    self._step = DialogueState.ActionFeedback
                    return Prompt(eval(f'f"""{user_prompt_1}"""'))

            self._action = action
            self._step = DialogueState.Command
        elif self._step == DialogueState.ActionFeedback:
            if response == "y":
                pass
            elif response == "n":
                if "click" in self._action:
                    self._action = " type"
                elif "type" in self._action:
                    self._action = " click"
            elif response == "examples":
                examples = "\n".join(examples)
                return Prompt(f"Examples:\n{examples}\n\n"
                              "Please respond with 'y' or 'n'")
            elif re.match(r'search (.+)', response):
                query = re.match(r'search (.+)', response).group(1)
                results = search(self.co, query, self._page_elements, topk=50)
                return Prompt(f"Query: {query}\nResults:\n{results}\n\n"
                              "Please respond with 'y' or 'n'")
            else:
                return Prompt("Please respond with 'y' or 'n'")

            self._step = DialogueState.Command

    def _get_cmd_prediction(self, prompt: str, chosen_element: str) -> str:
        if "type" in self._action:
            text = None
            while text is None:
                try:
                    num_tokens = 20
                    if len(self.co.tokenize(prompt)) > 2048 - num_tokens:
                        print(f"WARNING: truncating sequence of length {len(self.co.tokenize(prompt))}")
                        prompt = truncate_left(self.co.tokenize,
                                               prompt,
                                               self._action,
                                               chosen_element,
                                               limit=2048 - num_tokens)

                    print(len(self.co.tokenize(prompt + self._action + chosen_element)))
                    text = max(self.co.generate(prompt=prompt + self._action + chosen_element,
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

        return (self._action + chosen_element + text).strip()

    def generate_command(self, url: str, pruned_elements: List[str], response: str = None):
        state = self._construct_state(self.objective, url, pruned_elements, self.previous_commands)
        examples = self.gather_examples(state)
        prompt = self._construct_prompt(state, examples)

        if self._step == DialogueState.Command:
            if len(pruned_elements) == 1:
                chosen_element = " " + " ".join(pruned_elements[0].split(" ")[:2])
                self._chosen_elements = [{"id": chosen_element}]
            else:
                state = self._construct_state(self.objective, url, ["$elements"], self.previous_commands)
                prompt = self._construct_prompt(state, examples)

                state, prompt = self._shorten_prompt(self.objective, url, ["$elements"], self.previous_commands,
                                                     examples, self._action)

                group_size = 20
                self._chosen_elements = self.choose_element(
                    prompt + self._action + "{id}",
                    list(map(lambda x: {
                        "id": " " + " ".join(x.split(" ")[:2]),
                        "elements": x
                    }, pruned_elements)),
                    group_size,
                    topk=5)
                chosen_element = self._chosen_elements[0]["id"]

                state = self._construct_state(self.objective, url, pruned_elements, self.previous_commands)
                prompt = self._construct_prompt(state, examples)

                state, prompt = self._shorten_prompt(self.objective, url, pruned_elements, self.previous_commands,
                                                     examples, self._action, chosen_element)

            cmd = self._get_cmd_prediction(prompt, chosen_element)

            self._cmd = cmd
            self._step = DialogueState.CommandFeedback
            other_options = "\n".join(
                f"\t({i+2}){self._action}{x['id']}" for i, x in enumerate(self._chosen_elements[1:]))
            return Prompt(eval(f'f"""{user_prompt_2}"""'))
        elif self._step == DialogueState.CommandFeedback:
            if response == "examples":
                examples = "\n".join(examples)
                return Prompt(f"Examples:\n{examples}\n\n"
                              "Please respond with 'y' or 's'")
            elif response == "prompt":
                chosen_element = self._chosen_elements[0]["id"]
                state, prompt = self._shorten_prompt(self.objective, url, pruned_elements, self.previous_commands,
                                                     examples, self._action, chosen_element)
                return Prompt(f"{prompt}\n\nPlease respond with 'y' or 's'")
            elif response == "recrawl":
                return Prompt(eval(f'f"""{user_prompt_3}"""'))
            elif response == "elements":
                return Prompt("\n".join(str(d) for d in self._chosen_elements))
            elif re.match(r'search (.+)', response):
                query = re.match(r'search (.+)', response).group(1)
                results = search(self.co, query, self._page_elements, topk=50)
                return Prompt(f"Query: {query}\nResults:\n{results}\n\n"
                              "Please respond with 'y' or 'n'")

            if re.match(r'\d+', response):
                chosen_element = self._chosen_elements[int(response) - 1]["id"]
                state, prompt = self._shorten_prompt(self.objective, url, pruned_elements, self.previous_commands,
                                                     examples, self._action, chosen_element)
                self._cmd = self._get_cmd_prediction(prompt, chosen_element)
                if "type" in self._action:
                    return Prompt(eval(f'f"""{user_prompt_3}"""'))
            elif response != "y" and response != "s":
                self._cmd = response

            cmd_pattern = r"(click|type) (link|button|input|select) [\d]+( \"\w+\")?"
            if not re.match(cmd_pattern, self._cmd):
                return Prompt(f"Invalid command '{self._cmd}'. Must match regex '{cmd_pattern}'. Try again...")

            if response == "s":
                self._save_example(objective=self.objective,
                                   url=url,
                                   elements=self._prioritized_elements,
                                   command=self._cmd,
                                   previous_commands=self.previous_commands)

        self.moments.append((url, self._prioritized_elements, self._cmd, self.previous_commands.copy()))
        self.previous_commands.append(self._cmd)

        cmd = Command(self._cmd.strip())
        self.reset_state()
        return cmd

    def step(self, url: str, page_elements: List[str], response: str = None) -> Union[Prompt, Command]:
        if self._error is not None:
            if response == "c":
                self._error = None
            elif response == "success":
                self.success()
                raise self._error from None
            elif response == "cancel":
                raise self._error from None
            else:
                return Prompt("Response not recognized"
                              "\nPlease choose one of the following:"
                              "\n\t(c) ignore exception and continue" + user_prompt_end)

        try:
            self._step = DialogueState.Action if self._step == DialogueState.Unset else self._step
            self._page_elements = page_elements

            if self._prioritized_elements is None or self._prioritized_elements_hash != hash(frozenset(page_elements)):
                self._generate_prioritization(page_elements, url)

            self.user_responses[response] += 1
            self._construct_responses()
            action_or_prompt = self.pick_action(url, page_elements, response)

            if isinstance(action_or_prompt, Prompt):
                return action_or_prompt

            if "click" in self._action:
                pruned_elements = list(
                    filter(lambda x: any(x.startswith(y) for y in CLICKABLE), self._pruned_prioritized_elements))
            elif "type" in self._action:
                pruned_elements = list(
                    filter(lambda x: any(x.startswith(y) for y in TYPEABLE), self._pruned_prioritized_elements))

            return self.generate_command(url, pruned_elements, response)
        except Exception as e:
            self._error = e
            return Prompt(f"Caught exception:\n{e}"
                          "\nPlease choose one of the following:"
                          "\n\t(c) ignore exception and continue" + user_prompt_end)
