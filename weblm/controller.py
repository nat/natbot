import heapq
import itertools
import json
import math
import os
import re
from enum import Enum
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple, Union

import cohere
import numpy as np

MAX_SEQ_LEN = 2000
MAX_NUM_ELEMENTS = 100
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

prioritization_template = """Here are the most relevant elements on the webpage (links, buttons, selects and inputs) to achieve the objective below:
Objective: $objective
URL: $url
Relevant elements:
{element}"""

user_prompt_end = ("\n\t(success) the goal is accomplished"
                   "\n\t(cancel) terminate the session"
                   "\nType a choice and then press enter:")
user_prompt_1 = ("Given web state:\n{state}"
                 "\n\nI have to choose between `clicking` and `typing` here."
                 "\n**I think I should{action}**"
                 "\n\t(y) proceed with this action"
                 "\n\t(n) do the other action" + user_prompt_end)
user_prompt_2 = ("Given state:\n{self._construct_state(url, pruned_elements)}"
                 "\n\nSuggested command: {cmd}.\n\t(y) accept and continue"
                 "\n\t(s) save example, accept, and continue"
                 "\n{other_options}"
                 "\n\t(enter a new command) type your own command to replace the model's suggestion" + user_prompt_end)
user_prompt_3 = ("Given state:\n{self._construct_state(url, pruned_elements)}"
                 "\n\nSuggested command: {self._cmd}.\n\t(y) accept and continue"
                 "\n\t(s) save example, accept, and continue"
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


def truncate_left(tokenize, prompt, *rest_of_prompt, limit=2048):
    i = 0
    chop_size = 5
    print(f"WARNING: truncating sequence of length {len(tokenize(prompt + ''.join(rest_of_prompt)))} to length {limit}")
    while len(tokenize(prompt + "".join(rest_of_prompt))) > limit:
        prompt = prompt[i * chop_size:]
        i += 1
    return prompt


def split_list_by_separators(l: List[Any], separator_sequences: List[List[Any]]) -> List[List[Any]]:
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

    def __init__(self, co: cohere.Client, objective: str):
        """
        Args:
            co (cohere.Client): a Cohere Client
            objective (str): the objective to accomplish
        """
        self.co = co
        self.objective = objective
        self.previous_commands: List[str] = []
        self.moments: List[Tuple[str, str]] = []
        self.reset_state()

    def is_running(self):
        return self._step != DialogueState.Unset

    def reset_state(self):
        self._step = DialogueState.Unset
        self._action = None
        self._cmd = None
        self._chosen_elements: List[Dict[str, str]] = []
        self._prioritized_elements = None
        self._prioritized_elements_hash = None

    def success(self):
        for state, command in self.moments:
            self._save_example(state, command)

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
        with Pool(num_options) as pp:
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
        with open("examples.json", "r") as fd:
            examples = json.load(fd)

        if len(examples) == 0:
            return []

        embeds, examples = zip(*examples)
        embeds = np.array(embeds)
        embedded_state = np.array(self.co.embed(texts=[state], truncate="RIGHT").embeddings[0])
        scores = np.einsum("i,ji->j", embedded_state,
                           embeds) / (np.linalg.norm(embedded_state) * np.linalg.norm(embeds, axis=1))
        ind = np.argsort(scores)[-topk:]
        examples = np.array(examples)[ind]

        return examples

    def _construct_prev_cmds(self) -> str:
        return "\n".join(
            f"{i+1}. {x}" for i, x in enumerate(self.previous_commands)) if self.previous_commands else "None"

    def _construct_state(self, url: str, page_elements: List[str]) -> str:
        state = state_template
        state = state.replace("$objective", self.objective)
        state = state.replace("$url", url[:100])
        state = state.replace("$previous_commands", self._construct_prev_cmds())
        return state.replace("$browser_content", "\n".join(page_elements))

    def _construct_prompt(self, state: str, examples: List[str]) -> str:
        prompt = prompt_template
        prompt = prompt.replace("$examples", "\n\n".join(examples))
        return prompt.replace("$state", state)

    def _save_example(self, state: str, command: str):
        example = ("Example:\n"
                   f"{state}\n"
                   f"Next Command: {command}\n"
                   "----")
        print(f"Example being saved:\n{example}")
        with open("examples.json", "r") as fd:
            embeds_examples = json.load(fd)
            embeds, examples = zip(*embeds_examples)
            embeds, examples = list(embeds), list(examples)

        if example in examples:
            print("example already exists")
            return

        examples.append(example)
        embeds.append(self.co.embed(texts=[example]).embeddings[0])

        embeds_examples = list(zip(embeds, examples))
        with open("examples_tmp.json", "w") as fd:
            json.dump(embeds_examples, fd)
        os.replace("examples_tmp.json", "examples.json")

    def _shorten_prompt(self, url, elements, examples, *rest_of_prompt, target: int = MAX_SEQ_LEN):
        state = self._construct_state(url, elements)
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
            state = self._construct_state(url, elements[:len(elements) - i])
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
        prioritization = prioritization_template
        prioritization = prioritization.replace("$objective", self.objective)
        prioritization = prioritization.replace("$url", url)

        self._prioritized_elements = self.choose(prioritization, [{
            "element": x
        } for x in page_elements],
                                                 topk=len(page_elements))
        self._prioritized_elements = [x[1]["element"] for x in self._prioritized_elements][:MAX_NUM_ELEMENTS]
        self._prioritized_elements_hash = hash(frozenset(page_elements))
        self._step = DialogueState.Action
        print(self._prioritized_elements)

    def pick_action(self, url: str, page_elements: List[str], response: str = None):
        # this strategy for action selection does not work very well, TODO improve this

        if self._step not in [DialogueState.Action, DialogueState.ActionFeedback]:
            return

        state = self._construct_state(url, self._prioritized_elements)
        examples = self.gather_examples(state)
        prompt = self._construct_prompt(state, examples)

        if self._step == DialogueState.Action:
            action = " click"
            if any(y in x for y in TYPEABLE for x in page_elements):
                state, prompt = self._shorten_prompt(url, self._prioritized_elements, examples, target=MAX_SEQ_LEN)

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
                    print(text)
                except cohere.error.CohereError as e:
                    print(f"Cohere fucked up: {e}")
                    continue
        else:
            text = ""

        return (self._action + chosen_element + text).strip()

    def generate_command(self, url: str, pruned_elements: List[str], response: str = None):
        state = self._construct_state(url, pruned_elements)
        examples = self.gather_examples(state)
        prompt = self._construct_prompt(state, examples)

        if self._step == DialogueState.Command:
            if len(pruned_elements) == 1:
                chosen_element = " " + " ".join(pruned_elements[0].split(" ")[:2])
                self._chosen_elements = [{"id": chosen_element}]
            else:
                state = self._construct_state(url, ["$elements"])
                prompt = self._construct_prompt(state, examples)

                state, prompt = self._shorten_prompt(url, ["$elements"], examples, self._action)

                group_size = 40
                self._chosen_elements = self.choose_element(
                    prompt + self._action + "{id}",
                    list(map(lambda x: {
                        "id": " " + " ".join(x.split(" ")[:2]),
                        "elements": x
                    }, pruned_elements)),
                    group_size,
                    topk=5)
                print(self._chosen_elements)
                chosen_element = self._chosen_elements[0]["id"]

                state = self._construct_state(url, pruned_elements)
                prompt = self._construct_prompt(state, examples)

                state, prompt = self._shorten_prompt(url, pruned_elements, examples, self._action, chosen_element)

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
                state, prompt = self._shorten_prompt(url, pruned_elements, examples, self._action, chosen_element)
                return Prompt(f"{prompt}\n\nPlease respond with 'y' or 's'")
            elif response == "recrawl":
                return Prompt(eval(f'f"""{user_prompt_3}"""'))
            elif response == "elements":
                return Prompt("\n".join(str(d) for d in self._chosen_elements))

            if re.match(r'\d+', response):
                chosen_element = self._chosen_elements[int(response) - 1]["id"]
                state, prompt = self._shorten_prompt(url, pruned_elements, examples, self._action, chosen_element)
                self._cmd = self._get_cmd_prediction(prompt, chosen_element)
                if "type" in self._action:
                    return Prompt(eval(f'f"""{user_prompt_3}"""'))
            elif response != "y" and response != "s":
                self._cmd = response

            cmd_pattern = r"(click|type) (link|button|input|select) [\d]+( \"\w+\")?"
            if not re.match(cmd_pattern, self._cmd):
                return Prompt(f"Invalid command '{self._cmd}'. Must match regex '{cmd_pattern}'. Try again...")

            if response == "s":
                self._save_example(state=self._construct_state(url, pruned_elements[:50]), command=self._cmd)

        self.moments.append((self._construct_state(url, pruned_elements), self._cmd))
        self.previous_commands.append(self._cmd)

        cmd = Command(self._cmd.strip())
        self.reset_state()
        return cmd

    def step(self, url: str, page_elements: List[str], response: str = None) -> Union[Prompt, Command]:
        self._step = DialogueState.Action if self._step == DialogueState.Unset else self._step

        if self._prioritized_elements is None or self._prioritized_elements_hash != hash(frozenset(page_elements)):
            self._generate_prioritization(page_elements, url)

        action_or_prompt = self.pick_action(url, page_elements, response)

        if isinstance(action_or_prompt, Prompt):
            return action_or_prompt

        if "click" in self._action:
            pruned_elements = list(filter(lambda x: any(x.startswith(y) for y in CLICKABLE),
                                          self._prioritized_elements))
        elif "type" in self._action:
            pruned_elements = list(filter(lambda x: any(x.startswith(y) for y in TYPEABLE), self._prioritized_elements))

        return self.generate_command(url, pruned_elements, response)
