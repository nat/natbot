import json
import math
import os
import re
from enum import Enum
from multiprocessing import Pool
from typing import Dict, List, Union

import cohere
import numpy as np

MAX_SEQ_LEN = 2020

prompt_template = """Given:
    (1) an objective that you are trying to achieve
    (2) the URL of your current web page
    (3) a simplified text description of what's visible in the browser window

Your commands are:
    click X - click on element X.
    type X "TEXT" - type the specified text into input X

Here are some examples:

$examples

$state
YOUR COMMAND:"""

state_template = """==================================================
CURRENT BROWSER CONTENT:
------------------
$browser_content
------------------
OBJECTIVE: $objective
CURRENT URL: $url
PREVIOUS STEPS:
$previous_commands"""


def _fn(x):
    option, prompt, self = x
    while True:
        try:
            print(len(self.co.tokenize(prompt)))
            return (self.co.generate(prompt=prompt, max_tokens=0, model="xlarge",
                                     return_likelihoods="ALL").generations[0].likelihood, option)
        except cohere.error.CohereError:
            print("Cohere fucked up")
            continue


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
        self.reset_state()

    def is_running(self):
        return self._step != DialogueState.Unset

    def reset_state(self):
        self._step = DialogueState.Unset
        self._action = None
        self._cmd = None

    def choose(self, template: str, options: List[Dict[str, str]]) -> Dict[str, str]:
        """Choose the most likely continuation of `prompt` from a set of `options`.

        Args:
            template (str): a string template with keys that match the dictionaries in `options`
            options (List[Dict[str, str]]): the options to be chosen from

        Returns:
            str: the most likely option from `options`
        """
        num_options = len(options)
        with Pool() as pp:
            _lh = pp.map(_fn, zip(options, [template.format(**option) for option in options], [self] * num_options))
        return max(_lh, key=lambda x: x[0])[1]

    def choose_element(self, template: str, options: List[Dict[str, str]], group_size: int = 10) -> Dict[str, str]:
        num_options = len(options)
        num_groups = int(math.ceil(num_options / group_size))

        choices = []
        for i in range(num_groups):
            group = options[i * group_size:(i + 1) * group_size]
            template_tmp = template.replace("elements", "\n".join(item["elements"] for item in group))
            options_tmp = [{"id": item["id"]} for item in group]

            choice = self.choose(template_tmp, options_tmp)
            choices.append(list(filter(lambda x: x["id"] == choice["id"], group))[0])

        if len(choices) == 1:
            return choices[0]
        else:
            return self.choose_element(template, choices, group_size)

    def gather_examples(self, state: str, topk: int = 2) -> str:
        with open("examples.json", "r") as fd:
            examples = json.load(fd)

        embeds, examples = zip(*examples)
        embeds = np.array(embeds)
        embedded_state = np.array(self.co.embed(texts=[state]).embeddings[0])
        scores = np.einsum("i,ji->j", embedded_state, embeds)
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

    def _construct_prompt(self, state: str, examples: str) -> str:
        prompt = prompt_template
        prompt = prompt.replace("$examples", "\n\n".join(examples))
        return prompt.replace("$state", state)

    def _save_example(self, state: str, command: str):
        example = ("EXAMPLE:\n"
                   f"{state}\n"
                   f"YOUR COMMAND: {command}\n"
                   "==================================================")
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

    def _shorten_prompt(self, url, elements, examples, *rest_of_prompt):
        state = self._construct_state(url, elements)
        prompt = self._construct_prompt(state, examples)
        j = 0

        if len(self.co.tokenize(prompt + "".join(rest_of_prompt))) - len(self.co.tokenize(state)) > MAX_SEQ_LEN:
            j += 1

        while len(self.co.tokenize(prompt + "".join(rest_of_prompt))) > MAX_SEQ_LEN:
            state = self._construct_state(url, elements)
            prompt = self._construct_prompt(state, examples[j:])

            i = 1
            while len(self.co.tokenize(prompt + "".join(rest_of_prompt))) > MAX_SEQ_LEN and i < len(elements):
                print(i, j, len(self.co.tokenize(prompt + "".join(rest_of_prompt))))
                print(elements[:-i])
                state = self._construct_state(url, elements[:-i])
                prompt = self._construct_prompt(state, examples[j:])
                i += 1

            if len(self.co.tokenize(prompt + "".join(rest_of_prompt))) <= MAX_SEQ_LEN:
                print(i, j, len(self.co.tokenize(prompt + "".join(rest_of_prompt))))
                return state, prompt

            j += 1

        return state, prompt

    def cli_step(self, url: str, page_elements: List[str]):
        state = self._construct_state(url, page_elements)
        examples = self.gather_examples(state)
        prompt = self._construct_prompt(state, examples)
        if any("input" in x for x in page_elements):
            state, prompt = self._shorten_prompt(url, page_elements, examples)
            action = self.choose(prompt + "{action}", [
                {
                    "action": " click"
                },
                {
                    "action": " type"
                },
            ])["action"]

            print(f"Given state:\n{state}\n\nI think I should{action}")
            i = input("Is that right? (enter/y) yes, (n) no")
            if i == "n":
                if "click" in action:
                    action = " type"
                elif "type" in action:
                    action = " click"
        else:
            action = " click"

        if "click" in action:
            pruned_elements = list(filter(lambda x: "link" in x or "button" in x, page_elements))
        elif "type" in action:
            pruned_elements = list(filter(lambda x: "input" in x, page_elements))

        if len(pruned_elements) == 1:
            chosen_element = " " + " ".join(pruned_elements[0].split(" ")[:2])
        else:
            state = self._construct_state(url, pruned_elements)
            examples = self.gather_examples(state)
            state = self._construct_state(url, ["$elements"])
            prompt = self._construct_prompt(state, examples)

            # print(state)
            state, prompt = self._shorten_prompt(url, ["$elements"], examples, action)

            group_size = 20
            chosen_element = self.choose_element(
                prompt + action + "{id}",
                list(map(lambda x: {
                    "id": " " + " ".join(x.split(" ")[:2]),
                    "elements": x
                }, pruned_elements)),
                group_size,
            )["id"]

            state = self._construct_state(url, pruned_elements)
            prompt = self._construct_prompt(state, examples)

            state, prompt = self._shorten_prompt(url, pruned_elements, examples, action, chosen_element)

        print(prompt + action + chosen_element)
        text = None
        while text is None:
            try:
                print(len(self.co.tokenize(prompt + action + chosen_element)))
                text = max(self.co.generate(prompt=prompt + action + chosen_element,
                                            model="xlarge",
                                            temperature=0.5,
                                            num_generations=1,
                                            max_tokens=20,
                                            stop_sequences=["\n"],
                                            return_likelihoods="GENERATION").generations,
                           key=lambda x: x.likelihood).text
            except cohere.error.CohereError:
                print("Cohere fucked up")
                continue
        cmd = (action + chosen_element + text).strip()
        response = input(f"Suggested command: {cmd}.\n\t(enter) accept and continue"
                         "\n\t(s) save example, accept, and continue"
                         "\n\t(enter a new command) type your own command to replace the model's suggestion"
                         "\nType a choice and then press enter:")
        if response != "" and response != "s":
            cmd = response

        if response == "s" or response != "":
            self._save_example(state=self._construct_state(url, pruned_elements), command=cmd)

        self.previous_commands.append(cmd)

        return cmd

    def dialogue_step(self, url: str, page_elements: List[str], response: str = None):
        self._step = DialogueState.Action if self._step == DialogueState.Unset else self._step

        state = self._construct_state(url, page_elements)
        examples = self.gather_examples(state)
        prompt = self._construct_prompt(state, examples)

        if self._step == DialogueState.Action:
            if any("input" in x for x in page_elements):
                state, prompt = self._shorten_prompt(url, page_elements, examples)
                action = self.choose(prompt + "{action}", [
                    {
                        "action": " click"
                    },
                    {
                        "action": " type"
                    },
                ])["action"]

                self._action = action
                self._step = DialogueState.ActionFeedback
                return Prompt(f"Given web state:\n{self._construct_state(url, page_elements)}"
                              f"\n\nI think I should{action}"
                              "\nIs that right? (y) yes, (n) no")

            self._action = " click"
            self._step = DialogueState.Command
        elif self._step == DialogueState.ActionFeedback:
            if response == "y":
                pass
            elif response == "n":
                if "click" in self._action:
                    self._action = " type"
                elif "type" in self._action:
                    self._action = " click"
            else:
                return Prompt("Please respond with 'y' or 'n'")

            self._step = DialogueState.Command

        if "click" in self._action:
            pruned_elements = list(filter(lambda x: "link" in x or "button" in x, page_elements))
        elif "type" in self._action:
            pruned_elements = list(filter(lambda x: "input" in x, page_elements))

        state = self._construct_state(url, pruned_elements)
        examples = self.gather_examples(state)
        prompt = self._construct_prompt(state, examples)

        if self._step == DialogueState.Command:
            if len(pruned_elements) == 1:
                chosen_element = " " + " ".join(pruned_elements[0].split(" ")[:2])
            else:
                state = self._construct_state(url, ["$elements"])
                prompt = self._construct_prompt(state, examples)

                state, prompt = self._shorten_prompt(url, ["$elements"], examples, self._action)

                group_size = 20
                chosen_element = self.choose_element(
                    prompt + self._action + "{id}",
                    list(map(lambda x: {
                        "id": " " + " ".join(x.split(" ")[:2]),
                        "elements": x
                    }, pruned_elements)),
                    group_size,
                )["id"]

                state = self._construct_state(url, pruned_elements)
                prompt = self._construct_prompt(state, examples)

                state, prompt = self._shorten_prompt(url, pruned_elements, examples, self._action, chosen_element)

            text = None
            while text is None:
                try:
                    print(len(self.co.tokenize(prompt + self._action + chosen_element)))
                    text = max(self.co.generate(prompt=prompt + self._action + chosen_element,
                                                model="xlarge",
                                                temperature=0.5,
                                                num_generations=1,
                                                max_tokens=20,
                                                stop_sequences=["\n"],
                                                return_likelihoods="GENERATION").generations,
                               key=lambda x: x.likelihood).text
                    print(text)
                except cohere.error.CohereError:
                    print("Cohere fucked up")
                    continue
            cmd = (self._action + chosen_element + text).strip()

            self._cmd = cmd
            self._step = DialogueState.CommandFeedback
            return Prompt(f"Suggested command: {cmd}.\n\t(y) accept and continue"
                          "\n\t(s) save example, accept, and continue"
                          "\n\t(enter a new command) type your own command to replace the model's suggestion"
                          "\n\t(cancel) terminate the session"
                          "\nType a choice and then press enter:")
        elif self._step == DialogueState.CommandFeedback:
            cmd = self._cmd

            if response != "y" and response != "s":
                cmd = response

            cmd_pattern = r"(click|type) (link|button|input) [\d]+( \"\w+\")?"
            if not re.match(cmd_pattern, cmd):
                return Prompt(f"Invalid command '{cmd}'. Must match regex '{cmd_pattern}'. Try again...")

            if response == "s" or response != "y":
                self._save_example(state=self._construct_state(url, pruned_elements), command=cmd)

        self.previous_commands.append(cmd)

        # reset state and return the decided upon command
        self.reset_state()
        return Command(cmd.strip())
