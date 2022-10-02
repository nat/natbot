import json
import os
from multiprocessing import Pool
from typing import List

import cohere
import numpy as np

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
    return (self.co.generate(prompt=prompt + option, max_tokens=0, model="xlarge",
                             return_likelihoods="ALL").generations[0].likelihood, option)


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

    def choose(self, options: List[str], prompt: str) -> str:
        """Choose the most like continuation of `prompt` from a set of `options`.

        Args:
            options (List[str]): the options to be chosen from
            prompt (str): the prompt

        Returns:
            str: the most likely option from `options`
        """
        num_options = len(options)
        with Pool(min(num_options, 16)) as pp:
            _lh = pp.map(_fn, zip(options, [prompt] * num_options, [self] * num_options))
        return max(_lh, key=lambda x: x[0])[1]

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

    def _construct_state(self, objective: str, url: str, page_elements: List[str]) -> str:
        state = state_template
        state = state.replace("$objective", objective)
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

    def step(self, objective: str, url: str, page_elements: List[str]):
        state = self._construct_state(objective, url, page_elements)
        examples = self.gather_examples(state)
        prompt = self._construct_prompt(state, examples)

        if any("input" in x for x in page_elements):
            action = self.choose([
                " click",
                " type",
            ], prompt)
        else:
            action = " click"

        if "click" in action:
            pruned_elements = list(filter(lambda x: "link" in x or "button" in x, page_elements))
        elif "type" in action:
            pruned_elements = list(filter(lambda x: "input" in x, page_elements))

        state = self._construct_state(objective, url, pruned_elements)
        examples = self.gather_examples(state)
        prompt = self._construct_prompt(state, examples)

        print(state)

        chosen_element = self.choose(list(map(lambda x: " " + " ".join(x.split(" ")[:2]), pruned_elements)),
                                     prompt=prompt + action)

        text = max(self.co.generate(prompt=prompt + action + chosen_element,
                                    model="xlarge",
                                    temperature=0.5,
                                    num_generations=5,
                                    max_tokens=20,
                                    stop_sequences=["\n"],
                                    return_likelihoods="GENERATION").generations,
                   key=lambda x: x.likelihood).text
        cmd = (action + chosen_element + text).strip()
        response = input(f"Suggested command: {cmd}.\n\t(enter) accept and continue"
                         "\n\t(s) save example, accept, and continue"
                         "\n\t(enter a new command) type your own command to replace the model's suggestion"
                         "\nType a choice and then press enter:")
        if response != "" and response != "s":
            cmd = response

        if response == "s" or response != "":
            self._save_example(state=self._construct_state(objective, url, pruned_elements), command=cmd)

        self.previous_commands.append(cmd)

        return cmd
