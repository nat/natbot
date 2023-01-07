import csv
import json
import os
import re
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple, Union

import cohere
from weblm.basic_controller.pick_command import generate_command
from weblm.basic_controller.prioritize import generate_prioritization
from weblm.basic_controller.pick_action import pick_action
from weblm.basic_controller.utils import (CLICKABLE, MAX_NUM_ELEMENTS, TYPEABLE, Command, DialogueState, Prompt,
                                          construct_state, search, shorten_prompt, user_prompt_end)

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

    def _save_example(self, objective: str, url: str, elements: List[str], command: str, previous_commands: List[str]):
        state = construct_state(objective, url, elements[:MAX_NUM_ELEMENTS], previous_commands)
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
                self._prioritized_elements = generate_prioritization(self.co, self.objective, page_elements, url,
                                                                     self.previous_commands)
                self._prioritized_elements_hash = hash(frozenset(page_elements))
                self._pruned_prioritized_elements = self._prioritized_elements[:MAX_NUM_ELEMENTS]
                self._step = DialogueState.Action

            if re.match(r'search (.+)', response or ""):
                query = re.match(r'search (.+)', response).group(1)
                results = search(self.co, query, self._page_elements, topk=50)
                return Prompt(f"Query: {query}\nResults:\n{results}\n\n"
                              "Please respond with 'y' or 'n'")

            self.user_responses[response] += 1
            self._construct_responses()

            if self._step in [DialogueState.Action, DialogueState.ActionFeedback]:
                self._step, self._action, prompt = pick_action(self.co, self._step, self._action, self.objective, url,
                                                               self._pruned_prioritized_elements,
                                                               self.previous_commands, response)

                if prompt is not None:
                    return prompt

            if "click" in self._action:
                pruned_elements = list(
                    filter(lambda x: any(x.startswith(y) for y in CLICKABLE), self._pruned_prioritized_elements))
            elif "type" in self._action:
                pruned_elements = list(
                    filter(lambda x: any(x.startswith(y) for y in TYPEABLE), self._pruned_prioritized_elements))

            if response == "prompt":
                chosen_element = self._chosen_elements[0]["id"]
                _, prompt = shorten_prompt(self.objective, url, pruned_elements, self.previous_commands, examples,
                                           self._action, chosen_element)
                return Prompt(f"{prompt}\n\nPlease respond with 'y' or 's'")
            elif response == "elements":
                return Prompt("\n".join(str(d) for d in self._chosen_elements))

            self._step, self._cmd, self._chosen_elements, prompt = generate_command(self.co, self._step, self._action,
                                                                                    self._cmd, self._chosen_elements,
                                                                                    self.objective, url,
                                                                                    pruned_elements,
                                                                                    self.previous_commands, response)
            if self._step == DialogueState.CommandFeedback and response == "s":
                self._save_example(objective=self.objective,
                                   url=url,
                                   elements=self._prioritized_elements,
                                   command=self._cmd,
                                   previous_commands=self.previous_commands)

            if prompt is not None:
                return prompt

            self.moments.append((url, self._prioritized_elements, self._cmd, self.previous_commands.copy()))
            self.previous_commands.append(self._cmd)

            cmd = Command(self._cmd.strip())
            self.reset_state()
            return cmd

        except Exception as e:
            self._error = e
            return Prompt(f"Caught exception:\n{e}"
                          "\nPlease choose one of the following:"
                          "\n\t(c) ignore exception and continue" + user_prompt_end)
