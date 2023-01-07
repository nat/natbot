"""Given an objective, the current webpage, and the list of previous actions. Choose an action to take next."""

from typing import List
import cohere

from weblm.basic_controller.utils import (MAX_SEQ_LEN, TYPEABLE, CLICKABLE, DialogueState, Prompt, choose,
                                          construct_prompt, construct_state, gather_examples, shorten_prompt,
                                          user_prompt_1)


def pick_action(co: cohere.Client,
                step: str,
                action: str,
                objective: str,
                url: str,
                page_elements: List[str],
                previous_commands: List[str],
                response: str = None):
    # this strategy for action selection does not work very well, TODO improve this

    state = construct_state(objective, url, page_elements, previous_commands)
    examples = gather_examples(co, state)
    prompt = construct_prompt(state, examples)

    if step == DialogueState.Action:
        action = " click"
        if any(y in x for y in TYPEABLE for x in page_elements):
            elements = list(filter(lambda x: any(x.startswith(y) for y in CLICKABLE + TYPEABLE), page_elements))

            state, prompt = shorten_prompt(co,
                                           objective,
                                           url,
                                           elements,
                                           previous_commands,
                                           examples,
                                           target=MAX_SEQ_LEN)

            action = choose(co, prompt + "{action}", [
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
                step = DialogueState.ActionFeedback
                return step, action, Prompt(eval(f'f"""{user_prompt_1}"""'))

        step = DialogueState.Command
    elif step == DialogueState.ActionFeedback:
        if response == "y":
            pass
        elif response == "n":
            if "click" in action:
                action = " type"
            elif "type" in action:
                action = " click"
        elif response == "examples":
            examples = "\n".join(examples)
            return step, action, Prompt(f"Examples:\n{examples}\n\n"
                                        "Please respond with 'y' or 'n'")
        else:
            return step, action, Prompt("Please respond with 'y' or 'n'")

        step = DialogueState.Command

    return step, action, None
