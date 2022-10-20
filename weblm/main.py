#!/usr/bin/env python
#
# natbot.py
#
# Set COHERE_KEY to your API key from os.cohere.ai, and then run this from a terminal.
#

import os
import re
import time
from multiprocessing import Pool

import cohere

from .controller import Command, Controller, Prompt
from .crawler import URL_PATTERN, Crawler

co = cohere.Client(os.environ.get("COHERE_KEY"), check_api_key=False)

if (__name__ == "__main__"):

    def reset():
        _crawler = Crawler()

        def print_help():
            print("(g) to visit url\n(u) scroll up\n(d) scroll dow\n(c) to click\n(t) to type\n" +
                  "(h) to view commands again\n(r) to run suggested command\n(o) change objective")

        objective = "Make a reservation for 2 at 7pm at bistro vida in menlo park"
        print("\nWelcome to WebLM! What is your objective?")
        i = input()
        if len(i) > 0:
            objective = i

        _controller = Controller(co, objective)
        return _crawler, _controller

    crawler, controller = reset()

    response = None
    crawler.go_to_page("google.com")
    while True:
        if response == "cancel":
            crawler, controller = reset()
        elif response == "success":
            controller.success()
            crawler, controller = reset()
        elif response is not None and re.match(
                f"goto {URL_PATTERN}",
                response,
        ):
            url = re.match(URL_PATTERN, response[5:]).group(0)
            response = None
            crawler.go_to_page(url)
            time.sleep(2)

        content = crawler.crawl()
        print(content)
        response = controller.step(crawler.page.url, content, response)

        print(response)

        if isinstance(response, Command):
            crawler.run_cmd(str(response))
            response = None
        elif isinstance(response, Prompt):
            response = input(str(response))
