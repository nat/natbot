#!/usr/bin/env python
#
# natbot.py
#
# Set COHERE_KEY to your API key from os.cohere.ai, and then run this from a terminal.
#

import os
from multiprocessing import Pool

import cohere

from .controller import Controller
from .crawler import Crawler

co = cohere.Client(os.environ.get("COHERE_KEY"))

if (__name__ == "__main__"):
    _crawler = Crawler()

    def print_help():
        print("(g) to visit url\n(u) scroll up\n(d) scroll dow\n(c) to click\n(t) to type\n" +
              "(h) to view commands again\n(r) to run suggested command\n(o) change objective")

    objective = "Make a reservation for 2 at 7pm at bistro vida in menlo park"
    print("\nWelcome to natbot! What is your objective?")
    i = input()
    if len(i) > 0:
        objective = i

    _controller = Controller(co, objective)

    cmd = None
    _crawler.go_to_page("google.com")
    while True:
        content = _crawler.crawl()
        cmd = _controller.cli_step(_crawler.page.url, content).strip()

        if len(cmd) > 0:
            print("Suggested command: " + cmd)

        _crawler.run_cmd(cmd)
