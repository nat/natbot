#!/usr/bin/env python
#
# natbot.py
#
# Set COHERE_KEY to your API key from os.cohere.ai, and then run this from a terminal.
#

import os
import time
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

    def run_cmd(cmd):
        print("cmd", cmd)
        cmd = cmd.split("\n")[0]

        if cmd.startswith("SCROLL UP"):
            _crawler.scroll("up")
        elif cmd.startswith("SCROLL DOWN"):
            _crawler.scroll("down")
        elif cmd.startswith("click"):
            commasplit = cmd.split(",")
            id = commasplit[0].split(" ")[2]
            _crawler.click(id)
        elif cmd.startswith("type"):
            spacesplit = cmd.split(" ")
            id = spacesplit[2]
            text = spacesplit[3:]
            text = " ".join(text)
            # Strip leading and trailing double quotes
            text = text[1:-1]
            text += '\n'
            _crawler.type(id, text)

        time.sleep(2)

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
        cmd = _controller.step(objective, _crawler.page.url, content).strip()

        if len(cmd) > 0:
            print("Suggested command: " + cmd)

        command = input()
        if command == "r" or command == "":
            run_cmd(cmd)
        elif command == "g":
            url = input("URL:")
            _crawler.go_to_page(url)
        elif command == "u":
            _crawler.scroll("up")
            time.sleep(1)
        elif command == "d":
            _crawler.scroll("down")
            time.sleep(1)
        elif command == "c":
            id = input("id:")
            _crawler.click(id)
            time.sleep(1)
        elif command == "t":
            id = input("id:")
            text = input("text:")
            _crawler.type(id, text)
            time.sleep(1)
        elif command == "o":
            objective = input("Objective:")
        else:
            run_cmd(command)
            cmd = command
