import asyncio
import chunk
import os
import traceback
from typing import Dict, Tuple

import cohere
import discord
from discord import Embed, File
from discord.ext import commands
from playwright.async_api import async_playwright

from .controller import Command, Controller, Prompt
from .crawler import AsyncCrawler

co = cohere.Client(os.environ.get("COHERE_KEY"))

MSG_LEN_LIMIT = 1800


def chunk_message_for_sending(msg):
    chunks = []
    tmp_chunk = ""
    for line in msg.split("\n"):
        if len(tmp_chunk + line) > MSG_LEN_LIMIT:
            chunks.append(tmp_chunk)
            tmp_chunk = line
        else:
            tmp_chunk += "\n" + line

    if tmp_chunk != "":
        chunks.append(tmp_chunk)

    return chunks


class MyClient(discord.Client):

    def __init__(self, playwright, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sessions: Dict[int, Tuple[Crawler, Controller]] = {}
        self.playwright = playwright

    async def on_ready(self):
        """Initializes bot"""
        print(f"We have logged in as {self.user}")

        for guild in self.guilds:
            print(f"{self.user} is connected to the following guild:\n"
                  f"{guild.name}(id: {guild.id})")

    async def find_session(self, id, message):
        print(message.clean_content)
        objective = message.clean_content.removeprefix("weblm ")

        if id not in self.sessions:
            print("did not find session")
            crawler, controller = (AsyncCrawler(self.playwright), Controller(co, objective))
            await crawler._init_browser()
            print("browser inited")
            self.sessions[id] = (crawler, controller)
            await crawler.go_to_page("google.com")
            print("crawler on page")

        crawler, controller = self.sessions[id]

        return (crawler, controller)

    async def respond_to_message(self, message):
        print(message.clean_content)
        objective = message.clean_content.removeprefix("weblm ")
        crawler, controller = await self.find_session(message.id, message)

        if objective == "cancel":
            del self.sessions[message.id]
            return
        elif objective == "success":
            controller.success()
            del self.sessions[message.channel.starter_message.id]
            msg = await message.channel.send("🎉🎉🎉")
            await msg.edit(suppress=True)
            return

        while True:
            content = await crawler.crawl()

            async with message.channel.typing():
                if not controller.is_running():
                    print("Controller not yet running")
                    response = controller.step(crawler.page.url, content)
                else:
                    response = controller.step(crawler.page.url, content, response=objective)

                print(response)

                if isinstance(response, Command):
                    print("running command", response)
                    await crawler.run_cmd(str(response))
                elif isinstance(response, Prompt):
                    thread = await message.create_thread(name=objective)
                    for chunk in chunk_message_for_sending(str(response)):
                        msg = await thread.send(chunk)
                        await msg.edit(suppress=True)
                    return

    async def respond_in_thread(self, message):
        if message.channel.starter_message.id not in self.sessions:
            print("Session not running, killing")
            await message.channel.send("This session is dead please begin a new one in #web-lm.")
            return

        print(message.clean_content)
        objective = message.clean_content.removeprefix("weblm ")
        crawler, controller = await self.find_session(message.channel.starter_message.id, message)

        if objective == "cancel":
            del self.sessions[message.channel.starter_message.id]
            return
        elif objective == "success":
            controller.success()
            del self.sessions[message.channel.starter_message.id]
            msg = await message.channel.send("🎉🎉🎉")
            await msg.edit(suppress=True)
            return
        elif objective == "show":
            path = await crawler.screenshot()
            await message.channel.send(file=discord.File(path))
            return

        while True:
            content = await crawler.crawl()
            print("AIDAN", content)

            async with message.channel.typing():
                if not controller.is_running():
                    print("Controller not yet running")
                    response = controller.step(crawler.page.url, content)
                else:
                    response = controller.step(crawler.page.url, content, response=objective)

                print(response)

                if isinstance(response, Command):
                    print("running command", response)
                    await crawler.run_cmd(str(response))
                elif isinstance(response, Prompt):
                    for chunk in chunk_message_for_sending(str(response)):
                        msg = await message.channel.send(chunk)
                        await msg.edit(suppress=True)
                    return

    async def respond_to_dm(self, message):
        print(message.clean_content)
        objective = message.clean_content.removeprefix("weblm ")
        crawler, controller = await self.find_session(message.author.id, message)

        if objective == "cancel":
            del self.sessions[message.author.id]
            return
        elif objective == "success":
            controller.success()
            del self.sessions[message.author.id]
            msg = await message.channel.send("🎉🎉🎉")
            await msg.edit(suppress=True)
            return
        elif objective == "show":
            path = await crawler.screenshot()
            await message.channel.send(file=discord.File(path))
            return

        while True:
            content = await crawler.crawl()
            print("AIDAN", content)

            async with message.channel.typing():
                if not controller.is_running():
                    print("Controller not yet running")
                    response = controller.step(crawler.page.url, content)
                else:
                    response = controller.step(crawler.page.url, content, response=objective)

                print(response)

                if isinstance(response, Command):
                    print("running command", response)
                    await crawler.run_cmd(str(response))
                elif isinstance(response, Prompt):
                    for chunk in chunk_message_for_sending(str(response)):
                        msg = await message.channel.send(chunk)
                        await msg.edit(suppress=True)
                    return

    async def on_message(self, message):
        try:
            print(message)
            if isinstance(message.channel, discord.DMChannel) and message.author != self.user:
                await self.respond_to_dm(message)
            elif isinstance(message.channel, discord.TextChannel) and message.channel.id in [
                    1026557845308723212, 1032611829186306048
            ] and message.author != self.user and message.clean_content.startswith("weblm "):
                await self.respond_to_message(message)
            elif isinstance(message.channel, discord.Thread) and message.channel.parent.id in [
                    1026557845308723212, 1032611829186306048
            ] and message.author != self.user:
                await self.respond_in_thread(message)
        except Exception:
            print(f"Exception caught:\n{traceback.format_exc()}")


async def main():
    intents = discord.Intents.all()
    async with async_playwright() as playwright:
        async with MyClient(playwright, intents=intents) as client:
            try:
                await client.start(os.environ.get("DISCORD_KEY"))
            except Exception as e:
                print(f"Exception caught: {e}")


if __name__ == "__main__":
    asyncio.run(main())
