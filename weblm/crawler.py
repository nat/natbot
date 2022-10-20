import asyncio
import json
import re
import time
from os.path import exists
from sys import platform

from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright

black_listed_elements = set([
    "html",
    "head",
    "title",
    "meta",
    "iframe",
    "body",
    "script",
    "style",
    "path",
    "svg",
    "br",
    "::marker",
])

URL_PATTERN = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
WINDOW_SIZE = {"width": 1280, "height": 1080}


class Crawler:

    def __init__(self):
        self.browser = sync_playwright().start().chromium.launch(headless=False,)
        self.context = self.browser.new_context(
            user_agent=
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"
        )

        self.page = self.context.new_page()
        self.page.set_viewport_size(WINDOW_SIZE)

    def go_to_page(self, url):
        self.page.goto(url=url if "://" in url else "http://" + url)
        self.client = self.page.context.new_cdp_session(self.page)
        self.page_element_buffer = {}

    def scroll(self, direction):
        if direction == "up":
            self.page.evaluate(
                "(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop - window.innerHeight;"
            )
        elif direction == "down":
            self.page.evaluate(
                "(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop + window.innerHeight;"
            )

    def click(self, id):
        # Inject javascript into the page which removes the target= attribute from all links
        js = """
        links = document.getElementsByTagName("a");
        for (var i = 0; i < links.length; i++) {
            links[i].removeAttribute("target");
        }
        """
        self.page.evaluate(js)

        element = self.page_element_buffer.get(int(id))
        if element:
            x = element.get("center_x")
            y = element.get("center_y")

            height, width = WINDOW_SIZE["height"], WINDOW_SIZE["width"]

            x_d = max(0, x - width)
            x_d += 5 * int(x_d > 0)
            y_d = max(0, y - height)
            y_d += 5 * int(y_d > 0)

            self.page.evaluate(f"() => window.scrollTo({x_d}, {y_d})")

            self.page.mouse.click(x - x_d, y - y_d)
        else:
            print("Could not find element")

    def type(self, id, text):
        self.click(id)
        self.page.keyboard.type(text)

    def enter(self):
        self.page.keyboard.press("Enter")

    def crawl(self):
        start = time.time()

        page = self.page
        tree = self.client.send(
            "DOMSnapshot.captureSnapshot",
            {
                "computedStyles": ["display"],
                "includeDOMRects": True,
                "includePaintOrder": True
            },
        )
        device_pixel_ratio = page.evaluate("window.devicePixelRatio")
        win_scroll_x = page.evaluate("window.scrollX")
        win_scroll_y = page.evaluate("window.scrollY")
        win_upper_bound = page.evaluate("window.pageYOffset")
        win_left_bound = page.evaluate("window.pageXOffset")
        win_width = page.evaluate("window.screen.width")
        win_height = page.evaluate("window.screen.height")
        elements_of_interest = self._crawl(tree, win_upper_bound, win_width, win_left_bound, win_height,
                                           device_pixel_ratio)

        print("Parsing time: {:0.2f} seconds".format(time.time() - start))
        return elements_of_interest

    def _crawl(self, tree, win_upper_bound, win_width, win_left_bound, win_height, device_pixel_ratio):
        page_element_buffer = self.page_element_buffer

        page_state_as_text = []

        if platform == "darwin" and device_pixel_ratio == 1:  # lies
            device_pixel_ratio = 2

        win_right_bound = win_left_bound + win_width * 2
        win_lower_bound = win_upper_bound + win_height * 2

        percentage_progress_start = 1
        percentage_progress_end = 2

        page_state_as_text.append({
            "x":
                0,
            "y":
                0,
            "text":
                "[scrollbar {:0.2f}-{:0.2f}%]".format(round(percentage_progress_start, 2),
                                                      round(percentage_progress_end)),
        })

        strings = tree["strings"]
        document = tree["documents"][0]
        nodes = document["nodes"]
        backend_node_id = nodes["backendNodeId"]
        attributes = nodes["attributes"]
        node_value = nodes["nodeValue"]
        parent = nodes["parentIndex"]
        node_types = nodes["nodeType"]
        node_names = nodes["nodeName"]
        is_clickable = set(nodes["isClickable"]["index"])

        text_value = nodes["textValue"]
        text_value_index = text_value["index"]
        text_value_values = text_value["value"]

        input_value = nodes["inputValue"]
        input_value_index = input_value["index"]
        input_value_values = input_value["value"]

        input_checked = nodes["inputChecked"]
        layout = document["layout"]
        layout_node_index = layout["nodeIndex"]
        bounds = layout["bounds"]
        styles = layout["styles"]

        cursor = 0
        html_elements_text = []

        child_nodes = {}
        elements_in_view_port = []

        def convert_name(node_name, has_click_handler):
            if node_name == "a":
                return "link"
            elif node_name in ["select", "img", "input"]:
                return node_name
            elif (node_name in "button" or has_click_handler):  # found pages that needed this quirk
                return "button"
            else:
                return "text"

        def find_attributes(attributes, keys):
            values = {}

            for [key_index, value_index] in zip(*(iter(attributes),) * 2):
                if value_index < 0:
                    continue
                key = strings[key_index]
                value = strings[value_index]

                if key in keys:
                    values[key] = value
                    keys.remove(key)

                    if not keys:
                        return values

            return values

        def add_to_hash_tree(hash_tree, tag, node_id, node_name, parent_id):
            parent_id_str = str(parent_id)
            if not parent_id_str in hash_tree:
                parent_name = strings[node_names[parent_id]].lower()
                grand_parent_id = parent[parent_id]

                add_to_hash_tree(hash_tree, tag, parent_id, parent_name, grand_parent_id)

            is_parent_desc_anchor, anchor_id = hash_tree[parent_id_str]

            element_attributes = find_attributes(attributes[node_id], ["role"])

            # even if the anchor is nested in another anchor, we set the "root" for all descendants to be ::Self
            if node_name in tag or element_attributes.get("role") in tag:
                value = (True, node_id)
            elif (is_parent_desc_anchor):  # reuse the parent's anchor_id (which could be much higher in the tree)
                value = (True, anchor_id)
            else:
                value = (
                    False,
                    None,
                )  # not a descendant of an anchor, most likely it will become text, an interactive element or discarded

            hash_tree[str(node_id)] = value

            return value

        anchor_ancestry = {"-1": (False, None)}
        button_ancestry = {"-1": (False, None)}
        select_ancestry = {"-1": (False, None)}

        for index, node_name_index in enumerate(node_names):
            node_parent = parent[index]
            node_name = strings[node_name_index].lower()

            is_ancestor_of_anchor, anchor_id = add_to_hash_tree(anchor_ancestry, ["a"], index, node_name, node_parent)

            is_ancestor_of_button, button_id = add_to_hash_tree(button_ancestry, ["button"], index, node_name,
                                                                node_parent)

            is_ancestor_of_select, select_id = add_to_hash_tree(select_ancestry, ["select"], index, node_name,
                                                                node_parent)

            try:
                cursor = layout_node_index.index(select_id) if is_ancestor_of_select else layout_node_index.index(index)
            except:
                continue

            if node_name in black_listed_elements:
                continue

            style = map(lambda x: strings[x], styles[cursor])
            if "none" in style:
                continue

            [x, y, width, height] = bounds[cursor]
            x /= device_pixel_ratio
            y /= device_pixel_ratio
            width /= device_pixel_ratio
            height /= device_pixel_ratio

            elem_left_bound = x
            elem_top_bound = y
            elem_right_bound = x + width
            elem_lower_bound = y + height

            # comment this bit out to process the whole thing
            partially_is_in_viewport = (elem_left_bound < win_right_bound and elem_right_bound >= win_left_bound and
                                        elem_top_bound < win_lower_bound and elem_lower_bound >= win_upper_bound)

            if not partially_is_in_viewport:
                continue

            meta_data = []

            # inefficient to grab the same set of keys for kinds of objects but its fine for now
            element_attributes = find_attributes(
                attributes[index], ["type", "placeholder", "aria-label", "name", "title", "alt", "role", "value"])

            ancestor_exception = is_ancestor_of_anchor or is_ancestor_of_button or is_ancestor_of_select
            ancestor_node_key = None
            if ancestor_exception:
                if is_ancestor_of_anchor:
                    ancestor_node_key = str(anchor_id)
                elif is_ancestor_of_button:
                    ancestor_node_key = str(button_id)
                elif is_ancestor_of_select:
                    ancestor_node_key = str(select_id)
            ancestor_node = (None if not ancestor_exception else child_nodes.setdefault(str(ancestor_node_key), []))

            if node_name == "#text" and ancestor_exception:
                text = strings[node_value[index]]
                if text == "|" or text == "•":
                    continue
                ancestor_node.append({"type": "type", "value": text})
            else:
                if (node_name == "input" and element_attributes.get("type")
                        == "submit") or node_name == "button" or element_attributes.get("role") == "button":
                    node_name = "button"
                    element_attributes.pop("type", None)  # prevent [button ... (button)..]
                    element_attributes.pop("role", None)  # prevent [button ... (button)..]

                if element_attributes.get("role") == "textbox":
                    node_name = "input"

                for key in element_attributes:
                    if ancestor_exception and not is_ancestor_of_select:
                        ancestor_node.append({"type": "attribute", "key": key, "value": element_attributes[key]})
                    else:
                        meta_data.append(element_attributes[key])

            element_node_value = None

            if node_value[index] >= 0:
                element_node_value = strings[node_value[index]]
                if element_node_value == "|":  #commonly used as a seperator, does not add much context - lets save ourselves some token space
                    continue
            elif (node_name == "input" and index in input_value_index and element_node_value is None):
                node_input_text_index = input_value_index.index(index)
                text_index = input_value_values[node_input_text_index]
                if node_input_text_index >= 0 and text_index >= 0:
                    element_node_value = strings[text_index]

            # remove redudant elements
            if ancestor_exception and (node_name not in ["a", "button", "select"]):
                continue

            elements_in_view_port.append({
                "node_index": str(index),
                "backend_node_id": backend_node_id[index],
                "node_name": node_name,
                "node_value": element_node_value,
                "node_meta": meta_data,
                "is_clickable": index in is_clickable,
                "origin_x": int(x),
                "origin_y": int(y),
                "center_x": int(x + (width / 2)),
                "center_y": int(y + (height / 2)),
            })

        # lets filter further to remove anything that does not hold any text nor has click handlers + merge text from leaf#text nodes with the parent
        elements_of_interest = []
        id_counter = 0

        for element in elements_in_view_port:
            node_index = element.get("node_index")
            node_name = element.get("node_name")
            node_value = element.get("node_value")
            is_clickable = element.get("is_clickable")
            origin_x = element.get("origin_x")
            origin_y = element.get("origin_y")
            center_x = element.get("center_x")
            center_y = element.get("center_y")
            meta_data = element.get("node_meta")

            inner_text = f"{node_value} " if node_value else ""
            meta = ""

            if node_index in child_nodes:
                for child in child_nodes.get(node_index):
                    entry_type = child.get('type')
                    entry_value = child.get('value')

                    if entry_type == "attribute":
                        entry_key = child.get('key')
                        meta_data.append(f'{entry_key}="{entry_value}"')
                    else:
                        inner_text += f"{entry_value} "

            if meta_data:
                meta_string = " ".join(meta_data)
                meta = f" {meta_string}"

            if inner_text != "":
                inner_text = f"{inner_text.strip()}"

            converted_node_name = convert_name(node_name, is_clickable)

            # not very elegant, more like a placeholder
            if converted_node_name not in ["button", "link", "input", "img", "textarea", "select"
                                          ] and inner_text.strip() == "":
                continue
            elif converted_node_name == "button" and meta == "" and inner_text.strip() == "":
                continue

            page_element_buffer[id_counter] = element

            meta = re.sub('\s+', ' ', meta)
            inner_text = re.sub('\s+', ' ', inner_text)

            if inner_text != "":
                elements_of_interest.append(f"""{converted_node_name} {id_counter}{meta} \"{inner_text}\"""")
            elif converted_node_name in ["input", "button"] or "alt" in meta:
                elements_of_interest.append(f"""{converted_node_name} {id_counter}{meta}""")
            elif converted_node_name == "select" and meta != "":
                elements_of_interest.append(f"""{converted_node_name} {id_counter}{meta}""")
            else:
                # print(f"""{converted_node_name} {id_counter}{meta}""")
                pass
            id_counter += 1

        return elements_of_interest

    def run_cmd(self, cmd):
        print("cmd", cmd)
        cmd = replace_special_fields(cmd.strip())

        if cmd.startswith("SCROLL UP"):
            self.scroll("up")
        elif cmd.startswith("SCROLL DOWN"):
            self.scroll("down")
        elif cmd.startswith("click"):
            commasplit = cmd.split(",")
            id = commasplit[0].split(" ")[2]
            self.click(id)
        elif cmd.startswith("type"):
            spacesplit = cmd.split(" ")
            id = spacesplit[2]
            text = spacesplit[3:]
            text = " ".join(text)
            # Strip leading and trailing double quotes
            text = text[1:-1]
            text += '\n'
            self.type(id, text)

        time.sleep(2)


class AsyncCrawler(Crawler):

    def __init__(self, playwright) -> None:
        self.playwright = playwright

    async def _init_browser(self):
        self.browser = await self.playwright.chromium.launch(headless=True,)
        self.context = await self.browser.new_context(
            user_agent=
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"
        )

        self.page = await self.context.new_page()
        await self.page.set_viewport_size({"width": 1280, "height": 1080})

    async def screenshot(self):
        _path = "screenshot.png"
        await self.page.screenshot(path=_path)
        return _path

    async def crawl(self):
        start = time.time()

        page = self.page
        tree = await self.client.send(
            "DOMSnapshot.captureSnapshot",
            {
                "computedStyles": ["display"],
                "includeDOMRects": True,
                "includePaintOrder": True
            },
        )
        device_pixel_ratio = await page.evaluate("window.devicePixelRatio")
        win_scroll_x = await page.evaluate("window.scrollX")
        win_scroll_y = await page.evaluate("window.scrollY")
        win_upper_bound = await page.evaluate("window.pageYOffset")
        win_left_bound = await page.evaluate("window.pageXOffset")
        win_width = await page.evaluate("window.screen.width")
        win_height = await page.evaluate("window.screen.height")
        elements_of_interest = self._crawl(tree, win_upper_bound, win_width, win_left_bound, win_height,
                                           device_pixel_ratio)

        print("Parsing time: {:0.2f} seconds".format(time.time() - start))
        return elements_of_interest

    async def go_to_page(self, url):
        await self.page.goto(url=url if "://" in url else "http://" + url)
        self.client = await self.page.context.new_cdp_session(self.page)
        self.page_element_buffer = {}

    async def scroll(self, direction):
        if direction == "up":
            await self.page.evaluate(
                "(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop - window.innerHeight;"
            )
        elif direction == "down":
            await self.page.evaluate(
                "(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop + window.innerHeight;"
            )

    async def click(self, id):
        # Inject javascript into the page which removes the target= attribute from all links
        js = """
        links = document.getElementsByTagName("a");
        for (var i = 0; i < links.length; i++) {
            links[i].removeAttribute("target");
        }
        """
        await self.page.evaluate(js)

        element = self.page_element_buffer.get(int(id))
        if element:
            x = element.get("center_x")
            y = element.get("center_y")

            await self.page.mouse.click(x, y)
        else:
            print("Could not find element")

    async def type(self, id, text):
        await self.click(id)
        await self.page.keyboard.type(text)

    async def enter(self):
        await self.page.keyboard.press("Enter")

    async def run_cmd(self, cmd):
        print("cmd", cmd)
        cmd = replace_special_fields(cmd.strip())

        if cmd.startswith("SCROLL UP"):
            await self.scroll("up")
        elif cmd.startswith("SCROLL DOWN"):
            await self.scroll("down")
        elif cmd.startswith("click"):
            commasplit = cmd.split(",")
            id = commasplit[0].split(" ")[2]
            await self.click(id)
        elif cmd.startswith("type"):
            spacesplit = cmd.split(" ")
            id = spacesplit[2]
            text = spacesplit[3:]
            text = " ".join(text)
            # Strip leading and trailing double quotes
            text = text[1:-1]
            text += '\n'
            await self.type(id, text)
        else:
            raise Exception(f"Invalid command: {cmd}")

        time.sleep(2)


def replace_special_fields(cmd):
    if exists("specials.json"):
        with open("specials.json", "r") as fd:
            specials = json.load(fd)

        for k, v in specials.items():
            cmd = cmd.replace(k, v)

    return cmd
