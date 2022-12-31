# Setup

Follow commands below. If needed, install [Playwright for Python](https://playwright.dev/python/docs/intro).

```
# Optional: create and activate Python virtualenv
python3 -m venv .venv --prompt=BrowserGPT
source .venv/bin/activate

# Install requirements
pip3 install requirements.txt

# Install required browsers
python3 -m playwright install
```

# Usage

```
python3 natbot.py
```

# natbot

Drive a browser with GPT-3

Here's a demo: https://twitter.com/natfriedman/status/1575631194032549888

Lots of ideas for improvement:
- Better prompt
- Prompt chaining
- Make a recorder to collect human feedback and do better few-shot
- Better DOM serialization
- Let the agent use multiple tabs and switch between them

Improvements welcome!
