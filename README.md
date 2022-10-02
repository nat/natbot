# WebLM

Drive a browser with Cohere

Derived from Nat's repo here: https://github.com/nat/natbot

Features:
- uses likelihoods instead of raw generations to guide the model's decision-making
- ability to build a reference set of examples that the LM can make reference to
- ability to fix the LM's mistakes online and add them to the reference set for future use
