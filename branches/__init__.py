"""Rendering branches.

Each branch is a self-contained FastAPI app exposing /ws and reusing core/*
for LLM + TTS + viseme generation. A future dispatcher_server.py will mount
them under /ws/<mode>; for now each branch can be run standalone.
"""
