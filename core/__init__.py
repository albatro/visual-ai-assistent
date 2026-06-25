"""Shared upstream pipeline for all rendering branches (MuseTalk, RPM 3D, Live2D/Rive).

Each branch consumes:
  * core.llm        — LLM streaming and sentence splitting
  * core.tts        — Edge / Silero TTS, both producing a 16k mono WAV
  * core.audio_chunks — WAV → fixed-duration chunk files for incremental rendering
  * core.visemes    — PCM window → RPM-style viseme weights (rig-agnostic)

Branches differ only in how they turn the WAV + visemes into something a client renders.
"""
