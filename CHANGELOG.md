Commit: 50226ea
Branch: fix/kokoro-audio-silence
Files: Autopilot/modules/audio_processor.py, Autopilot/modules/tts_engine_manager.py
Fixes: Removed destructive agate filter; fixed PCM normalization 32768->32767; preserved original channel count.
Date: 2026-07-23
