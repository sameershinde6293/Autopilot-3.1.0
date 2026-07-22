"""TTS engine manager with mandatory lazy loading.

Never loads Piper/Kokoro/XTTS models at import or construction time.
Generation routes by character engine; RAM management unloads heavy engines.
"""

from __future__ import annotations

import gc
import math
import os
import random
import re
import shutil
import struct
import subprocess
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.service_container import BaseModule, ServiceContainer
from core.time_helper import utc_now_str
from modules.tts_presets import (
    ADMIN_PASSWORD,
    BREATH_CHANCE,
    BREATH_VOLUME_RANGE,
    EMOTION_ALIASES,
    EMOTION_PRESETS,
    EQ_PRESETS,
    PAUSE_BASE_DURATIONS,
    PAUSE_EMOTION_MULTIPLIERS,
    PAUSE_VARIATIONS,
    REVERB_PRESETS,
    SPECIAL_EFFECTS,
)

MODULE_NAME = "tts_engine_manager"
ENGINE_PIPER = "piper"
ENGINE_KOKORO = "kokoro"
ENGINE_XTTS = "xtts"

PAUSE_TAG_RE = re.compile(r"\[PAUSE:([A-Za-z_]+)\]", re.IGNORECASE)
WORD_RE = re.compile(r"\S+")

# RAM free thresholds (MB)
XTTS_UNLOAD_BELOW_MB = 1500
KOKORO_UNLOAD_BELOW_MB = 800
IDLE_UNLOAD_SECONDS = 60.0

# Default voice per engine, used when a requested voice is not valid for
# the engine actually resolved after fallback (a Piper voice such as
# "british_male_01" reaching Kokoro is rejected, which previously
# surfaced as synthetic beep audio).
ENGINE_DEFAULT_VOICES = {
    ENGINE_PIPER: "default",
    ENGINE_KOKORO: "af_heart",
    ENGINE_XTTS: "default",
}

# Kokoro voice ids look like "af_heart" / "bm_george"; Piper voice names
# used by this app (e.g. "british_male_01", "deep_male_us") never match.
KOKORO_VOICE_RE = re.compile(r"^[a-z][fm]_[A-Za-z]+$")


class TTSEngineManager(BaseModule):
    """Install, manage, and run TTS engines with lazy loading."""

    def __init__(self, container: ServiceContainer) -> None:
        """Initialize manager without loading any TTS engine."""
        super().__init__(container, MODULE_NAME)
        self.kokoro_instance: Any = None
        self.xtts_instance: Any = None
        self._last_use: Dict[str, float] = {}
        self._project_root = Path.cwd()
        # Resolve project root from config if available
        try:
            cfg_folder = getattr(self.config, "config_folder", None)
            if cfg_folder is not None:
                self._project_root = Path(cfg_folder).resolve().parent
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    def generate_audio(
        self,
        text: str,
        character_profile: Dict[str, Any],
        output_path: str | Path,
        voice_sample_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate speech for text using the character's configured engine.

        Args:
            text: Dialogue text (may include [PAUSE:TYPE] tags).
            character_profile: Voice profile dict.
            output_path: Destination WAV path.
            voice_sample_path: Optional XTTS reference sample.

        Returns:
            Standard response with audio_path, duration, word_timestamps.
        """
        started = time.perf_counter()
        if not self._enabled:
            return self._err("tts_engine_manager is disabled", started)
        clean_text, pause_markers = self.process_pause_tags(text or "")
        if not clean_text.strip():
            return self._err("text is empty after pause tag removal", started)

        emotion = str(
            character_profile.get("default_emotion")
            or character_profile.get("emotion")
            or "neutral"
        )
        params = self.apply_emotion_parameters(character_profile, emotion)
        preferred = str(character_profile.get("engine") or ENGINE_KOKORO).lower()
        engine = self._resolve_engine_with_fallback(preferred)
        if engine is None:
            return self._err(
                "No TTS engine available (Piper/Kokoro/XTTS not installed)",
                started,
            )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        voice_model = str(
            character_profile.get("voice_model")
            or character_profile.get("voice")
            or "default"
        )
        voice_model = self._ensure_voice_matches_engine(
            preferred, engine, voice_model
        )

        if engine == ENGINE_PIPER:
            gen = self.generate_with_piper(clean_text, voice_model, params, str(out))
        elif engine == ENGINE_KOKORO:
            gen = self.generate_with_kokoro(clean_text, voice_model, params, str(out))
        else:
            sample = voice_sample_path or character_profile.get("avatar_path")
            gen = self.generate_with_xtts(clean_text, sample, params, str(out))

        if not gen.get("success"):
            return gen

        audio_path = Path(gen["data"]["audio_path"])
        timestamps = list(gen["data"].get("word_timestamps") or [])

        # Pitch adjust if needed
        pitch = float(params.get("pitch", 0.0))
        if abs(pitch) > 0.01:
            self._apply_pitch_shift(audio_path, pitch)

        # Insert pauses
        if pause_markers:
            self.insert_pauses_into_audio(
                str(audio_path),
                pause_markers,
                timestamps,
                breathing=bool(character_profile.get("breathing_enabled")),
                character_profile=character_profile,
            )
            timestamps = self._adjust_timestamps_for_pauses(timestamps, pause_markers)

        # Voice effects chain
        self.apply_voice_effects(str(audio_path), character_profile, params)

        duration = self._wav_duration(audio_path)
        self._mark_engine_used(engine)
        self.check_ram_and_manage_engines()
        if engine == ENGINE_XTTS:
            self.unload_engine_from_memory(ENGINE_XTTS)

        return self.make_response(
            True,
            {
                "audio_path": str(audio_path),
                "duration": duration,
                "word_timestamps": timestamps,
                "engine": engine,
                "emotion": emotion,
                "params": params,
            },
            duration_ms=_ms(started),
        )

    def generate_with_piper(
        self,
        text: str,
        voice_model: str,
        settings: Dict[str, Any],
        output_path: str,
    ) -> Dict[str, Any]:
        """Generate WAV via Piper subprocess (no persistent model load)."""
        started = time.perf_counter()
        piper = self._find_piper_binary()
        if piper is None:
            return self._err(
                "Piper executable not found (STATUS: NOT VERIFIED on this host)",
                started,
            )
        model_path = self._find_piper_model(voice_model)
        if model_path is None:
            # Fallback: synthetic tone so pipeline can be tested offline
            return self._synthetic_speech_fallback(
                text, output_path, settings, engine=ENGINE_PIPER, started=started
            )
        speed = max(0.5, float(settings.get("speed", 1.0)))
        length_scale = 1.0 / speed
        cmd = [
            str(piper),
            "--model",
            str(model_path),
            "--output_file",
            output_path,
            "--length_scale",
            f"{length_scale:.4f}",
        ]
        try:
            proc = subprocess.run(
                cmd,
                input=text,
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
            if proc.returncode != 0 or not Path(output_path).exists():
                # D2a: log the meaningful tail of stderr only — piper
                # prints a full Python traceback; the old 800+ char
                # WARNING per narration line made render logs unusable.
                tail = (proc.stderr or "").strip().splitlines()
                detail = tail[-1] if tail else "no stderr"
                self.log.warning(
                    "Piper failed (rc=%s): %s", proc.returncode, detail[:300]
                )
                return self._synthetic_speech_fallback(
                    text, output_path, settings, engine=ENGINE_PIPER, started=started
                )
            timestamps = self._approximate_word_timestamps(
                text, self._wav_duration(Path(output_path))
            )
            return self.make_response(
                True,
                {
                    "audio_path": output_path,
                    "word_timestamps": timestamps,
                    "duration": self._wav_duration(Path(output_path)),
                },
                duration_ms=_ms(started),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self.log.error("Piper subprocess error: %s", exc)
            return self._synthetic_speech_fallback(
                text, output_path, settings, engine=ENGINE_PIPER, started=started
            )

    def generate_with_kokoro(
        self,
        text: str,
        voice_name: str,
        settings: Dict[str, Any],
        output_path: str,
    ) -> Dict[str, Any]:
        """Generate WAV with Kokoro (lazy-loaded)."""
        started = time.perf_counter()
        loaded = self._ensure_kokoro_loaded()
        if not loaded:
            return self._synthetic_speech_fallback(
                text, output_path, settings, engine=ENGINE_KOKORO, started=started
            )
        speed = float(settings.get("speed", 1.0))
        try:
            samples, sample_rate = self.kokoro_instance.create(
                text=text,
                voice=voice_name,
                speed=speed,
                lang="en-us",
            )
            self._write_wav_samples(output_path, samples, int(sample_rate))
            duration = self._wav_duration(Path(output_path))
            timestamps = self._approximate_word_timestamps(text, duration)
            self._mark_engine_used(ENGINE_KOKORO)
            return self.make_response(
                True,
                {
                    "audio_path": output_path,
                    "word_timestamps": timestamps,
                    "duration": duration,
                },
                duration_ms=_ms(started),
            )
        except Exception as exc:  # noqa: BLE001
            self.log.error("Kokoro generation failed: %s", exc)
            return self._synthetic_speech_fallback(
                text, output_path, settings, engine=ENGINE_KOKORO, started=started
            )

    def generate_with_xtts(
        self,
        text: str,
        voice_sample_path: Optional[str],
        settings: Dict[str, Any],
        output_path: str,
    ) -> Dict[str, Any]:
        """Generate WAV with XTTS v2 (lazy-loaded, unload after use)."""
        started = time.perf_counter()
        loaded = self._ensure_xtts_loaded()
        if not loaded:
            return self._synthetic_speech_fallback(
                text, output_path, settings, engine=ENGINE_XTTS, started=started
            )
        speed = float(settings.get("speed", 1.0))
        try:
            kwargs: Dict[str, Any] = {
                "text": text,
                "language": "en",
                "file_path": output_path,
                "speed": speed,
            }
            if voice_sample_path and Path(voice_sample_path).exists():
                kwargs["speaker_wav"] = voice_sample_path
            self.xtts_instance.tts_to_file(**kwargs)
            duration = self._wav_duration(Path(output_path))
            timestamps = self._approximate_word_timestamps(text, duration)
            self._mark_engine_used(ENGINE_XTTS)
            return self.make_response(
                True,
                {
                    "audio_path": output_path,
                    "word_timestamps": timestamps,
                    "duration": duration,
                },
                duration_ms=_ms(started),
            )
        except Exception as exc:  # noqa: BLE001
            self.log.error("XTTS generation failed: %s", exc)
            return self._synthetic_speech_fallback(
                text, output_path, settings, engine=ENGINE_XTTS, started=started
            )
        finally:
            # Always unload XTTS after generation attempt (RAM)
            self.unload_engine_from_memory(ENGINE_XTTS)

    # ------------------------------------------------------------------
    # Emotion / pause / tags
    # ------------------------------------------------------------------

    def apply_emotion_parameters(
        self, voice_profile: Dict[str, Any], emotion: str
    ) -> Dict[str, Any]:
        """Adjust speed/pitch/volume from base profile using emotion preset.

        Args:
            voice_profile: Base character profile.
            emotion: Emotion name (supports aliases).

        Returns:
            Dict with speed, pitch, volume, emotion, pause_multiplier.
        """
        key = self._normalize_emotion(emotion)
        preset = EMOTION_PRESETS.get(key, EMOTION_PRESETS["neutral"])
        base_speed = float(voice_profile.get("speed", 1.0))
        base_pitch = float(voice_profile.get("pitch", 0.0))
        base_volume = float(voice_profile.get("volume", 1.0))
        speed = base_speed * float(preset["speed_mult"])
        # Micro-variation ±2% for naturalness
        speed = speed * (1.0 + random.uniform(-0.02, 0.02))
        speed = max(0.5, min(2.0, speed))
        pitch = base_pitch + float(preset["pitch_off"])
        pitch = max(-12.0, min(12.0, pitch))
        volume = base_volume * float(preset["vol_mult"])
        volume = max(0.0, min(2.0, volume))
        pause_mult = PAUSE_EMOTION_MULTIPLIERS.get(key, 1.0)
        return {
            "speed": round(speed, 4),
            "pitch": round(pitch, 3),
            "volume": round(volume, 4),
            "emotion": key,
            "pause_multiplier": pause_mult,
            "speed_mult": preset["speed_mult"],
            "pitch_off": preset["pitch_off"],
            "vol_mult": preset["vol_mult"],
        }

    def generate_pause(
        self,
        pause_type: str,
        emotion: str = "neutral",
        speed: float = 1.0,
    ) -> float:
        """Generate a humanized pause duration with random variation.

        Args:
            pause_type: MICRO|SHORT|MEDIUM|LONG|DRAMATIC.
            emotion: Emotion context for multiplier.
            speed: Speaking rate (faster speech → shorter pauses).

        Returns:
            Duration in seconds.
        """
        ptype = str(pause_type or "SHORT").upper()
        if ptype not in PAUSE_BASE_DURATIONS:
            ptype = "SHORT"
        base = PAUSE_BASE_DURATIONS[ptype]
        variation = PAUSE_VARIATIONS[ptype]
        duration = base + random.uniform(-variation, variation)
        emotion_key = self._normalize_emotion(emotion)
        duration *= PAUSE_EMOTION_MULTIPLIERS.get(emotion_key, 1.0)
        rate = max(0.5, float(speed) if speed else 1.0)
        duration *= 1.0 / rate
        duration = max(0.10, min(5.00, duration))
        return round(duration, 3)

    def process_pause_tags(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract [PAUSE:TYPE] tags and return clean text + markers.

        Args:
            text: Raw dialogue text.

        Returns:
            (clean_text, markers) where each marker has word_index, type, duration.
        """
        markers: List[Dict[str, Any]] = []
        parts: List[str] = []
        last = 0
        word_index = 0
        for match in PAUSE_TAG_RE.finditer(text or ""):
            segment = text[last : match.start()]
            parts.append(segment)
            word_index += len(WORD_RE.findall(segment))
            ptype = match.group(1).upper()
            markers.append(
                {
                    "word_index": word_index,
                    "type": ptype,
                    "duration": None,  # filled later with emotion context
                    "char_position": match.start(),
                }
            )
            last = match.end()
        parts.append(text[last:] if text else "")
        clean = re.sub(r"\s+", " ", "".join(parts)).strip()
        return clean, markers

    def insert_pauses_into_audio(
        self,
        audio_path: str,
        pause_markers: List[Dict[str, Any]],
        word_timestamps: List[Dict[str, Any]],
        breathing: bool = False,
        character_profile: Optional[Dict[str, Any]] = None,
        emotion: str = "neutral",
        speed: float = 1.0,
    ) -> Dict[str, Any]:
        """Insert silence (and optional breath) at pause marker positions.

        Args:
            audio_path: WAV path to modify in place.
            pause_markers: Markers from process_pause_tags.
            word_timestamps: Word timing list.
            breathing: Whether to mix breath samples.
            character_profile: Optional profile for breath volume.
            emotion: Emotion for pause duration.
            speed: Speaking rate for pause duration.

        Returns:
            Standard response.
        """
        started = time.perf_counter()
        path = Path(audio_path)
        if not path.exists():
            return self._err(f"Audio not found: {audio_path}", started)
        try:
            from pydub import AudioSegment
        except ImportError:
            return self._err("pydub not installed", started)

        audio = AudioSegment.from_file(str(path))
        # Build pause list with durations and ms positions (process back-to-front)
        prepared: List[Tuple[int, int, str]] = []
        for marker in pause_markers:
            ptype = str(marker.get("type") or "SHORT").upper()
            duration = marker.get("duration")
            if duration is None:
                duration = self.generate_pause(ptype, emotion, speed)
            ms_pos = self._marker_to_ms(marker, word_timestamps, len(audio))
            prepared.append((ms_pos, int(float(duration) * 1000), ptype))
        prepared.sort(key=lambda item: item[0], reverse=True)

        for ms_pos, silence_ms, ptype in prepared:
            silence = AudioSegment.silent(duration=max(1, silence_ms))
            if breathing and BREATH_CHANCE.get(ptype, 0) > 0:
                if random.random() <= BREATH_CHANCE[ptype]:
                    breath = self._load_breath_segment(
                        ptype, character_profile, silence_ms
                    )
                    if breath is not None:
                        silence = (
                            breath.overlay(silence)
                            if len(breath) < silence_ms
                            else breath[:silence_ms]
                        )
            audio = audio[:ms_pos] + silence + audio[ms_pos:]

        audio.export(str(path), format="wav")
        return self.make_response(
            True,
            {"audio_path": str(path), "pauses_inserted": len(prepared)},
            duration_ms=_ms(started),
        )

    def add_breathing_sounds(
        self,
        audio_path: str,
        character_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Insert breath samples at detected sentence-boundary silences."""
        started = time.perf_counter()
        path = Path(audio_path)
        if not path.exists():
            return self._err(f"Audio not found: {audio_path}", started)
        if not character_profile.get("breathing_enabled"):
            return self.make_response(
                True,
                {"audio_path": str(path), "breaths_added": 0},
                duration_ms=_ms(started),
            )
        try:
            from pydub import AudioSegment
            from pydub.silence import detect_silence
        except ImportError:
            return self._err("pydub not installed", started)

        audio = AudioSegment.from_file(str(path))
        silences = detect_silence(audio, min_silence_len=300, silence_thresh=-40)
        breaths = 0
        # Process from end so indices stay valid
        for start_ms, end_ms in reversed(silences):
            gap = end_ms - start_ms
            if gap < 300:
                continue
            ptype = "MEDIUM" if gap < 1200 else "LONG"
            breath = self._load_breath_segment(ptype, character_profile, gap)
            if breath is None:
                continue
            insert_at = start_ms + 50
            chunk = breath[: min(len(breath), gap - 50)]
            audio = (
                audio[:insert_at]
                + chunk.overlay(audio[insert_at : insert_at + len(chunk)])
                + audio[insert_at + len(chunk) :]
            )
            breaths += 1
        audio.export(str(path), format="wav")
        return self.make_response(
            True,
            {"audio_path": str(path), "breaths_added": breaths},
            duration_ms=_ms(started),
        )

    def apply_voice_effects(
        self,
        audio_path: str,
        character_profile: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Apply EQ → reverb → special effect → volume → limiter chain.

        Order: highpass/gate basics, EQ, reverb, special, volume, limiter.
        Uses FFmpeg when available; otherwise applies pydub volume only.
        """
        started = time.perf_counter()
        path = Path(audio_path)
        if not path.exists():
            return self._err(f"Audio not found: {audio_path}", started)

        filters: List[str] = [
            "highpass=f=80",
            "agate=threshold=0.008:ratio=10:attack=1:release=200",
        ]
        eq_key = str(character_profile.get("eq_preset") or "documentary_male")
        eq = EQ_PRESETS.get(eq_key, "")
        if eq:
            filters.append(eq)
        reverb_key = str(
            character_profile.get("reverb_preset")
            or character_profile.get("reverb")
            or "none"
        )
        reverb = REVERB_PRESETS.get(reverb_key, "")
        if reverb:
            filters.append(reverb)
        special_key = str(character_profile.get("special_effect") or "none")
        special = SPECIAL_EFFECTS.get(special_key, "")
        if special:
            filters.append(special)
        volume = float(
            (params or {}).get("volume", character_profile.get("volume", 1.0))
        )
        if abs(volume - 1.0) > 0.001:
            filters.append(f"volume={volume:.4f}")
        filters.append("alimiter=limit=0.95:level=false")
        filter_chain = ",".join(filters)

        ffmpeg = self._find_ffmpeg()
        if ffmpeg is None:
            # STATUS: NOT VERIFIED full chain — apply volume via pydub only
            return self._apply_volume_pydub(path, volume, started, filter_chain)

        temp_out = path.with_suffix(".fx.wav")
        cmd = [
            str(ffmpeg),
            "-y",
            "-i",
            str(path),
            "-af",
            filter_chain,
            "-ar",
            "48000",
            "-ac",
            "2",
            str(temp_out),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, check=False
            )
            if proc.returncode != 0 or not temp_out.exists():
                self.log.warning("FFmpeg effects failed: %s", proc.stderr[-500:])
                return self._apply_volume_pydub(path, volume, started, filter_chain)
            temp_out.replace(path)
            return self.make_response(
                True,
                {
                    "audio_path": str(path),
                    "filter_chain": filter_chain,
                    "ffmpeg": True,
                },
                duration_ms=_ms(started),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self.log.warning("FFmpeg effects error: %s", exc)
            return self._apply_volume_pydub(path, volume, started, filter_chain)

    # ------------------------------------------------------------------
    # Engine install / test / memory
    # ------------------------------------------------------------------

    def get_available_engines(self) -> Dict[str, Any]:
        """Return installed/available engine status without loading models."""
        engines = []
        if self._find_piper_binary() is not None:
            engines.append(
                {"name": ENGINE_PIPER, "status": "available", "loaded": False}
            )
        else:
            engines.append(
                {"name": ENGINE_PIPER, "status": "not_installed", "loaded": False}
            )
        kokoro_status = "available" if self._kokoro_importable() else "not_installed"
        engines.append(
            {
                "name": ENGINE_KOKORO,
                "status": kokoro_status,
                "loaded": self.kokoro_instance is not None,
            }
        )
        xtts_status = "available" if self._xtts_importable() else "not_installed"
        engines.append(
            {
                "name": ENGINE_XTTS,
                "status": xtts_status,
                "loaded": self.xtts_instance is not None,
            }
        )
        return self.make_response(True, {"engines": engines})

    def install_engine(self, engine_file_path: str | Path) -> Dict[str, Any]:
        """Install engine/model file into engines/ folder tree."""
        started = time.perf_counter()
        src = Path(engine_file_path)
        if not src.exists():
            return self._err(f"File not found: {src}", started)
        suffix = src.suffix.lower()
        if suffix == ".exe" or src.name.lower().startswith("piper"):
            dest_dir = self._project_root / "engines" / "piper"
            engine = ENGINE_PIPER
        elif suffix in (".onnx",):
            dest_dir = self._project_root / "engines" / "piper" / "models"
            engine = ENGINE_PIPER
        elif suffix in (".pt", ".pth", ".bin"):
            dest_dir = self._project_root / "engines" / "xtts" / "models"
            engine = ENGINE_XTTS
        elif suffix == ".zip":
            dest_dir = self._project_root / "engines" / "kokoro" / "models"
            engine = ENGINE_KOKORO
        else:
            dest_dir = self._project_root / "engines" / "kokoro" / "models"
            engine = ENGINE_KOKORO
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        self._record_engine_install(engine, str(dest))
        return self.make_response(
            True,
            {"engine": engine, "status": "installed", "path": str(dest)},
            duration_ms=_ms(started),
        )

    def install_engine_from_url(self, url: str, admin_password: str) -> Dict[str, Any]:
        """Download and install engine; requires admin password IAMKING."""
        started = time.perf_counter()
        if admin_password != ADMIN_PASSWORD:
            return self._err("Invalid admin password", started)
        if not url:
            return self._err("url is required", started)
        try:
            import requests
        except ImportError:
            return self._err("requests package not installed", started)
        temp_dir = self._project_root / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        filename = url.rstrip("/").split("/")[-1] or "engine_download.bin"
        dest = temp_dir / filename
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with dest.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            handle.write(chunk)
            return self.install_engine(dest)
        except Exception as exc:  # noqa: BLE001
            return self._err(f"Download failed: {exc}", started)

    def test_engine(self, engine_name: str) -> Dict[str, Any]:
        """Generate a short test phrase with the named engine."""
        started = time.perf_counter()
        out = self._project_root / "temp" / f"tts_test_{engine_name}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        profile = {
            "engine": engine_name,
            "voice_model": "default",
            "speed": 1.0,
            "pitch": 0.0,
            "volume": 1.0,
            "default_emotion": "neutral",
            "reverb_preset": "none",
            "eq_preset": "flat",
            "breathing_enabled": False,
        }
        result = self.generate_audio(
            "This is a test of the Autopilot system.",
            profile,
            out,
        )
        result["duration_ms"] = _ms(started)
        return result

    def unload_engine_from_memory(self, engine_name: str) -> Dict[str, Any]:
        """Explicitly free a loaded engine from RAM."""
        started = time.perf_counter()
        before = self._rss_mb()
        name = str(engine_name or "").lower()
        if name == ENGINE_KOKORO:
            self.kokoro_instance = None
        elif name == ENGINE_XTTS:
            self.xtts_instance = None
        # Piper is subprocess-only
        gc.collect()
        after = self._rss_mb()
        freed = max(0.0, before - after)
        self.log.info("Unloaded %s; RSS delta ~%.1f MB", name, freed)
        return self.make_response(
            True,
            {
                "engine": name,
                "rss_before_mb": before,
                "rss_after_mb": after,
                "freed_mb": freed,
            },
            duration_ms=_ms(started),
        )

    def check_ram_and_manage_engines(self) -> Dict[str, Any]:
        """Unload heavy engines when free RAM is low or idle timeout exceeded."""
        actions: List[str] = []
        free_mb = self._available_ram_mb()
        now = time.time()
        if (
            free_mb
            and free_mb < XTTS_UNLOAD_BELOW_MB
            and self.xtts_instance is not None
        ):
            self.unload_engine_from_memory(ENGINE_XTTS)
            actions.append("unloaded_xtts_low_ram")
        if (
            free_mb
            and free_mb < KOKORO_UNLOAD_BELOW_MB
            and self.kokoro_instance is not None
        ):
            self.unload_engine_from_memory(ENGINE_KOKORO)
            actions.append("unloaded_kokoro_low_ram")
        # Idle unload for kokoro
        last_kokoro = self._last_use.get(ENGINE_KOKORO, 0)
        if (
            self.kokoro_instance is not None
            and last_kokoro
            and now - last_kokoro > IDLE_UNLOAD_SECONDS
            and free_mb
            and free_mb < 2000
        ):
            self.unload_engine_from_memory(ENGINE_KOKORO)
            actions.append("unloaded_kokoro_idle")
        return self.make_response(
            True, {"actions": actions, "available_ram_mb": free_mb}
        )

    def list_emotions(self) -> List[str]:
        """Return all supported emotion names (28 base presets)."""
        return sorted(EMOTION_PRESETS.keys())

    def engines_loaded_in_memory(self) -> Dict[str, bool]:
        """Return which persistent engines currently hold models in RAM."""
        return {
            ENGINE_PIPER: False,  # never persistent
            ENGINE_KOKORO: self.kokoro_instance is not None,
            ENGINE_XTTS: self.xtts_instance is not None,
        }

    # ------------------------------------------------------------------
    # Lazy load helpers
    # ------------------------------------------------------------------

    def _ensure_kokoro_loaded(self) -> bool:
        """Import and construct Kokoro only when needed."""
        if self.kokoro_instance is not None:
            return True
        try:
            from kokoro_onnx import Kokoro  # type: ignore

            model_dir = self._project_root / "engines" / "kokoro" / "models"
            model_path = (
                next(model_dir.glob("*.onnx"), None) if model_dir.exists() else None
            )
            voices_path = (
                next(model_dir.glob("*voices*"), None) if model_dir.exists() else None
            )
            if model_path is None:
                self.log.warning("Kokoro models not found under %s", model_dir)
                return False
            self.kokoro_instance = Kokoro(
                str(model_path), str(voices_path) if voices_path else str(model_path)
            )
            self.log.info("Kokoro loaded lazily from %s", model_path)
            return True
        except Exception as exc:  # noqa: BLE001
            self.log.warning("Kokoro not available: %s", exc)
            return False

    def _ensure_xtts_loaded(self) -> bool:
        """Import and construct XTTS only when needed."""
        if self.xtts_instance is not None:
            return True
        try:
            from TTS.api import TTS  # type: ignore

            self.xtts_instance = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            self.log.info("XTTS loaded lazily")
            return True
        except Exception as exc:  # noqa: BLE001
            self.log.warning("XTTS not available: %s", exc)
            return False

    def _kokoro_importable(self) -> bool:
        """True if kokoro package can be imported (does not load model)."""
        try:
            import importlib.util

            return importlib.util.find_spec("kokoro_onnx") is not None
        except Exception:  # noqa: BLE001
            return False

    def _xtts_importable(self) -> bool:
        """True if TTS package can be imported (does not load model)."""
        try:
            import importlib.util

            return importlib.util.find_spec("TTS") is not None
        except Exception:  # noqa: BLE001
            return False

    def _find_piper_binary(self) -> Optional[Path]:
        """Locate piper executable without loading models."""
        candidates = [
            self._project_root / "engines" / "piper" / "piper",
            self._project_root / "engines" / "piper" / "piper.exe",
            Path(shutil.which("piper") or ""),
        ]
        for path in candidates:
            if path and path.exists():
                return path
        return None

    def _find_piper_model(self, voice_model: str) -> Optional[Path]:
        """Find a Piper .onnx model matching voice name.

        D2a: a candidate is only usable when its REQUIRED sidecar
        <model>.onnx.json sits next to it — piper refuses to load a
        bare .onnx (crashes FileNotFoundError on the config). This
        also neutralises leftover junk models (e.g. test-written
        fake_model.onnx) that previously made every narration line
        retry piper, crash, and fall back after ~2s plus a full
        traceback in the log.
        """

        def _usable(candidate: Path) -> bool:
            return (candidate.parent / (candidate.name + ".json")).exists()

        model_dir = self._project_root / "engines" / "piper" / "models"
        if not model_dir.exists():
            return None
        exact = model_dir / f"{voice_model}.onnx"
        if exact.exists() and _usable(exact):
            return exact
        matches = [
            m for m in model_dir.glob(f"*{voice_model}*.onnx") if _usable(m)
        ]
        if matches:
            return matches[0]
        any_onnx = [m for m in model_dir.glob("*.onnx") if _usable(m)]
        return any_onnx[0] if any_onnx else None

    def _find_ffmpeg(self) -> Optional[Path]:
        """Locate ffmpeg binary."""
        hint = None
        try:
            hint = self.config.get("ffmpeg_path")
        except Exception:  # noqa: BLE001
            hint = None
        candidates = []
        if hint:
            candidates.append(Path(str(hint)))
        candidates.extend(
            [
                self._project_root / "engines" / "ffmpeg" / "ffmpeg",
                self._project_root / "engines" / "ffmpeg" / "ffmpeg.exe",
                Path(shutil.which("ffmpeg") or ""),
            ]
        )
        for path in candidates:
            if path and str(path) and path.exists():
                return path
        return None

    def _resolve_engine_with_fallback(self, engine: str) -> Optional[str]:
        """Fallback XTTS→kokoro→piper when preferred engine missing."""
        order = [engine]
        for candidate in (ENGINE_XTTS, ENGINE_KOKORO, ENGINE_PIPER):
            if candidate not in order:
                order.append(candidate)
        available = self.get_available_engines()["data"]["engines"]
        status = {item["name"]: item["status"] for item in available}
        for name in order:
            if status.get(name) == "available":
                return name
            # Piper may still use synthetic fallback for offline tests
            if name == ENGINE_PIPER:
                return ENGINE_PIPER
        return ENGINE_PIPER

    def _ensure_voice_matches_engine(
        self, requested_engine: str, resolved_engine: str, voice_model: str
    ) -> str:
        """Substitute the engine default when the voice is incompatible.

        After engine fallback the requested voice may only be valid for
        the requested engine (e.g. a Piper voice such as "british_male_01"
        reaching Kokoro is rejected, which previously surfaced as
        synthetic beep audio). In that case log the substitution and
        return the resolved engine's default voice; the synthetic
        fallback stays reserved for genuine synthesis failures.
        """
        if self._voice_supported_by_engine(resolved_engine, voice_model):
            return voice_model
        replacement = ENGINE_DEFAULT_VOICES.get(resolved_engine, "default")
        self.log.warning(
            "Voice '%s' (requested engine '%s') is not compatible with "
            "resolved engine '%s'; using default voice '%s' instead",
            voice_model,
            requested_engine,
            resolved_engine,
            replacement,
        )
        return replacement

    def _voice_supported_by_engine(self, engine: str, voice_model: str) -> bool:
        """Check a voice name against the resolved engine's naming scheme."""
        name = (voice_model or "").strip()
        if engine == ENGINE_KOKORO:
            return bool(KOKORO_VOICE_RE.match(name))
        if engine == ENGINE_PIPER:
            # _find_piper_model resolves unknown names to an installed
            # model; only Kokoro-style ids can never map to a Piper model.
            return bool(name) and not KOKORO_VOICE_RE.match(name)
        return True

    def _mark_engine_used(self, engine: str) -> None:
        """Record last-use timestamp for idle unload."""
        self._last_use[engine] = time.time()

    # ------------------------------------------------------------------
    # Audio utilities
    # ------------------------------------------------------------------

    def _synthetic_speech_fallback(
        self,
        text: str,
        output_path: str,
        settings: Dict[str, Any],
        engine: str,
        started: float,
    ) -> Dict[str, Any]:
        """Generate a tone-based WAV so offline tests can exercise the pipeline.

        STATUS: NOT a real TTS voice — used when engines are not installed.
        """
        words = WORD_RE.findall(text)
        speed = max(0.5, float(settings.get("speed", 1.0)))
        # ~0.35s per word adjusted by speed
        duration = max(0.4, len(words) * 0.35 / speed)
        sample_rate = 22050
        frequency = 180.0 + float(settings.get("pitch", 0.0)) * 8.0
        self._write_sine_wav(output_path, duration, frequency, sample_rate)
        timestamps = self._approximate_word_timestamps(text, duration)
        return self.make_response(
            True,
            {
                "audio_path": output_path,
                "word_timestamps": timestamps,
                "duration": duration,
                "engine": engine,
                "synthetic": True,
                "warning": "Synthetic fallback audio — real TTS engine not installed",
            },
            warnings=["Synthetic fallback audio used (engine not installed)"],
            duration_ms=_ms(started),
        )

    def _write_sine_wav(
        self,
        path: str | Path,
        duration: float,
        frequency: float,
        sample_rate: int,
    ) -> None:
        """Write a simple mono sine wave WAV file."""
        n_samples = int(duration * sample_rate)
        with wave.open(str(path), "w") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            frames = bytearray()
            for i in range(n_samples):
                # Soft envelope to avoid clicks
                env = 1.0
                if i < sample_rate * 0.02:
                    env = i / (sample_rate * 0.02)
                elif i > n_samples - sample_rate * 0.02:
                    env = max(0.0, (n_samples - i) / (sample_rate * 0.02))
                value = int(
                    10000 * env * math.sin(2 * math.pi * frequency * i / sample_rate)
                )
                frames.extend(struct.pack("<h", value))
            handle.writeframes(bytes(frames))

    def _write_wav_samples(self, path: str, samples: Any, sample_rate: int) -> None:
        """Write numpy/list samples to WAV via soundfile or wave."""
        try:
            import numpy as np
            import soundfile as sf

            arr = np.asarray(samples, dtype=np.float32)
            sf.write(path, arr, sample_rate)
            return
        except Exception:  # noqa: BLE001
            pass
        # Fallback minimal writer
        import numpy as np

        arr = np.asarray(samples, dtype=np.float32)
        arr = np.clip(arr, -1.0, 1.0)
        pcm = (arr * 32767).astype("<i2")
        with wave.open(path, "w") as handle:
            handle.setnchannels(1 if pcm.ndim == 1 else pcm.shape[1])
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())

    def _wav_duration(self, path: Path) -> float:
        """Return WAV duration in seconds."""
        try:
            with wave.open(str(path), "r") as handle:
                return handle.getnframes() / float(handle.getframerate())
        except Exception:  # noqa: BLE001
            return 0.0

    def _approximate_word_timestamps(
        self, text: str, duration: float
    ) -> List[Dict[str, Any]]:
        """Evenly distribute word timings across duration."""
        words = WORD_RE.findall(text)
        if not words:
            return []
        slot = duration / len(words)
        result = []
        for index, word in enumerate(words):
            start = index * slot
            end = start + slot * 0.9
            result.append(
                {"word": word, "start": round(start, 3), "end": round(end, 3)}
            )
        return result

    def _adjust_timestamps_for_pauses(
        self,
        timestamps: List[Dict[str, Any]],
        pause_markers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Shift word timestamps by inserted pause durations."""
        if not timestamps or not pause_markers:
            return timestamps
        # Sort markers by word_index
        markers = sorted(pause_markers, key=lambda m: int(m.get("word_index") or 0))
        offset = 0.0
        marker_i = 0
        adjusted: List[Dict[str, Any]] = []
        for index, word in enumerate(timestamps):
            while (
                marker_i < len(markers)
                and int(markers[marker_i].get("word_index") or 0) <= index
            ):
                duration = markers[marker_i].get("duration")
                if duration is None:
                    duration = self.generate_pause(
                        str(markers[marker_i].get("type") or "SHORT")
                    )
                offset += float(duration)
                marker_i += 1
            adjusted.append(
                {
                    "word": word.get("word"),
                    "start": round(float(word.get("start", 0)) + offset, 3),
                    "end": round(float(word.get("end", 0)) + offset, 3),
                }
            )
        return adjusted

    def _marker_to_ms(
        self,
        marker: Dict[str, Any],
        word_timestamps: List[Dict[str, Any]],
        audio_len_ms: int,
    ) -> int:
        """Map pause marker word_index to millisecond position in audio."""
        idx = int(marker.get("word_index") or 0)
        if word_timestamps and 0 < idx <= len(word_timestamps):
            # Insert after word at idx-1
            end = float(word_timestamps[idx - 1].get("end", 0.0))
            return int(end * 1000)
        if word_timestamps and idx == 0:
            return 0
        # Fallback: proportional
        return min(audio_len_ms, max(0, int(audio_len_ms * 0.5)))

    def _load_breath_segment(
        self,
        pause_type: str,
        character_profile: Optional[Dict[str, Any]],
        max_ms: int,
    ) -> Any:
        """Load a breath WAV if present; else synthesize a quiet noise burst."""
        try:
            from pydub import AudioSegment
            from pydub.generators import WhiteNoise
        except ImportError:
            return None
        gender = "male"
        if character_profile:
            model = str(character_profile.get("voice_model") or "").lower()
            if "female" in model or "woman" in model:
                gender = "female"
        breath_dir = self._project_root / "assets" / "sfx" / "breathing"
        candidates = [
            breath_dir / f"breath_gentle_{gender}.wav",
            breath_dir / f"breath_normal_{gender}.wav",
            breath_dir / f"breath_deep_{gender}.wav",
        ]
        segment = None
        for candidate in candidates:
            if candidate.exists():
                segment = AudioSegment.from_file(str(candidate))
                break
        if segment is None:
            # Synthetic soft noise as stand-in breath
            duration = min(max_ms, 300 if pause_type in ("SHORT", "MEDIUM") else 500)
            segment = WhiteNoise().to_audio_segment(duration=duration, volume=-35)
        vol_range = BREATH_VOLUME_RANGE.get(pause_type, (0.1, 0.15))
        target_ratio = random.uniform(vol_range[0], vol_range[1])
        # pydub volume is dB; approximate scale
        if target_ratio <= 0:
            return None
        db_adjust = 20 * math.log10(max(target_ratio, 0.01))
        return segment + db_adjust

    def _apply_pitch_shift(self, path: Path, semitones: float) -> None:
        """Pitch-shift WAV using FFmpeg asetrate when available."""
        ffmpeg = self._find_ffmpeg()
        if ffmpeg is None or abs(semitones) < 0.01:
            return
        factor = 2 ** (semitones / 12.0)
        temp = path.with_suffix(".pitch.wav")
        cmd = [
            str(ffmpeg),
            "-y",
            "-i",
            str(path),
            "-af",
            f"asetrate=44100*{factor:.6f},aresample=44100",
            str(temp),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, check=False
            )
            if proc.returncode == 0 and temp.exists():
                temp.replace(path)
        except (OSError, subprocess.SubprocessError):
            pass

    def _apply_volume_pydub(
        self,
        path: Path,
        volume: float,
        started: float,
        filter_chain: str,
    ) -> Dict[str, Any]:
        """Volume-only fallback when FFmpeg is unavailable."""
        try:
            from pydub import AudioSegment

            audio = AudioSegment.from_file(str(path))
            if abs(volume - 1.0) > 0.001:
                db = 20 * math.log10(max(volume, 0.01))
                audio = audio + db
            audio.export(str(path), format="wav")
            return self.make_response(
                True,
                {
                    "audio_path": str(path),
                    "filter_chain": filter_chain,
                    "ffmpeg": False,
                    "warning": "FFmpeg not found; applied volume-only effects",
                },
                warnings=["FFmpeg not found — full effect chain NOT VERIFIED"],
                duration_ms=_ms(started),
            )
        except Exception as exc:  # noqa: BLE001
            return self._err(f"Effects failed: {exc}", started)

    def _record_engine_install(self, engine: str, path: str) -> None:
        """Best-effort DB update for engine_installations."""
        try:
            now = utc_now_str()
            row = self.db.db.fetch_one(
                "SELECT id FROM engine_installations WHERE engine_name = ?",
                (engine,),
            )
            if row:
                self.db.db.execute(
                    "UPDATE engine_installations SET install_path = ?, status = ?, "
                    "updated_at = ? WHERE engine_name = ?",
                    (path, "installed", now, engine),
                )
        except Exception as exc:  # noqa: BLE001
            self.log.debug("engine install DB update skipped: %s", exc)

    def _normalize_emotion(self, emotion: str) -> str:
        """Map aliases to base emotion keys."""
        key = str(emotion or "neutral").strip().lower()
        key = EMOTION_ALIASES.get(key, key)
        if key not in EMOTION_PRESETS:
            return "neutral"
        return key

    def _available_ram_mb(self) -> float:
        """Return free system RAM in MB."""
        try:
            import psutil

            return float(psutil.virtual_memory().available) / 1024 / 1024
        except Exception:  # noqa: BLE001
            return 0.0

    def _rss_mb(self) -> float:
        """Return current process RSS in MB."""
        try:
            import psutil

            return float(psutil.Process(os.getpid()).memory_info().rss) / 1024 / 1024
        except Exception:  # noqa: BLE001
            return 0.0

    def _err(self, message: str, started: float) -> Dict[str, Any]:
        """Build error response."""
        return self.make_response(
            False,
            data={
                "error_code": "TTS_ERROR",
                "user_message": message,
                "is_recoverable": True,
            },
            error=message,
            duration_ms=_ms(started),
        )


def _ms(started: float) -> float:
    """Elapsed milliseconds."""
    return round((time.perf_counter() - started) * 1000.0, 3)
