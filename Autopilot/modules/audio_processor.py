"""Audio processor: narration assembly, ducking, mix, LUFS, limiter.

Required BaseModule for final render audio. Uses NumPy for ducking envelopes,
PyDub for mixing, and FFmpeg loudnorm when available (graceful fallback).
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.service_container import BaseModule, ServiceContainer

MODULE_NAME = "audio_processor"

DEFAULT_DUCKING: Dict[str, float] = {
    "ducking_threshold": 0.02,
    "ducking_depth": 0.15,
    "ducking_ceiling": 0.50,
    "attack_time": 0.30,
    "release_time": 0.80,
    "min_silence_duration": 0.50,
    "window_sec": 0.02,
    "hop_sec": 0.01,
}

SAMPLE_RATE = 48000


class AudioProcessor(BaseModule):
    """Mix narration, music, SFX, and ambient into a final render track."""

    def __init__(self, container: ServiceContainer) -> None:
        """Initialize without loading heavy audio into memory."""
        super().__init__(container, MODULE_NAME)
        self._project_root = Path.cwd()
        try:
            cfg = getattr(self.config, "config_folder", None)
            if cfg is not None:
                self._project_root = Path(cfg).resolve().parent
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Public orchestration
    # ------------------------------------------------------------------

    def generate_final_mix(
        self,
        project_id: str,
        output_path: str | Path,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run full mix pipeline for a project.

        Args:
            project_id: Project UUID (used for temp naming / future DB).
            output_path: Final WAV path.
            settings: Optional ducking/volume overrides and track paths.

        Returns:
            Standard response with final path and metadata.
        """
        started = time.perf_counter()
        cfg = self._merge_settings(settings)
        work = self._work_dir(project_id)
        work.mkdir(parents=True, exist_ok=True)

        narration = cfg.get("narration_path")
        if not narration:
            built = self.build_narration_track(
                project_id, cfg.get("line_paths") or [], work / "narration.wav"
            )
            if not built["success"]:
                return built
            narration = built["data"]["audio_path"]

        music = cfg.get("music_path")
        ducked = None
        if music and Path(music).exists():
            duck_out = work / "music_ducked.wav"
            ducked_resp = self.apply_music_ducking(narration, music, cfg, duck_out)
            if not ducked_resp["success"]:
                return ducked_resp
            ducked = ducked_resp["data"]["audio_path"]

        mixed_path = work / "mixed.wav"
        mix_resp = self.mix_tracks(
            narration,
            ducked,
            cfg.get("sfx_list") or [],
            cfg.get("ambient_path"),
            mixed_path,
            cfg,
        )
        if not mix_resp["success"]:
            return mix_resp

        limited = work / "limited.wav"
        lim = self.apply_limiter(
            mix_resp["data"]["audio_path"], limited, ceiling_db=-1.0
        )
        if not lim["success"]:
            return lim

        final = Path(output_path)
        final.parent.mkdir(parents=True, exist_ok=True)
        norm = self.normalize_to_lufs(
            lim["data"]["audio_path"], final, target_lufs=-14.0
        )
        if not norm["success"]:
            # Fall back to limited file if loudnorm unavailable
            shutil.copy2(lim["data"]["audio_path"], final)
            warnings = list(norm.get("warnings") or []) + [
                "LUFS normalization unavailable; used limited mix"
            ]
            return self.make_response(
                True,
                {
                    "audio_path": str(final),
                    "duration": self._wav_duration(final),
                    "lufs_normalized": False,
                    "peak_db": self.measure_peak_db(str(final)),
                },
                warnings=warnings,
                duration_ms=_ms(started),
            )

        return self.make_response(
            True,
            {
                "audio_path": str(final),
                "duration": self._wav_duration(final),
                "lufs_normalized": True,
                "peak_db": self.measure_peak_db(str(final)),
                "target_lufs": -14.0,
            },
            duration_ms=_ms(started),
        )

    def build_narration_track(
        self,
        project_id: str,
        line_paths: Sequence[str | Path],
        output_path: str | Path,
        pause_seconds: float = 0.4,
        crossfade_ms: int = 20,
    ) -> Dict[str, Any]:
        """Concatenate TTS line WAVs with pauses and light crossfades."""
        started = time.perf_counter()
        paths = [Path(p) for p in line_paths if p and Path(p).exists()]
        if not paths:
            return self._err("No narration line files provided", started)
        try:
            from pydub import AudioSegment
        except ImportError:
            return self._err("pydub not installed", started)

        combined = AudioSegment.silent(duration=0)
        timestamps: List[Dict[str, Any]] = []
        silence = AudioSegment.silent(duration=int(max(0.0, pause_seconds) * 1000))
        fade = max(0, int(crossfade_ms))
        for index, path in enumerate(paths):
            seg = AudioSegment.from_file(str(path))
            seg = self._ensure_stereo_48k(seg)
            if index > 0:
                # Pause between lines, then optional micro fade into next clip
                combined = combined + silence
                if fade > 0 and len(seg) > fade and len(combined) > fade:
                    combined = combined.append(seg, crossfade=fade)
                else:
                    combined = combined + seg
            else:
                combined = combined + seg
            end_ms = len(combined)
            start_ms = end_ms - len(seg)
            if index > 0 and fade > 0:
                start_ms = max(0, end_ms - len(seg))
            timestamps.append(
                {
                    "index": index,
                    "path": str(path),
                    "start": start_ms / 1000.0,
                    "end": end_ms / 1000.0,
                }
            )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        combined.export(str(out), format="wav")
        return self.make_response(
            True,
            {
                "audio_path": str(out),
                "duration": len(combined) / 1000.0,
                "line_count": len(paths),
                "segment_timestamps": timestamps,
                "project_id": project_id,
            },
            duration_ms=_ms(started),
        )

    def apply_music_ducking(
        self,
        narration_path: str | Path,
        music_path: str | Path,
        settings: Optional[Dict[str, Any]],
        output_path: str | Path,
    ) -> Dict[str, Any]:
        """Duck music under narration using RMS envelope (File 08 algorithm)."""
        started = time.perf_counter()
        cfg = self._merge_settings(settings)
        try:
            narr, sr = self._read_audio(narration_path)
            music, sr2 = self._read_audio(music_path)
        except Exception as exc:  # noqa: BLE001
            return self._err(f"Failed to load audio: {exc}", started)

        if sr != sr2:
            music = self._resample_np(music, sr2, sr)
        music = self._match_length(music, len(narr))
        envelope = self.calculate_ducking_envelope(narration_path, cfg)
        if not envelope["success"]:
            return envelope
        env = np.asarray(envelope["data"]["envelope"], dtype=np.float64)
        env = self._match_length_1d(env, len(narr))
        ducked = self._apply_envelope(music, env)
        self._write_audio(output_path, ducked, sr)
        return self.make_response(
            True,
            {
                "audio_path": str(output_path),
                "sample_rate": sr,
                "duration": len(ducked) / float(sr),
            },
            duration_ms=_ms(started),
        )

    def calculate_ducking_envelope(
        self,
        narration_path: str | Path,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build smoothed music volume multipliers from narration RMS."""
        started = time.perf_counter()
        cfg = self._merge_settings(settings)
        narr, sr = self._read_audio(narration_path)
        mono = narr.mean(axis=1) if narr.ndim == 2 else narr
        speech_mask = self._speech_mask(mono, sr, cfg)
        speech_mask = self._fill_short_silences(speech_mask, sr, cfg)
        depth = float(cfg["ducking_depth"])
        ceiling = float(cfg["ducking_ceiling"])
        envelope = np.where(speech_mask > 0, depth, ceiling).astype(np.float64)
        smoothed = self._smooth_envelope(
            envelope,
            int(cfg["attack_time"] * sr),
            int(cfg["release_time"] * sr),
        )
        return self.make_response(
            True,
            {
                "envelope": smoothed,
                "sample_rate": sr,
                "speech_ratio": float(np.mean(speech_mask)),
            },
            duration_ms=_ms(started),
        )

    def mix_tracks(
        self,
        narration_path: str | Path,
        music_path: Optional[str | Path],
        sfx_list: Sequence[Dict[str, Any]],
        ambient_path: Optional[str | Path],
        output_path: str | Path,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Layer narration, ducked music, timed SFX, and ambient."""
        started = time.perf_counter()
        cfg = self._merge_settings(settings)
        narr, sr = self._read_audio(narration_path)
        mix = narr.astype(np.float64)
        if music_path and Path(music_path).exists():
            music, msr = self._read_audio(music_path)
            if msr != sr:
                music = self._resample_np(music, msr, sr)
            music = self._match_length(music, len(mix))
            music_vol = float(cfg.get("music_volume", 1.0))
            mix = mix + music * music_vol
        if ambient_path and Path(ambient_path).exists():
            amb, asr = self._read_audio(ambient_path)
            if asr != sr:
                amb = self._resample_np(amb, asr, sr)
            amb = self._loop_to_length(amb, len(mix))
            amb_vol = float(cfg.get("ambient_volume", 0.2))
            mix = mix + amb * amb_vol
        mix = self._overlay_sfx(mix, sr, sfx_list, float(cfg.get("sfx_volume", 0.7)))
        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        if peak > 0.95:
            mix = mix * (0.95 / peak)
        self._write_audio(output_path, mix, sr)
        return self.make_response(
            True,
            {
                "audio_path": str(output_path),
                "duration": len(mix) / float(sr),
                "peak": peak,
                "sample_rate": sr,
            },
            duration_ms=_ms(started),
        )

    def normalize_to_lufs(
        self,
        audio_path: str | Path,
        output_path: str | Path,
        target_lufs: float = -14.0,
    ) -> Dict[str, Any]:
        """Two-pass FFmpeg loudnorm when available; RMS fallback otherwise."""
        started = time.perf_counter()
        src = Path(audio_path)
        if not src.exists():
            return self._err(f"Audio not found: {src}", started)
        ffmpeg = self._find_ffmpeg()
        if ffmpeg is None:
            return self._normalize_rms_fallback(src, output_path, target_lufs, started)
        measured = self._loudnorm_measure(ffmpeg, src, target_lufs)
        if measured is None:
            return self._normalize_rms_fallback(src, output_path, target_lufs, started)
        ok = self._loudnorm_apply(ffmpeg, src, Path(output_path), target_lufs, measured)
        if not ok:
            return self._normalize_rms_fallback(src, output_path, target_lufs, started)
        return self.make_response(
            True,
            {
                "audio_path": str(output_path),
                "target_lufs": target_lufs,
                "measured": measured,
                "two_pass": True,
                "ffmpeg": True,
            },
            duration_ms=_ms(started),
        )

    def apply_limiter(
        self,
        audio_path: str | Path,
        output_path: str | Path,
        ceiling_db: float = -1.0,
    ) -> Dict[str, Any]:
        """Peak limiter with ceiling (default -1 dBFS)."""
        started = time.perf_counter()
        data, sr = self._read_audio(audio_path)
        ceiling = 10 ** (ceiling_db / 20.0)
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        if peak > ceiling and peak > 0:
            data = data * (ceiling / peak)
        self._write_audio(output_path, data, sr)
        new_peak = float(np.max(np.abs(data))) if data.size else 0.0
        return self.make_response(
            True,
            {
                "audio_path": str(output_path),
                "peak_before": peak,
                "peak_after": new_peak,
                "ceiling": ceiling,
                "clipping": bool(new_peak > 1.0 + 1e-6),
            },
            duration_ms=_ms(started),
        )

    def detect_silence_regions(
        self,
        audio_path: str | Path,
        threshold_db: float = -40.0,
        min_duration: float = 0.1,
    ) -> Dict[str, Any]:
        """Return silence regions as (start, end) second tuples."""
        started = time.perf_counter()
        data, sr = self._read_audio(audio_path)
        mono = data.mean(axis=1) if data.ndim == 2 else data
        # Convert threshold_db to linear RMS-ish amplitude
        thresh = 10 ** (threshold_db / 20.0)
        window = max(1, int(0.02 * sr))
        hop = max(1, int(0.01 * sr))
        silent = np.zeros(len(mono), dtype=bool)
        for i in range(0, max(1, len(mono) - window), hop):
            frame = mono[i : i + window]
            rms = float(np.sqrt(np.mean(frame**2))) if frame.size else 0.0
            if rms < thresh:
                silent[i : i + window] = True
        regions = self._bool_runs_to_regions(silent, sr, min_duration)
        return self.make_response(
            True,
            {"regions": regions, "count": len(regions), "threshold_db": threshold_db},
            duration_ms=_ms(started),
        )

    def measure_peak_db(self, audio_path: str | Path) -> float:
        """Return peak level in dBFS."""
        data, _ = self._read_audio(audio_path)
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        if peak <= 0:
            return -120.0
        return 20.0 * math.log10(peak)

    def measure_approx_lufs(self, audio_path: str | Path) -> float:
        """Rough integrated loudness estimate from RMS (not true ITU BS.1770)."""
        data, _ = self._read_audio(audio_path)
        mono = data.mean(axis=1) if data.ndim == 2 else data
        rms = float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0
        if rms <= 0:
            return -70.0
        # Empirical offset so sine ~ -3 dBFS peak tracks near -14 after normalize
        return 20.0 * math.log10(rms) + 0.0

    def crossfade_join(
        self,
        path_a: str | Path,
        path_b: str | Path,
        output_path: str | Path,
        crossfade_ms: int = 50,
    ) -> Dict[str, Any]:
        """Join two audio files with a short crossfade (no click)."""
        started = time.perf_counter()
        try:
            from pydub import AudioSegment
        except ImportError:
            return self._err("pydub not installed", started)
        a = self._ensure_stereo_48k(AudioSegment.from_file(str(path_a)))
        b = self._ensure_stereo_48k(AudioSegment.from_file(str(path_b)))
        joined = a.append(b, crossfade=max(1, crossfade_ms))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        joined.export(str(output_path), format="wav")
        return self.make_response(
            True,
            {"audio_path": str(output_path), "duration": len(joined) / 1000.0},
            duration_ms=_ms(started),
        )

    # ------------------------------------------------------------------
    # Ducking internals
    # ------------------------------------------------------------------

    def _speech_mask(
        self, mono: np.ndarray, sr: int, cfg: Dict[str, Any]
    ) -> np.ndarray:
        """Binary speech mask from RMS windows."""
        window = max(1, int(float(cfg["window_sec"]) * sr))
        hop = max(1, int(float(cfg["hop_sec"]) * sr))
        thr = float(cfg["ducking_threshold"])
        mask = np.zeros(len(mono), dtype=np.float64)
        if len(mono) < window:
            rms = float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0
            return np.ones(len(mono)) if rms > thr else mask
        num_frames = max(1, (len(mono) - window) // hop)
        for i in range(num_frames):
            start = i * hop
            end = start + window
            frame = mono[start:end]
            rms = float(np.sqrt(np.mean(frame**2)))
            if rms > thr:
                mask[start:end] = 1.0
        return mask

    def _fill_short_silences(
        self, mask: np.ndarray, sr: int, cfg: Dict[str, Any]
    ) -> np.ndarray:
        """Ignore silence gaps shorter than min_silence_duration."""
        min_samples = int(float(cfg["min_silence_duration"]) * sr)
        out = mask.copy()
        in_silence = False
        silence_start = 0
        for i in range(len(out)):
            if out[i] == 0 and not in_silence:
                in_silence = True
                silence_start = i
            elif out[i] == 1 and in_silence:
                if i - silence_start < min_samples:
                    out[silence_start:i] = 1.0
                in_silence = False
        if in_silence and len(out) - silence_start < min_samples:
            out[silence_start:] = 1.0
        return out

    def _smooth_envelope(
        self, envelope: np.ndarray, attack_samples: int, release_samples: int
    ) -> np.ndarray:
        """One-pole attack/release smoother."""
        attack_samples = max(1, attack_samples)
        release_samples = max(1, release_samples)
        smoothed = envelope.copy()
        for i in range(1, len(smoothed)):
            if smoothed[i] < smoothed[i - 1]:
                alpha = 1.0 / attack_samples
            else:
                alpha = 1.0 / release_samples
            smoothed[i] = alpha * smoothed[i] + (1.0 - alpha) * smoothed[i - 1]
        return smoothed

    def _apply_envelope(self, music: np.ndarray, env: np.ndarray) -> np.ndarray:
        """Multiply music by envelope (mono or stereo)."""
        if music.ndim == 2:
            return music * env[:, np.newaxis]
        return music * env

    # ------------------------------------------------------------------
    # Mix helpers
    # ------------------------------------------------------------------

    def _overlay_sfx(
        self,
        mix: np.ndarray,
        sr: int,
        sfx_list: Sequence[Dict[str, Any]],
        default_vol: float,
    ) -> np.ndarray:
        """Add SFX clips at timestamps (seconds)."""
        out = mix.copy()
        for item in sfx_list:
            path = item.get("path")
            if not path or not Path(path).exists():
                continue
            clip, csr = self._read_audio(path)
            if csr != sr:
                clip = self._resample_np(clip, csr, sr)
            start = int(float(item.get("timestamp", 0.0)) * sr)
            vol = float(item.get("volume", default_vol))
            clip = clip * vol
            end = min(len(out), start + len(clip))
            if start >= len(out) or end <= start:
                continue
            piece = clip[: end - start]
            if out.ndim == 2 and piece.ndim == 1:
                piece = np.stack([piece, piece], axis=1)
            if out.ndim == 1 and piece.ndim == 2:
                piece = piece.mean(axis=1)
            out[start:end] = out[start:end] + piece
        return out

    def _loop_to_length(self, audio: np.ndarray, length: int) -> np.ndarray:
        """Repeat audio to target sample length."""
        if len(audio) == 0:
            shape = (length, 2) if audio.ndim == 2 else (length,)
            return np.zeros(shape, dtype=np.float64)
        if len(audio) >= length:
            return audio[:length]
        reps = int(math.ceil(length / len(audio)))
        tiled = np.tile(audio, (reps, 1) if audio.ndim == 2 else reps)
        return tiled[:length]

    def _match_length(self, audio: np.ndarray, length: int) -> np.ndarray:
        """Trim or pad audio to length."""
        if len(audio) == length:
            return audio
        if len(audio) > length:
            return audio[:length]
        pad = length - len(audio)
        if audio.ndim == 2:
            return np.vstack([audio, np.zeros((pad, audio.shape[1]))])
        return np.concatenate([audio, np.zeros(pad)])

    def _match_length_1d(self, arr: np.ndarray, length: int) -> np.ndarray:
        """Trim/pad 1D array."""
        if len(arr) == length:
            return arr
        if len(arr) > length:
            return arr[:length]
        return np.concatenate(
            [arr, np.full(length - len(arr), arr[-1] if len(arr) else 0.0)]
        )

    # ------------------------------------------------------------------
    # FFmpeg / I/O
    # ------------------------------------------------------------------

    def _loudnorm_measure(
        self, ffmpeg: Path, src: Path, target_lufs: float
    ) -> Optional[Dict[str, str]]:
        """First-pass loudnorm measurement."""
        cmd = [
            str(ffmpeg),
            "-hide_banner",
            "-i",
            str(src),
            "-af",
            f"loudnorm=I={target_lufs}:LRA=11:TP=-1:print_format=json",
            "-f",
            "null",
            "-",
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120, check=False
            )
            text = (proc.stderr or "") + (proc.stdout or "")
            return self._parse_loudnorm_json(text)
        except (OSError, subprocess.SubprocessError) as exc:
            self.log.warning("loudnorm measure failed: %s", exc)
            return None

    def _loudnorm_apply(
        self,
        ffmpeg: Path,
        src: Path,
        dest: Path,
        target_lufs: float,
        measured: Dict[str, str],
    ) -> bool:
        """Second-pass loudnorm apply."""
        filt = (
            f"loudnorm=I={target_lufs}:LRA=11:TP=-1:"
            f"measured_I={measured.get('input_i', target_lufs)}:"
            f"measured_LRA={measured.get('input_lra', 11)}:"
            f"measured_TP={measured.get('input_tp', -1)}:"
            f"measured_thresh={measured.get('input_thresh', -24)}:"
            f"offset={measured.get('target_offset', 0)}:linear=true"
        )
        cmd = [
            str(ffmpeg),
            "-y",
            "-i",
            str(src),
            "-af",
            filt,
            "-ar",
            str(SAMPLE_RATE),
            str(dest),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=180, check=False
            )
            return proc.returncode == 0 and dest.exists()
        except (OSError, subprocess.SubprocessError):
            return False

    def _parse_loudnorm_json(self, text: str) -> Optional[Dict[str, str]]:
        """Extract loudnorm JSON block from FFmpeg stderr."""
        start = text.rfind("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
            return {str(k): str(v) for k, v in data.items()}
        except json.JSONDecodeError:
            return None

    def _normalize_rms_fallback(
        self,
        src: Path,
        output_path: str | Path,
        target_lufs: float,
        started: float,
    ) -> Dict[str, Any]:
        """Approximate loudness normalize without FFmpeg loudnorm."""
        data, sr = self._read_audio(src)
        mono = data.mean(axis=1) if data.ndim == 2 else data
        rms = float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0
        # Map target_lufs roughly to RMS target
        target_rms = 10 ** ((target_lufs + 3.0) / 20.0)
        if rms > 1e-9:
            data = data * (target_rms / rms)
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        ceiling = 10 ** (-1.0 / 20.0)
        if peak > ceiling:
            data = data * (ceiling / peak)
        self._write_audio(output_path, data, sr)
        return self.make_response(
            True,
            {
                "audio_path": str(output_path),
                "target_lufs": target_lufs,
                "two_pass": False,
                "ffmpeg": False,
                "approx_lufs": self.measure_approx_lufs(output_path),
            },
            warnings=[
                "FFmpeg loudnorm unavailable — used RMS approximation (NOT VERIFIED true LUFS)"
            ],
            duration_ms=_ms(started),
        )

    def _find_ffmpeg(self) -> Optional[Path]:
        """Locate ffmpeg binary."""
        candidates: List[Path] = []
        try:
            hint = self.config.get("ffmpeg_path")
            if hint:
                candidates.append(Path(str(hint)))
        except Exception:  # noqa: BLE001
            pass
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

    def _read_audio(self, path: str | Path) -> Tuple[np.ndarray, int]:
        """Read WAV/audio as float64 numpy array shape (n,) or (n, ch)."""
        path = Path(path)
        try:
            import soundfile as sf

            data, sr = sf.read(str(path), always_2d=False)
            return np.asarray(data, dtype=np.float64), int(sr)
        except Exception:  # noqa: BLE001
            pass
        # wave fallback for PCM wav
        with wave.open(str(path), "r") as handle:
            sr = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())
        if width == 2:
            arr = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32767.0
        elif width == 4:
            arr = np.frombuffer(frames, dtype="<i4").astype(np.float64) / 2147483648.0
        else:
            arr = np.frombuffer(frames, dtype=np.uint8).astype(np.float64)
            arr = (arr - 128.0) / 128.0
        if channels > 1:
            arr = arr.reshape(-1, channels)
        return arr, sr

    def _write_audio(self, path: str | Path, data: np.ndarray, sr: int) -> None:
        """Write float audio to WAV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        clipped = np.clip(data, -1.0, 1.0)
        try:
            import soundfile as sf

            sf.write(str(path), clipped, sr)
            return
        except Exception:  # noqa: BLE001
            pass
        pcm = (clipped * 32767.0).astype("<i2")
        channels = 1 if clipped.ndim == 1 else clipped.shape[1]
        with wave.open(str(path), "w") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(sr)
            handle.writeframes(pcm.tobytes())

    def _resample_np(self, data: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        """Simple linear resample."""
        if src_sr == dst_sr or len(data) == 0:
            return data
        duration = len(data) / float(src_sr)
        new_len = max(1, int(duration * dst_sr))
        x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        if data.ndim == 1:
            return np.interp(x_new, x_old, data)
        channels = [np.interp(x_new, x_old, data[:, c]) for c in range(data.shape[1])]
        return np.stack(channels, axis=1)

    def _ensure_stereo_48k(self, seg: Any) -> Any:
        """Normalize pydub segment to stereo 48k."""
        if seg.channels == 1:
            seg = seg.set_channels(2)
        if seg.frame_rate != SAMPLE_RATE:
            seg = seg.set_frame_rate(SAMPLE_RATE)
        return seg

    def _wav_duration(self, path: Path) -> float:
        """WAV duration seconds."""
        try:
            with wave.open(str(path), "r") as handle:
                return handle.getnframes() / float(handle.getframerate())
        except Exception:  # noqa: BLE001
            data, sr = self._read_audio(path)
            return len(data) / float(sr) if sr else 0.0

    def _bool_runs_to_regions(
        self, silent: np.ndarray, sr: int, min_duration: float
    ) -> List[Tuple[float, float]]:
        """Convert boolean silence mask to time regions."""
        regions: List[Tuple[float, float]] = []
        in_run = False
        start = 0
        min_samples = int(min_duration * sr)
        for i, flag in enumerate(silent):
            if flag and not in_run:
                in_run = True
                start = i
            elif not flag and in_run:
                if i - start >= min_samples:
                    regions.append((start / sr, i / sr))
                in_run = False
        if in_run and len(silent) - start >= min_samples:
            regions.append((start / sr, len(silent) / sr))
        return regions

    def _work_dir(self, project_id: str) -> Path:
        """Temp working directory for a project mix."""
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in project_id)[:32]
        return self._project_root / "temp" / f"mix_{safe}"

    def _merge_settings(self, settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge user settings over ducking defaults."""
        cfg = dict(DEFAULT_DUCKING)
        if settings:
            cfg.update(settings)
        return cfg

    def _err(self, message: str, started: float) -> Dict[str, Any]:
        """Error response helper."""
        return self.make_response(
            False,
            data={
                "error_code": "AUDIO_PROCESSING_ERROR",
                "user_message": message,
                "is_recoverable": True,
            },
            error=message,
            duration_ms=_ms(started),
        )


def _ms(started: float) -> float:
    """Elapsed milliseconds."""
    return round((time.perf_counter() - started) * 1000.0, 3)
