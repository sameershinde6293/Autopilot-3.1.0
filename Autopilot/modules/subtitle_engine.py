"""Subtitle engine: SRT generation from word timestamps and FFmpeg rendering.

Optional BaseModule (can be disabled safely). Generates SRT files from
word-level timestamps, builds FFmpeg force_style strings for burn-in, and
builds drawtext chains for animated styles (word-by-word, typewriter).

Spec sources: modules_specification.txt MODULE 07 (methods/algorithm) and
presets_and_configs.txt config/subtitle_style_presets.json (8 styles).
Very long drawtext chains are exported per-segment by export_engine
(DEBT-B11a); this module guards with MAX_CHAIN_WORDS.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.service_container import BaseModule, ServiceContainer

MODULE_NAME = "subtitle_engine"

SENTENCE_END = frozenset(".!?")
FFMPEG_TIMEOUT = 300
MAX_CHAIN_WORDS = 250  # DEBT-B11a: export_engine segments longer videos
MIN_BLOCK_MS = 200
OVERLAP_GAP_MS = 40

BUILTIN_STYLES: Dict[str, Dict[str, Any]] = {
    "word_by_word": {
        "name": "Word by Word Highlight",
        "animation_type": "word_highlight",
        "font_family": "Montserrat-Bold",
        "font_size": 48,
        "font_color": "#FFFFFF",
        "highlight_color": "#FFD700",
        "secondary_color": "#AAAAAA",
        "outline_color": "#000000",
        "outline_size": 3,
        "shadow_enabled": True,
        "shadow_offset_x": 2,
        "shadow_offset_y": 2,
        "background_enabled": True,
        "background_color": "#000000",
        "background_opacity": 0.35,
        "position": "bottom",
        "margin_v": 90,
        "margin_h": 120,
        "max_chars_per_line": 42,
        "max_lines": 2,
    },
    "documentary_classic": {
        "name": "Documentary Classic",
        "animation_type": "sentence_fade",
        "font_family": "Montserrat-Bold",
        "font_size": 48,
        "font_color": "#FFFFFF",
        "highlight_color": "#FFFFFF",
        "secondary_color": "#FFFFFF",
        "outline_color": "#000000",
        "outline_size": 4,
        "shadow_enabled": True,
        "shadow_offset_x": 2,
        "shadow_offset_y": 2,
        "background_enabled": False,
        "background_color": "#000000",
        "background_opacity": 0.0,
        "position": "bottom",
        "margin_v": 60,
        "margin_h": 60,
        "max_chars_per_line": 42,
        "max_lines": 2,
    },
}
BUILTIN_DEFAULT = "word_by_word"

# ASS numpad alignment for position names.
ALIGNMENT = {"bottom": 2, "top": 8, "middle": 5, "center": 5}


def _ms(started: float) -> float:
    """Elapsed milliseconds."""
    return round((time.perf_counter() - started) * 1000.0, 3)


class SubtitleEngine(BaseModule):
    """Generate SRT files and FFmpeg subtitle filters."""

    def __init__(self, container: ServiceContainer) -> None:
        """Initialize engine and load subtitle style configuration."""
        super().__init__(container, MODULE_NAME)
        self._styles: Dict[str, Dict[str, Any]] = dict(BUILTIN_STYLES)
        self._default = BUILTIN_DEFAULT
        self._load_config()

    # ------------------------------------------------------------------
    # SRT generation (File 07 MODULE 07)
    # ------------------------------------------------------------------
    def format_srt_time(self, milliseconds: float) -> str:
        """Convert milliseconds to SRT time format 'HH:MM:SS,mmm'."""
        ms_total = max(0, int(round(float(milliseconds))))
        hours = ms_total // 3600000
        minutes = (ms_total % 3600000) // 60000
        seconds = (ms_total % 60000) // 1000
        ms = ms_total % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"

    def generate_srt_from_word_timestamps(
        self, project_id: str, settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create an SRT file from the project's word-level timestamps.

        Groups words into subtitle blocks (sentence boundaries, character
        and line budget), pads each block's end, validates no overlaps,
        and writes ``subtitles/generated_<project_id>.srt``.
        """
        started = time.perf_counter()
        if not self._enabled:
            return self.make_response(False, error="subtitle_engine is disabled")

        opts = settings or {}
        max_chars = int(self._as_float(opts.get("max_chars_per_line"), 42))
        max_lines = int(self._as_float(opts.get("max_lines"), 2))
        end_padding = int(self._as_float(opts.get("end_padding_ms"), 100))
        max_chars, max_lines = max(8, max_chars), max(1, max_lines)

        words = self._load_word_timestamps(str(project_id))
        if not words:
            return self.make_response(
                False,
                error=f"No word timestamps found for project '{project_id}'",
                duration_ms=_ms(started),
            )

        blocks, warnings = self._group_words(words, max_chars, max_lines)
        # File 07: block end = last word end_time_ms + 100ms gap — padded
        # BEFORE overlap validation so padding can never create overlaps.
        for block in blocks:
            block["end_ms"] += end_padding
        warnings += self._ensure_no_overlaps(blocks, MIN_BLOCK_MS)

        srt_dir = self._subtitles_folder()
        srt_dir.mkdir(parents=True, exist_ok=True)
        safe_project = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in str(project_id)
        )
        srt_path = srt_dir / f"generated_{safe_project}.srt"
        entries = []
        for index, block in enumerate(blocks, start=1):
            entries.append(
                f"{index}\n{self.format_srt_time(block['start_ms'])} --> "
                f"{self.format_srt_time(block['end_ms'])}\n{block['text']}\n"
            )
        srt_path.write_text("\n".join(entries), encoding="utf-8")
        self.log.info("SRT written: %s (%d blocks)", srt_path, len(blocks))
        return self.make_response(
            True,
            {
                "srt_path": str(srt_path),
                "count": len(blocks),
                "total_words": len(words),
                "end_padding_ms": end_padding,
            },
            warnings=warnings,
            duration_ms=_ms(started),
        )

    # ------------------------------------------------------------------
    # Style -> FFmpeg force_style (File 07 MODULE 07)
    # ------------------------------------------------------------------
    def get_style_ffmpeg_params(
        self, style_name: str, subtitle_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Convert a style preset (+overrides) to an ASS force_style string."""
        started = time.perf_counter()
        warnings: List[str] = []
        key = str(style_name or "").strip()
        style = self._styles.get(key)
        if style is None:
            warnings.append(
                f"Unknown subtitle style '{style_name}', using '{self._default}'"
            )
            key = self._default
            style = self._styles[self._default]
        style = dict(style)
        for k, v in (subtitle_settings or {}).items():
            if k in style:
                style[k] = v
            else:
                warnings.append(f"Ignoring unknown style override '{k}'")

        position = str(style.get("position") or "bottom").lower()
        params = {
            "FontName": str(style.get("font_family") or "Sans"),
            "FontSize": int(self._as_float(style.get("font_size"), 48)),
            "PrimaryColour": self._ass_color(str(style.get("font_color") or "#FFFFFF")),
            "OutlineColour": self._ass_color(
                str(style.get("outline_color") or "#000000")
            ),
            "BackColour": self._ass_back_color(
                str(style.get("background_color") or "#000000"),
                self._as_float(style.get("background_opacity"), 0.0),
            ),
            "Outline": int(self._as_float(style.get("outline_size"), 2)),
            "Shadow": (
                max(
                    int(self._as_float(style.get("shadow_offset_x"), 0)),
                    int(self._as_float(style.get("shadow_offset_y"), 0)),
                )
                if style.get("shadow_enabled")
                else 0
            ),
            "BorderStyle": 4 if style.get("background_enabled") else 1,
            "Alignment": ALIGNMENT.get(position, 2),
            "MarginV": int(self._as_float(style.get("margin_v"), 60)),
            "MarginL": int(self._as_float(style.get("margin_h"), 60)),
            "MarginR": int(self._as_float(style.get("margin_h"), 60)),
            # Styles are authored for a 1080p canvas. Without explicit
            # PlayRes, libass renders SRT on FFmpeg's default 384x288 ASS
            # canvas and upscales (~3.75x at 1080p) -> giant, re-wrapped
            # subtitles covering the frame.
            "PlayResX": 1920,
            "PlayResY": 1080,
        }
        force_style = ",".join(f"{k}={v}" for k, v in params.items())
        return self.make_response(
            True,
            {"force_style": force_style, "style_name": key, "params": params},
            warnings=warnings,
            duration_ms=_ms(started),
        )

    def burn_subtitles(
        self,
        video_path: str | Path,
        subtitle_path: str | Path,
        style: str,
        output_path: str | Path,
    ) -> Dict[str, Any]:
        """Burn an SRT into video with FFmpeg subtitles filter + force_style."""
        started = time.perf_counter()
        if not self._enabled:
            return self.make_response(False, error="subtitle_engine is disabled")
        if not Path(video_path).exists():
            return self.make_response(
                False, error=f"Video file not found: {video_path}"
            )
        if not Path(subtitle_path).exists():
            return self.make_response(
                False, error=f"Subtitle file not found: {subtitle_path}"
            )
        ffmpeg = self.hardware.find_ffmpeg() if self.hardware else None
        if not ffmpeg:
            return self.make_response(
                False, error="FFmpeg not available — cannot burn subtitles"
            )

        params = self.get_style_ffmpeg_params(style)
        force_style = params["data"]["force_style"]
        srt_filter_path = self._escape_filter_path(str(subtitle_path))
        vf = f"subtitles='{srt_filter_path}'"
        original_size = self._probe_video_size(video_path)
        if original_size:
            # Tell libass the true frame size so PlayRes 1920x1080 styles
            # map 1:1 pixels at 1080p and scale cleanly at other presets.
            vf += f":original_size={original_size}"
        vf += f":force_style='{force_style}'"
        command = [
            str(ffmpeg),
            "-y",
            "-i",
            str(video_path),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-c:a",
            "copy",
            str(output_path),
        ]
        run = self._run_ffmpeg(command, started)
        if not run["success"]:
            return run
        return self.make_response(
            True,
            {
                "output_path": str(output_path),
                "style_name": params["data"]["style_name"],
                "force_style": force_style,
            },
            warnings=list(params.get("warnings") or []),
            duration_ms=_ms(started),
        )

    def _probe_video_size(self, video_path: str | Path) -> Optional[str]:
        """Return 'WxH' of the first video stream, or None when unprobed."""
        ffprobe = self.hardware.find_ffprobe() if self.hardware else None
        if not ffprobe:
            return None
        try:
            proc = subprocess.run(
                [
                    str(ffprobe),
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "csv=p=0",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        parts = (proc.stdout or "").strip().split(",")
        if len(parts) >= 2 and all(p.strip().isdigit() for p in parts[:2]):
            return f"{parts[0].strip()}x{parts[1].strip()}"
        return None

    # ------------------------------------------------------------------
    # Animated styles (File 07 MODULE 07)
    # ------------------------------------------------------------------
    def build_word_highlight_filters(
        self, words: List[Dict[str, Any]], style_name: str
    ) -> Dict[str, Any]:
        """Build drawtext entries for a word-by-word highlight block.

        Each word gets a secondary-color entry for its full block span and a
        highlight-color entry for its own spoken window (File 07 algorithm).
        """
        started = time.perf_counter()
        style = self._styles.get(style_name) or self._styles[self._default]
        if not words:
            return self.make_response(
                False, error="No words supplied", duration_ms=_ms(started)
            )
        block_start = self._as_float(words[0].get("start_time_ms"), 0.0) / 1000.0
        block_end = self._as_float(words[-1].get("end_time_ms"), 0.0) / 1000.0

        filters: List[str] = []
        for word in words:
            text = self._escape_drawtext(str(word.get("word_text") or ""))
            if not text:
                continue
            ws = self._as_float(word.get("start_time_ms"), 0.0) / 1000.0
            we = self._as_float(word.get("end_time_ms"), 0.0) / 1000.0
            filters.append(
                self._drawtext(
                    text,
                    str(style.get("secondary_color") or "#AAAAAA"),
                    style,
                    block_start,
                    block_end,
                )
            )
            filters.append(
                self._drawtext(
                    text, str(style.get("highlight_color") or "#FFD700"), style, ws, we
                )
            )
        return self.make_response(
            True,
            {
                "filters": filters,
                "count": len(filters),
                "filter_complex": ",".join(filters),
            },
            duration_ms=_ms(started),
        )

    def apply_word_by_word_style(
        self,
        video_path: str | Path,
        project_id: str,
        style: str,
        output_path: str | Path,
    ) -> Dict[str, Any]:
        """Render animated word-by-word subtitles into video (one FFmpeg pass)."""
        started = time.perf_counter()
        words = self._load_word_timestamps(str(project_id))
        ready = self._preflight(video_path, words, started)
        if ready is not None:
            return ready
        style_key = style if style in self._styles else self._default
        blocks, _ = self._group_words_with_words(words, *self._style_limits(style_key))
        all_filters: List[str] = []
        for block in blocks:
            built = self.build_word_highlight_filters(block["words"], style_key)
            all_filters.extend(built["data"]["filters"])
        return self._run_drawtext_pass(
            all_filters, video_path, output_path, style_key, started
        )

    def apply_typewriter_style(
        self,
        video_path: str | Path,
        project_id: str,
        style: str,
        output_path: str | Path,
    ) -> Dict[str, Any]:
        """Render a typewriter (character-by-character) subtitle reveal."""
        started = time.perf_counter()
        words = self._load_word_timestamps(str(project_id))
        ready = self._preflight(video_path, words, started)
        if ready is not None:
            return ready
        style_key = style if style in self._styles else self._default
        style_cfg = self._styles[style_key]

        filters: List[str] = []
        for word in words:
            text = str(word.get("word_text") or "")
            ws = self._as_float(word.get("start_time_ms"), 0.0) / 1000.0
            we = self._as_float(word.get("end_time_ms"), 0.0) / 1000.0
            chars = list(text)
            if not chars:
                continue
            step = max(0.001, (we - ws) / len(chars))
            for i in range(len(chars)):
                shown = self._escape_drawtext("".join(chars[: i + 1]))
                char_on = ws + i * step
                filters.append(
                    self._drawtext(
                        shown,
                        str(style_cfg.get("font_color") or "#FFFFFF"),
                        style_cfg,
                        char_on,
                        we,
                    )
                )
        return self._run_drawtext_pass(
            filters, video_path, output_path, style_key, started
        )

    def get_available_styles(self) -> Dict[str, Any]:
        """Return the subtitle style catalog for UI display / validation."""
        started = time.perf_counter()
        catalog = [
            {
                "id": sid,
                "name": s.get("name", sid),
                "description": s.get("description", ""),
                "animation_type": s.get("animation_type", ""),
                "font_family": s.get("font_family", ""),
                "font_size": s.get("font_size"),
            }
            for sid, s in self._styles.items()
        ]
        return self.make_response(
            True,
            {"styles": catalog, "count": len(catalog), "default_style": self._default},
            duration_ms=_ms(started),
        )

    def is_optional_module(self) -> bool:
        """Return True — subtitle_engine may be disabled safely."""
        return True

    # ------------------------------------------------------------------
    # Grouping helpers
    # ------------------------------------------------------------------
    def _load_word_timestamps(self, project_id: str) -> List[Dict[str, Any]]:
        """Load word timestamps from the database ordered by start time."""
        # Container's "database" is a DatabaseService facade; raw SQL lives
        # on the wrapped SQLiteDatabase (same pattern as domain helpers).
        db = getattr(self.db, "db", self.db)
        try:
            rows = db.fetch_all(
                "SELECT word_text, start_time_ms, end_time_ms, dialogue_line_id "
                "FROM word_timestamps WHERE project_id = ? "
                "ORDER BY start_time_ms, word_index",
                (project_id,),
            )
        except Exception as exc:  # noqa: BLE001
            self.log.error("Failed loading word timestamps: %s", exc)
            return []
        return [dict(r) for r in rows]

    @staticmethod
    def _group_words(
        words: List[Dict[str, Any]], max_chars: int, max_lines: int
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Group words into subtitle blocks (sentence/character budget)."""
        blocks: List[Dict[str, Any]] = []
        warnings: List[str] = []
        current: List[Dict[str, Any]] = []

        def close_block() -> None:
            if not current:
                return
            text = " ".join(w["word_text"] for w in current)
            lines = SubtitleEngine._wrap_lines(text, max_chars, max_lines)
            if len(lines) > max_lines:
                warnings.append("Block exceeded max lines after wrap; split further")
            blocks.append(
                {
                    "start_ms": int(current[0]["start_time_ms"]),
                    "end_ms": int(current[-1]["end_time_ms"]),
                    "text": "\n".join(lines[:max_lines]),
                }
            )

        budget = max_chars * max_lines
        for word in words:
            text = str(word.get("word_text") or "")
            candidate_len = (
                len(" ".join(w["word_text"] for w in current))
                + len(text)
                + (1 if current else 0)
            )
            if current and (candidate_len > budget):
                close_block()
                current = []
            current.append(word)
            if text.rstrip().endswith(tuple(SENTENCE_END)):
                close_block()
                current = []
        close_block()
        return blocks, warnings

    def _group_words_with_words(
        self, words: List[Dict[str, Any]], max_chars: int, max_lines: int
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Like _group_words but keeps the word dicts per block."""
        blocks: List[Dict[str, Any]] = []
        warnings: List[str] = []
        current: List[Dict[str, Any]] = []
        budget = max_chars * max_lines

        def close() -> None:
            if current:
                blocks.append({"words": list(current)})

        for word in words:
            text = str(word.get("word_text") or "")
            candidate_len = (
                len(" ".join(w["word_text"] for w in current))
                + len(text)
                + (1 if current else 0)
            )
            if current and candidate_len > budget:
                close()
                current = []
            current.append(word)
            if text.rstrip().endswith(tuple(SENTENCE_END)):
                close()
                current = []
        close()
        return blocks, warnings

    @staticmethod
    def _wrap_lines(text: str, max_chars: int, max_lines: int) -> List[str]:
        """Greedy word wrap into at most max_lines lines."""
        words = text.split()
        lines: List[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) <= max_chars or not current:
                current = candidate if current else word
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def _ensure_no_overlaps(
        self, blocks: List[Dict[str, Any]], min_ms: int
    ) -> List[str]:
        """Clamp overlapping/too-short blocks, returning fix warnings."""
        warnings: List[str] = []
        previous_end: Optional[int] = None
        for index, block in enumerate(blocks):
            if previous_end is not None and block["start_ms"] < previous_end:
                warnings.append(f"Overlap at block {index + 1}: previous end clamped")
                prev = blocks[index - 1]
                prev["end_ms"] = max(
                    prev["start_ms"] + min_ms, block["start_ms"] - OVERLAP_GAP_MS
                )
                previous_end = prev["end_ms"]
            if block["end_ms"] - block["start_ms"] < min_ms:
                block["end_ms"] = block["start_ms"] + min_ms
                warnings.append(f"Block {index + 1} extended to minimum {min_ms}ms")
            previous_end = max(previous_end or 0, block["end_ms"])
        return warnings

    # ------------------------------------------------------------------
    # drawtext / FFmpeg helpers
    # ------------------------------------------------------------------
    def _drawtext(
        self, text: str, color: str, style: Dict[str, Any], start_s: float, end_s: float
    ) -> str:
        """Single drawtext filter string for a text window."""
        size = int(self._as_float(style.get("font_size"), 48))
        margin_v = int(self._as_float(style.get("margin_v"), 60))
        outline = str(style.get("outline_color") or "#000000")
        borderw = int(self._as_float(style.get("outline_size"), 2))
        parts = [
            f"drawtext=text='{text}'",
            f"fontsize={size}",
            f"fontcolor={color}",
            "x=(w-text_w)/2",
            f"y=h-{margin_v}",
            "enable='between(t,{:.3f},{:.3f})'".format(
                max(0.0, start_s), max(0.0, end_s)
            ),
        ]
        font = self._resolve_font(str(style.get("font_family") or ""))
        if font:
            parts.insert(1, f"fontfile='{font}'")
        if borderw > 0:
            parts.append(f"borderw={borderw}:bordercolor={outline}")
        if (
            style.get("background_enabled")
            and self._as_float(style.get("background_opacity"), 0) > 0
        ):
            opacity = self._as_float(style.get("background_opacity"), 0.5)
            parts.append(
                f"box=1:boxcolor={style.get('background_color', '#000000')}@{opacity:.2f}"
            )
        return ":".join(parts)

    def _run_drawtext_pass(
        self,
        filters: List[str],
        video_path: str | Path,
        output_path: str | Path,
        style_key: str,
        started: float,
    ) -> Dict[str, Any]:
        """Execute the one-pass drawtext chain via FFmpeg."""
        if not filters:
            return self.make_response(
                False, error="No drawtext filters built", duration_ms=_ms(started)
            )
        ffmpeg = self.hardware.find_ffmpeg() if self.hardware else None
        if not ffmpeg:
            return self.make_response(
                False, error="FFmpeg not available — cannot render subtitles"
            )
        command = [
            str(ffmpeg),
            "-y",
            "-i",
            str(video_path),
            "-vf",
            ",".join(filters),
            "-c:v",
            "libx264",
            "-c:a",
            "copy",
            str(output_path),
        ]
        run = self._run_ffmpeg(command, started, log_preview=False)
        if not run["success"]:
            return run
        return self.make_response(
            True,
            {
                "output_path": str(output_path),
                "style_name": style_key,
                "drawtext_count": len(filters),
            },
            duration_ms=_ms(started),
        )

    def _preflight(
        self, video_path: str | Path, words: List[Dict[str, Any]], started: float
    ) -> Optional[Dict[str, Any]]:
        """Shared guard for animated apply methods; None means OK."""
        if not self._enabled:
            return self.make_response(False, error="subtitle_engine is disabled")
        if not Path(video_path).exists():
            return self.make_response(
                False, error=f"Video file not found: {video_path}"
            )
        if not words:
            return self.make_response(
                False, error="No word timestamps for subtitle animation"
            )
        if len(words) > MAX_CHAIN_WORDS:
            return self.make_response(
                False,
                error=f"{len(words)} words exceed single-pass limit {MAX_CHAIN_WORDS}; "
                "export via 30-second segments (DEBT-B11a)",
                data={"words": len(words), "limit": MAX_CHAIN_WORDS},
                duration_ms=_ms(started),
            )
        return None

    def _run_ffmpeg(
        self, command: List[str], started: float, log_preview: bool = True
    ) -> Dict[str, Any]:
        """Run an FFmpeg command with logging and graceful errors (Rule 4)."""
        preview = " ".join(command)
        self.log.info(
            "FFmpeg command: %s",
            preview if log_preview else preview[:400] + " ...[drawtext chain]",
        )
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.log.error("FFmpeg run failed: %s", exc)
            return self.make_response(False, error=str(exc), duration_ms=_ms(started))
        if result.returncode != 0:
            self.log.error("FFmpeg error: %s", (result.stderr or "unknown")[-500:])
            return self.make_response(
                False,
                error=(result.stderr or "ffmpeg failed")[-300:],
                duration_ms=_ms(started),
            )
        return self.make_response(True, {"command": command}, duration_ms=_ms(started))

    # ------------------------------------------------------------------
    # Config & misc helpers
    # ------------------------------------------------------------------
    def _load_config(self) -> None:
        """Load subtitle_style_presets.json via ConfigService with file fallback."""
        data = self._safe_get_config("subtitle_style_presets")
        if not data:
            path = Path("config/subtitle_style_presets.json")
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    self.log.error(
                        "subtitle_style_presets.json corrupt (%s); using built-ins", exc
                    )
                    return
        if not data:
            self.log.warning(
                "subtitle_style_presets.json missing; using built-in styles"
            )
            return
        styles = {
            str(s.get("id")): s for s in (data.get("styles") or []) if s.get("id")
        }
        if styles:
            self._styles = styles
        if data.get("default_style"):
            self._default = str(data["default_style"])

    def _safe_get_config(self, name: str) -> Dict[str, Any]:
        """ConfigService lookup that never raises."""
        try:
            data = self.config.get_config(name)
        except Exception:  # noqa: BLE001
            return {}
        return data if isinstance(data, dict) else {}

    def _subtitles_folder(self) -> Path:
        """Subtitle output folder from settings with project default (Rule 3)."""
        try:
            configured = self.config.get("subtitles_folder")
        except Exception:  # noqa: BLE001
            configured = None
        return Path(str(configured)) if configured else Path("subtitles")

    def _resolve_font(self, family: str) -> Optional[str]:
        """Resolve a font family to an existing TTF path under assets/fonts."""
        if not family:
            return None
        fonts_dir = Path("assets") / "fonts"
        for candidate in (fonts_dir / f"{family}.ttf", fonts_dir / f"{family}.otf"):
            if candidate.exists():
                return str(candidate)
        return None

    def _style_limits(self, style_key: str) -> Tuple[int, int]:
        """(max_chars_per_line, max_lines) for a style."""
        style = self._styles.get(style_key) or {}
        return (
            int(self._as_float(style.get("max_chars_per_line"), 42)),
            int(self._as_float(style.get("max_lines"), 2)),
        )

    @staticmethod
    def _ass_color(hex_color: str) -> str:
        """'#RRGGBB' -> ASS '&H00BBGGRR' (opaque)."""
        h = hex_color.lstrip("#").upper()
        if len(h) != 6:
            h = "FFFFFF"
        return f"&H00{h[4:6]}{h[2:4]}{h[0:2]}"

    @staticmethod
    def _ass_back_color(hex_color: str, opacity: float) -> str:
        """Background colour with opacity mapped to ASS alpha (&HAA000000)."""
        alpha = max(0, min(255, 255 - int(opacity * 255)))
        h = hex_color.lstrip("#").upper()
        if len(h) != 6:
            h = "000000"
        return f"&H{alpha:02X}{h[4:6]}{h[2:4]}{h[0:2]}"

    @staticmethod
    def _escape_drawtext(text: str) -> str:
        """Escape drawtext-sensitive characters in literal text."""
        return (
            text.replace("\\", "\\\\")
            .replace("'", "'\\\\\\''")
            .replace(":", "\\:")
            .replace("%", "%%")
        )

    @staticmethod
    def _escape_filter_path(path: str) -> str:
        """Escape a file path for use inside an FFmpeg filter argument."""
        return path.replace("\\", "/").replace(":", "\\:")

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        """Best-effort float conversion with fallback."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)
