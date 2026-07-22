"""Unit tests for modules.subtitle_engine.SubtitleEngine.

SRT generation uses real DB rows (full FK chain seeded). FFmpeg execution
is covered with the shared cross-platform fake-ffmpeg test double from
tests/conftest.py (bash on POSIX, Python + subprocess shim on Windows).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.service_container import ServiceContainer
from modules.subtitle_engine import SubtitleEngine

STYLE_COUNT = 8  # File 11 subtitle_style_presets.json
PROJECT = "proj-sub-1"

WORDS = [
    ("In", 0, 400),
    ("the", 400, 700),
    ("year", 700, 1100),
    ("1347,", 1100, 1500),
    ("twelve", 1500, 1900),
    ("ships", 1900, 2300),
    ("arrived.", 2300, 2700),
    ("The", 2800, 3100),
    ("city", 3100, 3500),
    ("fell", 3500, 3900),
    ("silent.", 3900, 4400),
]


@pytest.fixture
def container(project_root: Path, tmp_path: Path) -> ServiceContainer:
    """Isolated container with the real project config folder."""
    return ServiceContainer.create_production_container(
        app_config={
            "database_path": str(tmp_path / "autopilot.db"),
            "schema_path": str(project_root / "database" / "schema.sql"),
            "config_folder": str(project_root / "config"),
            "cache_folder": str(tmp_path / "cache"),
            "log_folder": str(tmp_path / "logs"),
            "ffmpeg_path": "ffmpeg",
        },
        project_root=project_root,
    )


@pytest.fixture
def engine(container: ServiceContainer) -> SubtitleEngine:
    """Subtitle engine instance."""
    return SubtitleEngine(container)


@pytest.fixture
def seeded(engine: SubtitleEngine) -> SubtitleEngine:
    """Engine whose DB contains the WORDS list under PROJECT."""
    _seed_words(engine, PROJECT, WORDS)
    return engine


def _seed_words(
    engine: SubtitleEngine,
    project_id: str,
    words: list[tuple[str, int, int]],
) -> None:
    """Seed projects -> scenes -> dialogue_lines -> word_timestamps chain."""
    now = "2026-07-16 00:00:00"
    engine.db.db.execute(
        "INSERT INTO projects (id, title, created_at, updated_at, project_folder_path)"
        " VALUES (?, ?, ?, ?, ?)",
        (project_id, "Sub Test", now, now, "/tmp/x"),
    )
    engine.db.db.execute(
        "INSERT INTO scenes (id, project_id, scene_number, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (f"sc-{project_id}", project_id, 1, now, now),
    )
    engine.db.db.execute(
        "INSERT INTO dialogue_lines (id, project_id, scene_id, line_number,"
        " character_name, emotion, text_content, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            f"dl-{project_id}",
            project_id,
            f"sc-{project_id}",
            1,
            "NARRATOR",
            "dramatic",
            "x",
            now,
            now,
        ),
    )
    for i, (text, start, end) in enumerate(words):
        engine.db.db.execute(
            "INSERT INTO word_timestamps (id, project_id, dialogue_line_id, word_index,"
            " word_text, start_time_ms, end_time_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"w-{project_id}-{i}",
                project_id,
                f"dl-{project_id}",
                i,
                text,
                start,
                end,
            ),
        )


def _container_with_ffmpeg(
    project_root: Path, tmp_path: Path, ffmpeg: Path
) -> ServiceContainer:
    return ServiceContainer.create_production_container(
        app_config={
            "database_path": str(tmp_path / "autopilot.db"),
            "schema_path": str(project_root / "database" / "schema.sql"),
            "config_folder": str(project_root / "config"),
            "cache_folder": str(tmp_path / "cache"),
            "log_folder": str(tmp_path / "logs"),
            "ffmpeg_path": str(ffmpeg),
        },
        project_root=project_root,
    )


class TestSrtTimeAndGeneration:
    """format_srt_time and SRT writing rules."""

    def test_format_srt_time(self, engine: SubtitleEngine) -> None:
        assert engine.format_srt_time(0) == "00:00:00,000"
        assert engine.format_srt_time(83456) == "00:01:23,456"
        assert engine.format_srt_time(3723004) == "01:02:03,004"
        assert engine.format_srt_time(-50) == "00:00:00,000"

    def test_generate_srt_sentence_blocks(
        self, seeded: SubtitleEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(seeded, "_subtitles_folder", lambda: tmp_path / "subs")
        result = seeded.generate_srt_from_word_timestamps(PROJECT, {})
        assert result["success"] is True
        assert result["data"]["count"] == 2  # two sentences -> two blocks
        content = Path(result["data"]["srt_path"]).read_text(encoding="utf-8")
        assert (
            "1\n00:00:00,000 --> 00:00:02,800\nIn the year 1347, twelve ships arrived.\n"
            in content
        )
        assert "2\n00:00:02,800 --> 00:00:04,500\nThe city fell silent.\n" in content

    def test_generate_srt_character_budget_split(
        self, engine: SubtitleEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        long_words = [(f"wordnumber{i:02d}", i * 600, i * 600 + 500) for i in range(10)]
        _seed_words(engine, "proj-long", long_words)
        monkeypatch.setattr(engine, "_subtitles_folder", lambda: tmp_path / "subs")
        result = engine.generate_srt_from_word_timestamps(
            "proj-long", {"max_chars_per_line": 42, "max_lines": 2}
        )
        assert result["success"] is True
        assert result["data"]["count"] >= 2  # 120 chars exceeds 84-char budget
        content = Path(result["data"]["srt_path"]).read_text(encoding="utf-8")
        for line in content.splitlines():
            if line and not line[0].isdigit() and "-->" not in line:
                assert len(line) <= 42

    def test_generate_srt_no_words(self, engine: SubtitleEngine) -> None:
        result = engine.generate_srt_from_word_timestamps("ghost-project", {})
        assert result["success"] is False
        assert "No word timestamps" in (result["error"] or "")

    def test_overlap_and_min_duration_fixed(
        self, engine: SubtitleEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Block 1 closes at "One." (sentence) -> padded end 400; block 2 starts
        # at 250 -> guaranteed overlap. "Quick" is shorter than MIN_BLOCK_MS.
        tight = [("One.", 0, 300), ("Two.", 250, 400), ("Quick", 600, 650)]
        _seed_words(engine, "proj-tight", tight)
        monkeypatch.setattr(engine, "_subtitles_folder", lambda: tmp_path / "subs")
        result = engine.generate_srt_from_word_timestamps("proj-tight", {})
        assert result["success"] is True
        assert any("Overlap" in w or "minimum" in w for w in result["warnings"])
        ranges = re.findall(
            r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})",
            Path(result["data"]["srt_path"]).read_text(encoding="utf-8"),
        )
        for s, e in ranges:
            assert s < e  # every block is forward-moving


class TestStyleParams:
    """force_style string conversion."""

    def test_word_by_word_params(self, engine: SubtitleEngine) -> None:
        data = engine.get_style_ffmpeg_params("word_by_word")["data"]
        fs = data["force_style"]
        assert "FontName=Montserrat-Bold" in fs
        assert "FontSize=48" in fs  # retuned: 52 -> 48 (approved)
        assert "PrimaryColour=&H00FFFFFF" in fs
        assert "Outline=3" in fs
        assert "BorderStyle=4" in fs  # background box enabled
        assert "Alignment=2" in fs
        assert "MarginV=90" in fs  # retuned: 80 -> 90 (approved)
        assert "&HA6000000" in fs  # retuned: 0.35 opacity -> softer box (approved)

    def test_documentary_classic_outline_only(self, engine: SubtitleEngine) -> None:
        fs = engine.get_style_ffmpeg_params("documentary_classic")["data"][
            "force_style"
        ]
        assert "BorderStyle=1" in fs  # background disabled -> outline only
        assert "Outline=4" in fs

    def test_abgr_color_conversion(self, engine: SubtitleEngine) -> None:
        data = engine.get_style_ffmpeg_params(
            "word_by_word", {"font_color": "#FFD700"}
        )["data"]
        assert "PrimaryColour=&H0000D7FF" in data["force_style"]

    def test_unknown_style_and_override(self, engine: SubtitleEngine) -> None:
        result = engine.get_style_ffmpeg_params(
            "bogus", {"bogus_key": 1, "font_size": 61}
        )
        assert any("Unknown subtitle style" in w for w in result["warnings"])
        assert any("Ignoring unknown" in w for w in result["warnings"])
        assert "FontSize=61" in result["data"]["force_style"]


class TestBurnAndAnimation:
    """FFmpeg burn-in and animated styles via fake ffmpeg."""

    def test_burn_missing_files(self, engine: SubtitleEngine, tmp_path: Path) -> None:
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        missing_srt = engine.burn_subtitles(
            video, tmp_path / "no.srt", "word_by_word", tmp_path / "o.mp4"
        )
        assert missing_srt["success"] is False and "Subtitle file not found" in (
            missing_srt["error"] or ""
        )
        missing_video = engine.burn_subtitles(
            tmp_path / "no.mp4", video, "word_by_word", tmp_path / "o.mp4"
        )
        assert missing_video["success"] is False and "Video file not found" in (
            missing_video["error"] or ""
        )

    def test_burn_without_ffmpeg(
        self, engine: SubtitleEngine, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # D4b: simulate absence explicitly — independent of whether a
        # real ffmpeg is bundled/installed on the host.
        monkeypatch.setattr(engine.hardware, "find_ffmpeg", lambda: None)
        (tmp_path / "v.mp4").write_bytes(b"x")
        (tmp_path / "s.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8"
        )
        result = engine.burn_subtitles(
            tmp_path / "v.mp4", tmp_path / "s.srt", "word_by_word", tmp_path / "o.mp4"
        )
        assert result["success"] is False and "FFmpeg not available" in (
            result["error"] or ""
        )

    def test_burn_with_fake_ffmpeg(
        self,
        project_root: Path,
        seeded: SubtitleEngine,
        tmp_path: Path,
        fake_ffmpeg_factory,
    ) -> None:
        fake = fake_ffmpeg_factory(tmp_path, tmp_path / "ffmpeg_argv.txt")
        engine = SubtitleEngine(_container_with_ffmpeg(project_root, tmp_path, fake))
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        srt = tmp_path / "s.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")
        result = engine.burn_subtitles(
            video, srt, "documentary_classic", tmp_path / "o.mp4"
        )
        assert result["success"] is True
        argv = (tmp_path / "ffmpeg_argv.txt").read_text(encoding="utf-8")
        assert "subtitles=" in argv and "force_style=" in argv
        assert "BorderStyle=1" in argv
        assert "-c:a copy" in argv
        assert (tmp_path / "o.mp4").exists()

    def test_word_highlight_filter_structure(self, engine: SubtitleEngine) -> None:
        words = [
            {"word_text": "Dark", "start_time_ms": 0, "end_time_ms": 500},
            {"word_text": "secrets", "start_time_ms": 500, "end_time_ms": 1100},
            {"word_text": "emerged.", "start_time_ms": 1100, "end_time_ms": 1900},
        ]
        data = engine.build_word_highlight_filters(words, "word_by_word")["data"]
        assert data["count"] == 6  # 2 drawtext entries per word
        chain = data["filter_complex"]
        assert "enable='between(t,0.500,1.100)'" in chain
        assert "fontcolor=#FFD700" in chain  # highlight color
        assert "fontcolor=#AAAAAA" in chain  # secondary color
        assert "x=(w-text_w)/2" in chain and "y=h-90" in chain  # margin_v retuned 80 -> 90 (approved)

    def test_drawtext_escaping(self, engine: SubtitleEngine) -> None:
        words = [{"word_text": "don't: 100%", "start_time_ms": 0, "end_time_ms": 500}]
        chain = engine.build_word_highlight_filters(words, "word_by_word")["data"][
            "filter_complex"
        ]
        assert "\\:" in chain
        assert "%%" in chain
        assert "don" in chain  # text survives with escaped quote

    def test_apply_word_by_word(
        self,
        project_root: Path,
        seeded: SubtitleEngine,
        tmp_path: Path,
        fake_ffmpeg_factory,
    ) -> None:
        # `seeded` already wrote PROJECT rows into this tmp_path's database
        # file; the new engine shares that file, so no re-seed is needed.
        fake = fake_ffmpeg_factory(tmp_path, tmp_path / "ffmpeg_argv.txt")
        engine = SubtitleEngine(_container_with_ffmpeg(project_root, tmp_path, fake))
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        result = engine.apply_word_by_word_style(
            video, PROJECT, "word_by_word", tmp_path / "o.mp4"
        )
        assert result["success"] is True
        assert result["data"]["drawtext_count"] == 2 * len(WORDS)
        assert "drawtext=" in (tmp_path / "ffmpeg_argv.txt").read_text(encoding="utf-8")

    def test_apply_typewriter(
        self,
        project_root: Path,
        engine: SubtitleEngine,
        tmp_path: Path,
        fake_ffmpeg_factory,
    ) -> None:
        fake = fake_ffmpeg_factory(tmp_path, tmp_path / "ffmpeg_argv.txt")
        engine = SubtitleEngine(_container_with_ffmpeg(project_root, tmp_path, fake))
        _seed_words(engine, "proj-tw", [("ab", 0, 200), ("cd", 400, 600)])
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        result = engine.apply_typewriter_style(
            video, "proj-tw", "typewriter", tmp_path / "o.mp4"
        )
        assert result["success"] is True
        assert result["data"]["drawtext_count"] == 4  # 2 chars x 2 words
        argv = (tmp_path / "ffmpeg_argv.txt").read_text(encoding="utf-8")
        assert "between(t,0.000,0.200)" in argv and "between(t,0.100,0.200)" in argv

    def test_chain_limit_guard(
        self, seeded: SubtitleEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("modules.subtitle_engine.MAX_CHAIN_WORDS", 3)
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        result = seeded.apply_word_by_word_style(
            video, PROJECT, "word_by_word", tmp_path / "o.mp4"
        )
        assert result["success"] is False
        assert "DEBT-B11a" in (result["error"] or "")


class TestCatalogAndDisable:
    """Style catalog and module enable flag."""

    def test_available_styles(self, engine: SubtitleEngine) -> None:
        data = engine.get_available_styles()["data"]
        ids = {s["id"] for s in data["styles"]}
        assert ids == {
            "word_by_word",
            "netflix_style",
            "documentary_classic",
            "typewriter",
            "impact_style",
            "karaoke_style",
            "glitch_style",
            "high_contrast",
        }
        assert data["count"] == STYLE_COUNT
        assert data["default_style"] == "word_by_word"

    def test_module_can_be_disabled(self, engine: SubtitleEngine) -> None:
        assert engine.is_optional_module() is True
        engine.set_enabled(False)
        result = engine.generate_srt_from_word_timestamps(PROJECT, {})
        assert result["success"] is False and "disabled" in (result["error"] or "")
        engine.set_enabled(True)
        assert engine.get_available_styles()["success"] is True
