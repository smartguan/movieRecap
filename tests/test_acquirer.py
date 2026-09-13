"""
Unit tests for Video Acquirer Agent and Platform Extractors.
"""

from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src.acquirer.agent import VideoAcquirerAgent, slugify_title
from src.acquirer.generic import GenericVideoExtractor
from src.acquirer.models import AcquisitionResult, StreamInfo, VideoMetadata
from src.acquirer.yfsp import YfspExtractor
from src.models.project import MediaInfo


def test_slugify_title() -> None:
    assert slugify_title("微风襟袖同卿心") == "微风襟袖同卿心"
    assert slugify_title("Movie: The Great / Adventure?") == "Movie_The_Great_Adventure"
    assert slugify_title("  Spaces   and - dashes  ") == "Spaces_and_-_dashes"
    assert slugify_title("") == "movie"


def test_yfsp_can_handle() -> None:
    extractor = YfspExtractor()
    assert extractor.can_handle("https://www.yfsp.tv/play/zqBWB3mQYiB?id=2XQtVuG1mT3")
    assert extractor.can_handle("https://m.yfsp.tv/play/12345")
    assert extractor.can_handle("https://iyf.tv/play/abcdef")
    assert not extractor.can_handle("https://www.youtube.com/watch?v=12345")
    assert not extractor.can_handle("https://example.com/video.mp4")


def test_yfsp_parse_url_ids() -> None:
    extractor = YfspExtractor()
    vid, ep = extractor._parse_url_ids("https://www.yfsp.tv/play/zqBWB3mQYiB?id=2XQtVuG1mT3")
    assert vid == "zqBWB3mQYiB"
    assert ep == "2XQtVuG1mT3"

    vid2, ep2 = extractor._parse_url_ids("https://www.yfsp.tv/play/single_movie")
    assert vid2 == "single_movie"
    assert ep2 == "single_movie"


def test_yfsp_signing() -> None:
    extractor = YfspExtractor()
    signed = extractor._sign_request(
        "https://m10.yfsp.tv/v3/video/detail",
        {"cinema": 1, "id": "test_id"},
        pub_key="PUB_KEY",
        priv_key="PRIV_KEY",
    )
    assert "pub=PUB_KEY" in signed
    assert "vv=" in signed
    assert "cinema=1" in signed
    assert "id=test_id" in signed


def test_generic_extractor() -> None:
    extractor = GenericVideoExtractor()
    assert extractor.can_handle("https://example.com/videos/movie.mp4")
    assert extractor.can_handle("https://example.com/live/stream.m3u8")
    assert extractor.can_handle("https://cdn.test/video.mkv?token=123")
    assert not extractor.can_handle("https://example.com/watch?v=123")

    meta = extractor.extract_metadata("https://example.com/sample_movie_2026.mp4")
    assert meta.title == "sample movie 2026"

    stream = extractor.resolve_stream("https://example.com/stream.m3u8")
    assert stream.stream_type == "hls"
    assert stream.stream_url == "https://example.com/stream.m3u8"


@patch("src.acquirer.agent.subprocess.run")
@patch("src.acquirer.agent.extract_media_info")
def test_video_acquirer_agent_mocked(mock_extract_info, mock_subproc, tmp_path: Path) -> None:
    # Setup mock media info
    mock_extract_info.return_value = MediaInfo(
        duration_seconds=120.5,
        resolution=(1920, 1080),
        frame_rate=24.0,
        video_codec="h264",
        audio_codec="aac",
        audio_tracks=2,
        has_subtitles=False,
    )

    # Setup mock subprocess
    mock_subproc.return_value = MagicMock(returncode=0, stderr="")

    agent = VideoAcquirerAgent()

    # Mock custom extractor
    mock_extractor = MagicMock(spec=GenericVideoExtractor)
    mock_extractor.can_handle.return_value = True
    mock_extractor.extract_metadata.return_value = VideoMetadata(
        title="Test Movie",
        source_url="https://example.com/test.mp4",
        genre="Drama",
        year="2026",
    )
    mock_extractor.resolve_stream.return_value = StreamInfo(
        stream_url="https://example.com/test.mp4",
        stream_type="mp4",
    )

    agent.extractors = [mock_extractor]

    def side_effect_download(*args, **kwargs):
        # Create a mock video file so validate_incoming and stat() succeed
        incoming_folder = tmp_path / "Test_Movie"
        incoming_folder.mkdir(parents=True, exist_ok=True)
        video_file = incoming_folder / "source.mp4"
        video_file.write_bytes(b"dummy video data")

    agent.download_stream = MagicMock(side_effect=side_effect_download)

    # Run acquire
    result = agent.acquire(
        url="https://example.com/test.mp4",
        incoming_base_dir=tmp_path,
    )

    assert result.success is True
    assert result.movie_title == "Test Movie"
    assert Path(result.metadata_path).exists()
    assert Path(result.source_video_path).exists()

    with open(result.metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["title"] == "Test Movie"
        assert data["duration_seconds"] == 120.5
        assert data["resolution"] == "1920x1080"
