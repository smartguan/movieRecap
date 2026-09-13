"""
Native Video Extractor for 爱壹帆 (yfsp.tv / iYF).
"""

from __future__ import annotations
import hashlib
import json
import logging
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Tuple

from src.acquirer.base import BaseVideoExtractor
from src.acquirer.models import StreamInfo, VideoMetadata

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


class YfspExtractor(BaseVideoExtractor):
    """
    Extractor for yfsp.tv (爱壹帆).
    Handles dynamic signature hashing and direct HLS stream extraction.
    """

    DOMAINS = ["yfsp.tv", "iyf.tv", "m.yfsp.tv", "m.iyf.tv"]

    def __init__(self) -> None:
        self.headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Referer": "https://www.yfsp.tv/",
            "Origin": "https://www.yfsp.tv",
        }
        self.base_api = "https://m10.yfsp.tv/v3"

    def can_handle(self, url: str) -> bool:
        """Check if URL belongs to yfsp / iyf domain."""
        parsed = urllib.parse.urlparse(url)
        return any(d in parsed.netloc for d in self.DOMAINS)

    def _parse_url_ids(self, url: str) -> Tuple[str, str]:
        """Extract video_id and episode_id from play URL."""
        parsed = urllib.parse.urlparse(url)
        # Path format: /play/<video_id>
        path_parts = [p for p in parsed.path.split("/") if p]
        video_id = ""
        if len(path_parts) >= 2 and path_parts[0] == "play":
            video_id = path_parts[1]
        elif len(path_parts) == 1:
            video_id = path_parts[0]

        query = urllib.parse.parse_qs(parsed.query)
        episode_id = query.get("id", [video_id])[0]

        return video_id, episode_id

    def _fetch_page_keys(self, url: str) -> Tuple[str, str]:
        """Scrape injectJson from HTML to get public and private keys."""
        req = urllib.request.Request(url, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as res:
                html = res.read().decode("utf-8")
        except Exception as e:
            logger.warning("Failed to fetch yfsp webpage (%s), using fallback keys", e)
            return (
                "CJSuEJCpDJKvCousDZLVLLDVCZ0mCJerOJWwD3SvD3fXCp0mEcCvDZCwCpWqP3etE68mEcOrELyncpCSiB4QiR8ncHoQ6vYo71mPchAni9YRi9kOiB6S7AzP64qDp0oCJGpD39bOpDaCpTYDZSoDMHZEJDaEMOqPZ1",
                "SuEJJSuEJCpDJKvCousD",
            )

        match = re.search(r"var injectJson\s*=\s*(\{.*?\});", html, re.DOTALL)
        if match:
            try:
                inject = json.loads(match.group(1))
                p_config = inject["config"][0]["pConfig"]
                pub_key = p_config["publicKey"]
                priv_key = p_config["privateKey"][0]
                return pub_key, priv_key
            except Exception:
                pass

        # Fallback default static keypair
        return (
            "CJSuEJCpDJKvCousDZLVLLDVCZ0mCJerOJWwD3SvD3fXCp0mEcCvDZCwCpWqP3etE68mEcOrELyncpCSiB4QiR8ncHoQ6vYo71mPchAni9YRi9kOiB6S7AzP64qDp0oCJGpD39bOpDaCpTYDZSoDMHZEJDaEMOqPZ1",
            "SuEJJSuEJCpDJKvCousD",
        )

    def _sign_request(self, endpoint_url: str, params: Dict[str, Any], pub_key: str, priv_key: str) -> str:
        """Compute cryptographic MD5 signature parameter vv for API call."""
        clean_params = {k: v for k, v in params.items() if v is not None}
        q = "&".join(f"{k}={v}" for k, v in sorted(clean_params.items()))
        raw_str = f"{pub_key}&{q.lower()}&{priv_key}"
        vv = hashlib.md5(raw_str.encode("utf-8")).hexdigest()
        return f"{endpoint_url}?{q}&vv={vv}&pub={pub_key}"

    def extract_metadata(self, url: str) -> VideoMetadata:
        """Extract movie/video metadata from yfsp.tv API."""
        video_id, episode_id = self._parse_url_ids(url)
        pub_key, priv_key = self._fetch_page_keys(url)

        target_id = video_id or episode_id
        signed_url = self._sign_request(
            f"{self.base_api}/video/detail",
            {"cinema": 1, "id": target_id},
            pub_key,
            priv_key,
        )

        req = urllib.request.Request(signed_url, headers=self.headers)
        with urllib.request.urlopen(req, timeout=10) as res:
            resp_data = json.loads(res.read().decode("utf-8"))

        info_list = resp_data.get("data", {}).get("info", [])
        if not info_list or not info_list[0]:
            # Try with episode_id if different
            if episode_id and episode_id != target_id:
                signed_url = self._sign_request(
                    f"{self.base_api}/video/detail",
                    {"cinema": 1, "id": episode_id},
                    pub_key,
                    priv_key,
                )
                req = urllib.request.Request(signed_url, headers=self.headers)
                with urllib.request.urlopen(req, timeout=10) as res:
                    resp_data = json.loads(res.read().decode("utf-8"))
                    info_list = resp_data.get("data", {}).get("info", [])

        if not info_list or not info_list[0]:
            raise ValueError(f"No video metadata found for {url}")

        info = info_list[0]
        title = info.get("title", "Unknown Title")
        synopsis = info.get("contxt", "")
        stars = info.get("stars", [])
        directors = info.get("directors", [])
        year = str(info.get("post_Year", ""))
        channel = info.get("channel", "")
        genre = info.get("cidMapper", "") or info.get("videoType", "")
        cover_image = info.get("imgPath", "")

        return VideoMetadata(
            title=title,
            source_url=url,
            video_id=video_id,
            episode_id=episode_id,
            channel=channel,
            genre=genre,
            year=year,
            stars=stars,
            directors=directors,
            synopsis=synopsis,
            cover_image_url=cover_image,
            extra=info,
        )

    def _query_play_api(self, play_id: str, pub_key: str, priv_key: str) -> Dict[str, Any]:
        """Query video/play endpoint for a given play_id."""
        params = {
            "cinema": 1,
            "id": play_id,
            "a": 1,
            "usersign": 1,
            "device": 1,
            "region": "US",
            "isMasterSupport": 1,
        }

        signed_url = self._sign_request(
            f"{self.base_api}/video/play",
            params,
            pub_key,
            priv_key,
        )

        req = urllib.request.Request(signed_url, headers=self.headers)
        with urllib.request.urlopen(req, timeout=10) as res:
            return json.loads(res.read().decode("utf-8"))

    def resolve_stream(self, url: str) -> StreamInfo:
        """Resolve direct HLS stream URL with authentication tokens."""
        video_id, episode_id = self._parse_url_ids(url)
        pub_key, priv_key = self._fetch_page_keys(url)

        # Try video_id first, then episode_id if failed
        candidate_ids = [vid for vid in [video_id, episode_id] if vid]
        # Remove duplicates preserving order
        unique_ids = list(dict.fromkeys(candidate_ids))

        play_info = None
        last_resp = None

        for test_id in unique_ids:
            try:
                resp_data = self._query_play_api(test_id, pub_key, priv_key)
                last_resp = resp_data
                if resp_data.get("data", {}).get("code") == 0:
                    info_list = resp_data.get("data", {}).get("info", [])
                    if info_list and info_list[0]:
                        play_info = info_list[0]
                        break
            except Exception as e:
                logger.warning("Error querying play API for id %s: %s", test_id, e)

        if not play_info:
            raise ValueError(f"Failed to resolve video stream for {url}: {last_resp}")

        flv_list = play_info.get("flvPathList", [])
        clarity_list = play_info.get("clarity", [])

        stream_url = ""
        bitrate = 0
        resolution = ""

        # Find best available HLS stream from flvPathList
        for flv in flv_list:
            if flv.get("isHls") and flv.get("result"):
                stream_url = flv["result"]
                bitrate = flv.get("bitrate", 0)
                break

        # Fallback to clarity list
        if not stream_url:
            for clarity in clarity_list:
                path = clarity.get("path")
                if isinstance(path, dict) and path.get("result"):
                    stream_url = path["result"]
                    bitrate = clarity.get("bitrate", 0)
                    resolution = clarity.get("description", "")
                    break

        if not stream_url:
            raise ValueError(f"No playable HLS stream found in response for {url}")

        return StreamInfo(
            stream_url=stream_url,
            stream_type="hls",
            headers={
                "Referer": "https://www.yfsp.tv/",
                "User-Agent": DEFAULT_USER_AGENT,
            },
            bitrate=bitrate,
            resolution=resolution,
        )
