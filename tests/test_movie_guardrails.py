from __future__ import annotations
import pytest
from src.ai.gateway import LLMGateway
from src.ai.verify import _check_movie_identity_and_isolation, verify_script, is_same_movie
from src.eval.deterministic_eval import evaluate_deterministic
from src.eval.slop_detector import SlopDetector
from src.eval.models import EvalGrade


def test_cache_key_project_isolation(tmp_path):
    """Test that LLM Gateway cache keys are scoped by project_id to prevent cross-movie cache hits."""
    config = {
        "llm": {
            "cache_enabled": True,
            "cache_dir": str(tmp_path / "cache"),
        }
    }
    gateway = LLMGateway(config)

    prompt = "Generate commentary for the opening scene."
    key_curse = gateway._cache_key("script_writing", prompt, project_id="proj-curse")
    key_steel = gateway._cache_key("script_writing", prompt, project_id="proj-steel")
    key_empty = gateway._cache_key("script_writing", prompt, project_id="")

    # Must produce different hash keys
    assert key_curse != key_steel
    assert key_curse != key_empty
    assert key_steel != key_empty


def test_movie_identity_verification_detects_foreign_movie():
    """Test that script verification detects and rejects cross-movie contamination."""
    story = {"title": "诅咒", "movie_title": "诅咒", "characters": [{"name": "张三"}]}

    # Contaminated script mentioning Tears of Steel and Celia
    contaminated_script = {
        "title": "诅咒",
        "segments": [
            {
                "segment_id": "seg-001",
                "segment_type": "hook",
                "text": "这是一部扣人心弦的悬疑电影《诅咒》。",
            },
            {
                "segment_id": "seg-002",
                "segment_type": "plot_and_commentary",
                "text": "西莉亚在阿姆斯特丹启动了毁灭性的记忆机器。",
            },
        ],
    }

    findings = _check_movie_identity_and_isolation(contaminated_script, story)
    assert any(f["check"] == "movie_identity_isolation" and f["severity"] == "fail" for f in findings)


def test_movie_identity_verification_detects_mismatched_hook_title():
    """Test that hook segment introducing a foreign movie title fails verification."""
    story = {"title": "诅咒", "movie_title": "诅咒", "characters": []}

    mismatched_hook_script = {
        "title": "诅咒",
        "segments": [
            {
                "segment_id": "seg-001",
                "segment_type": "hook",
                "text": "今天我们来看科幻巨作《钢铁之泪》，探索未来的毁灭与救赎。",
            },
            {
                "segment_id": "seg-002",
                "segment_type": "plot_and_commentary",
                "text": "深山古庙中，诡异的木偶引发了一连串怪事。",
            },
        ],
    }

    findings = _check_movie_identity_and_isolation(mismatched_hook_script, story)
    assert any(f["check"] == "movie_identity_title_mismatch" and f["severity"] == "fail" for f in findings)


def test_movie_identity_verification_passes_clean_script():
    """Test that authentic matching script passes movie identity verification."""
    story = {"title": "诅咒", "movie_title": "诅咒", "characters": [{"name": "老李"}]}

    clean_script = {
        "title": "诅咒",
        "segments": [
            {
                "segment_id": "seg-001",
                "segment_type": "hook",
                "text": "一具神秘古尸引发的连环疑案，这部《诅咒》将人性的贪婪展现得淋漓尽致。",
            },
            {
                "segment_id": "seg-002",
                "segment_type": "plot_and_commentary",
                "text": "随着老李深入调查，隐藏在深山村落背后的惊天秘密终于浮出水面。",
            },
        ],
    }

    findings = _check_movie_identity_and_isolation(clean_script, story)
    assert not any(f["severity"] == "fail" for f in findings)
    assert any(f["check"] == "movie_identity_isolation" and f["severity"] == "pass" for f in findings)


def test_is_same_movie_alias_matching():
    """Test alias and cross-language title normalization."""
    signatures = ["钢铁之泪", "Tears of Steel", "西莉亚"]
    assert is_same_movie("钢铁之泪", "钢铁之泪", signatures) is True
    assert is_same_movie("Tears of Steel", "钢铁之泪", signatures) is True
    assert is_same_movie("《钢铁之泪》", "钢铁之泪", signatures) is True
    assert is_same_movie("诅咒", "钢铁之泪", signatures) is False
    assert is_same_movie("The Curse", "钢铁之泪", signatures) is False


def test_slop_detector_rejects_cross_contaminated_movie():
    """Test that SlopDetector flags contaminated movies with Tier F and saves LLM eval tokens."""
    detector = SlopDetector()

    contaminated_script = {
        "title": "诅咒",
        "project_id": "proj-curse-test",
        "estimated_duration_minutes": 2.0,
        "segments": [
            {
                "segment_id": "narration-000",
                "segment_type": "hook",
                "text": "这是一部经典的悬疑电影《诅咒》。",
                "supporting_scenes": [{"start_seconds": 10.0, "end_seconds": 30.0}],
            },
            {
                "segment_id": "narration-001",
                "segment_type": "plot_and_commentary",
                "text": "然而西莉亚在阿姆斯特丹的末日机甲已经觉醒。",
                "supporting_scenes": [{"start_seconds": 40.0, "end_seconds": 80.0}],
            },
        ],
    }

    story = {"title": "诅咒", "movie_title": "诅咒"}
    report = detector.evaluate_script(contaminated_script, story=story)

    assert report.grade == EvalGrade.TIER_F
    assert report.passed is False
    assert report.is_slop is True
    assert any("Cross-movie contamination" in fail for fail in report.deterministic.hard_failures)
    # LLM evaluation tokens should be 0 because hard failure skipped semantic evaluation
    assert report.total_eval_tokens == 0
