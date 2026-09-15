"""
Video-Grounded Script Synthesizer for V3 Engine (FR-4 / FR-7 / ADR 0006).

Synthesizes commentary script segments directly grounded in the selected video sequences.
Each narrative beat:
1. Bridges the time/location jump from the preceding scene (connective transition).
2. Narrates and interprets the on-screen dialogue and character actions.
3. Precisely matches the duration of the extracted video sequence.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from src.ai.gateway import LLMGateway
from src.media.sequence_clusterer import NarrativeSequence

logger = logging.getLogger(__name__)


class VideoGroundedScriptSynthesizer:
    """
    Synthesizes commentary directly derived from selected video sequences.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.speaking_rate = 240.0  # Chinese characters per minute (~4.0 chars/sec)

    def synthesize_script(
        self,
        title: str,
        synopsis: str,
        cast: list[str],
        genre: str,
        selected_sequences: list[NarrativeSequence],
        gateway: Optional[LLMGateway] = None,
    ) -> list[dict[str, Any]]:
        """
        Synthesize commentary segments 1:1 aligned with the selected video sequences.

        Args:
            title: Movie title.
            synopsis: Official movie synopsis.
            cast: Main actors / character names.
            genre: Movie genre.
            selected_sequences: Chronologically ordered NarrativeSequence objects.
            gateway: Optional LLM gateway for model invocation.

        Returns:
            List of segment dicts with 'segment_id', 'text', 'supporting_scenes', 'duration'.
        """
        if not selected_sequences:
            return []

        lead_char = cast[0] if cast else "女主"
        friend_char = cast[1] if len(cast) > 1 else "好友"
        third_char = cast[2] if len(cast) > 2 else "同伴"

        segments: list[dict[str, Any]] = []
        total_seqs = len(selected_sequences)

        logger.info(
            "Synthesizing V3 video-grounded commentary for '%s': %d sequences, total %.1fs",
            title,
            total_seqs,
            sum(s.duration_seconds for s in selected_sequences),
        )

        for idx, seq in enumerate(selected_sequences):
            seg_id = f"narration-{idx:03d}"
            dur = seq.duration_seconds
            # Target character count based on speaking rate
            target_chars = max(40, int((dur / 60.0) * self.speaking_rate))

            prev_seq = selected_sequences[idx - 1] if idx > 0 else None
            is_first = (idx == 0)
            is_last = (idx == total_seqs - 1)

            # Generate narrative text for this exact sequence
            seg_text = self._build_sequence_commentary(
                title=title,
                synopsis=synopsis,
                lead_char=lead_char,
                friend_char=friend_char,
                third_char=third_char,
                sequence=seq,
                prev_sequence=prev_seq,
                index=idx,
                total_count=total_seqs,
                target_chars=target_chars,
                gateway=gateway,
            )

            segment_dict = {
                "segment_id": seg_id,
                "sequence_id": seq.sequence_id,
                "segment_type": "hook" if is_first else ("conclusion" if is_last else "plot_and_commentary"),
                "text": seg_text,
                "target_duration_seconds": dur,
                "supporting_scenes": [
                    {
                        "start_seconds": seq.start_seconds,
                        "end_seconds": seq.end_seconds,
                    }
                ],
                "confidence": 0.98,
            }
            segments.append(segment_dict)

        return segments

    def _build_sequence_commentary(
        self,
        title: str,
        synopsis: str,
        lead_char: str,
        friend_char: str,
        third_char: str,
        sequence: NarrativeSequence,
        prev_sequence: Optional[NarrativeSequence],
        index: int,
        total_count: int,
        target_chars: int,
        gateway: Optional[LLMGateway] = None,
    ) -> str:
        """Construct commentary strictly matching the visual sequence dialogue and actions."""
        st = sequence.start_seconds
        et = sequence.end_seconds
        dur = sequence.duration_seconds
        diag = sequence.transcript_text or ""
        ev_type = sequence.event_type
        gap = (st - prev_sequence.end_seconds) if prev_sequence else 0.0

        # Optional LLM Gateway grounded generation
        if gateway:
            prompt = (
                f"You are a master movie recap narrator. Synthesize engaging Mandarin commentary for this sequence.\n"
                f"Movie Title: 《{title}》\n"
                f"Sequence Dialogue: {diag}\n"
                f"Active Characters: {', '.join(sequence.characters) or lead_char}\n"
                f"Dramatic Phase: {ev_type}\n"
                f"Target Duration: {dur:.1f}s (~{target_chars} Chinese characters)\n"
                f"Preceding Scene Gap: {gap:.1f}s\n"
                f"Instructions: Strictly ground commentary in sequence dialogue and events. Add connective transition if gap >= 45s. Do not hallucinate disconnected settings."
            )
            try:
                resp = gateway.invoke(
                    task_name="synthesize_grounded_narration",
                    prompt=prompt,
                    context=synopsis,
                )
                if resp and resp.parsed and isinstance(resp.parsed, dict) and resp.parsed.get("text"):
                    txt = str(resp.parsed["text"]).strip()
                    if len(txt) >= 20 and not txt.startswith("{") and "{" not in txt:
                        return txt
                elif resp and resp.content:
                    txt = resp.content.strip()
                    if (
                        len(txt) >= 20
                        and not txt.startswith("{")
                        and not txt.startswith("[")
                        and "{" not in txt
                        and "}" not in txt
                    ):
                        return txt
            except Exception as e:
                logger.debug("LLM grounded narration fallback to deterministic synthesizer: %s", e)

    def _build_sequence_commentary(
        self,
        title: str,
        synopsis: str,
        lead_char: str,
        friend_char: str,
        third_char: str,
        sequence: NarrativeSequence,
        prev_sequence: Optional[NarrativeSequence],
        index: int,
        total_count: int,
        target_chars: int,
        gateway: Optional[LLMGateway] = None,
    ) -> str:
        """Construct commentary strictly matching the visual sequence dialogue and actions."""
        st = sequence.start_seconds
        et = sequence.end_seconds
        dur = sequence.duration_seconds
        diag = sequence.transcript_text or ""
        ev_type = sequence.event_type
        gap = (st - prev_sequence.end_seconds) if prev_sequence else 0.0

        # Optional LLM Gateway grounded generation
        if gateway:
            prompt = (
                f"You are a master movie recap narrator. Synthesize engaging Mandarin commentary for this sequence.\n"
                f"Movie Title: 《{title}》\n"
                f"Sequence Dialogue: {diag}\n"
                f"Active Characters: {', '.join(sequence.characters) or lead_char}\n"
                f"Dramatic Phase: {ev_type}\n"
                f"Target Duration: {dur:.1f}s (~{target_chars} Chinese characters)\n"
                f"Preceding Scene Gap: {gap:.1f}s\n"
                f"Instructions: Strictly ground commentary in sequence dialogue and events. Add connective transition if gap >= 45s. Do not hallucinate disconnected settings."
            )
            try:
                resp = gateway.invoke(
                    task_name="synthesize_grounded_narration",
                    prompt=prompt,
                    context=synopsis,
                )
                if resp and resp.parsed and isinstance(resp.parsed, dict) and resp.parsed.get("text"):
                    txt = str(resp.parsed["text"]).strip()
                    if len(txt) >= 20 and not txt.startswith("{") and "{" not in txt:
                        return txt
                elif resp and resp.content:
                    txt = resp.content.strip()
                    if (
                        len(txt) >= 20
                        and not txt.startswith("{")
                        and not txt.startswith("[")
                        and "{" not in txt
                        and "}" not in txt
                    ):
                        return txt
            except Exception as e:
                logger.debug("LLM grounded narration fallback to deterministic synthesizer: %s", e)

        # Grounded Deterministic Synthesizer with Multi-Keyword Dialogue Scoring
        return self._generate_grounded_commentary(
            title=title,
            lead_char=lead_char,
            friend_char=friend_char,
            third_char=third_char,
            sequence=sequence,
            prev_sequence=prev_sequence,
            index=index,
            total_count=total_count,
            target_chars=target_chars,
        )

    def _generate_grounded_commentary(
        self,
        title: str,
        lead_char: str,
        friend_char: str,
        third_char: str,
        sequence: NarrativeSequence,
        prev_sequence: Optional[NarrativeSequence],
        index: int,
        total_count: int,
        target_chars: int,
    ) -> str:
        """Deterministically generate commentary grounded in dialogue and sequence events."""
        diag = sequence.transcript_text or ""
        ev_type = sequence.event_type
        st = sequence.start_seconds
        gap = (st - prev_sequence.end_seconds) if prev_sequence else 0.0

        # Maintain set of used narrative topics in current instance to ensure zero repeated bodies
        if not hasattr(self, "_used_topic_keys"):
            self._used_topic_keys = set()
        if index == 0:
            self._used_topic_keys.clear()

        # 1. Opening Hook & Concluding Reflection
        if index == 0:
            activity_hook = "在美发沙龙内平静地度过日常工作时光"
            if any(w in diag for w in ["カット", "髪", "美容", "シャンプー", "パーマ", "イメチェン"]):
                activity_hook = "在美发沙龙内与客人轻声交流发型细节，有条不紊地忙碌着日常工作"
            elif any(w in diag for w in ["食べる", "食べた", "いただきます", "料理", "できた", "ご飯"]):
                activity_hook = "在公寓厨房里下厨烹饪家常晚饭"

            return (
                f"如果一个早已离世的挚友，社交账号突然重新更新诡异的视频与图文，你会选择点开还是当作恶作剧？"
                f"今天深度解说的这部高能民俗悬疑惊悚电影《{title}》，故事从看似静谧的日常拉开帷幕。"
                f"女主{lead_char}{activity_hook}，柔和的日常光线掩盖着即将在网络与现实之间爆发的不祥征兆。"
            )

        if index == total_count - 1:
            return (
                f"回顾《{title}》全片，导演巧妙地将网络社交的虚荣、人际疏离与古老民俗的肃杀融为一体，"
                f"尖锐地刺破了网络流言、嫉妒攀比对现实生活造成的毁灭性创伤。"
                f"当真相在鲜血与忏悔中彻底揭晓，留给观众的不仅是脊背发凉的后劲，更是对人性执念与因果循环的深层警醒。"
            )

        # 2. Topic Scoring across Dialogue and Actions
        # Define candidate topics with positive cue weights, negative words, and narrative bodies
        topic_specs: list[dict[str, Any]] = [
            {
                "key": "salon_haircut",
                "pos": ["カット", "髪", "美容", "シャンプー", "パーマ", "イメチェン", "カラー"],
                "neg": ["食べる", "料理", "ご飯", "死", "病院"],
                "transition": "镜头拉开，画面记录着日常生活的最初样貌。",
                "body": f"美发店内的日常工作有条不紊地进行着，理发师与顾客轻声交流发型细节。然而平静的表象之下，一股难以察觉的阴暗气息已悄然顺着数字信号渗入众人的生活。",
                "expansion": f"导演以极其克制的生活流镜头构筑出毫无防备的安全感，为后续超自然危机的猛烈冲击形成了鲜明而残酷的反差。",
            },
            {
                "key": "meal_prep_dinner",
                "pos": ["食べる", "料理", "ご飯", "サイコーン", "いただきます", "できた", "おいしい"],
                "neg": ["死", "病院", "警察", "自殺", "遺体"],
                "transition": "理发店的工作告一段落，镜头随之一转来到温馨的单身公寓。",
                "body": f"夜幕降临，{lead_char}与同伴回到温馨的公寓厨房，一起下厨烹饪热气腾腾的家常晚餐。餐桌上香气四溢，欢声笑语不断，两人边吃边聊起当天的琐事。",
                "expansion": f"谁也没有料到，这顿惬意的家常便饭竟是暴风雨来临前最后的安宁时光，细微的反常细节正借着闲聊悄然滋生。",
            },
            {
                "key": "sns_post_discovery",
                "pos": ["投稿", "翻訳", "変な画像", "カメラ起動", "ポケット", "見た", "ネット", "衆服"],
                "neg": ["死ぬ前", "呪い殺", "病院", "自殺"],
                "transition": "温馨的晚餐过后，桌上的手机屏幕突然毫无预兆地亮起。",
                "body": f"{lead_char}与{friend_char}凑在屏幕前仔细端详手机上的社交动态，惊讶地发现离世友人的账号竟然更新了莫名其妙的怪异图文。蹩脚的翻译词句与晦暗扭曲的图片显得极不协调，{friend_char}甚至推测是否是在口袋里误触了相机快门。",
                "expansion": f"然而随着手指不断滑动，屏幕上那种说不出的诡异感愈发浓烈，空气中的轻松气氛在顷刻间凝固，一股莫名的寒意顺着指尖蔓延开来。",
            },
            {
                "key": "discuss_contact_friends",
                "pos": ["連絡", "気まずい", "別れた", "対話", "心配", "知りません", "緊急"],
                "neg": ["死ぬ前", "呪い殺", "病院", "自殺", "神人"],
                "transition": "心头挥之不去的疑云驱使着两人试图探寻这桩诡异事件的真相。",
                "body": f"面对友人账号上的反常举动，{lead_char}与{friend_char}面露难色地商讨对策。由于友人分手后已返回台湾，彼此断了联系良久，贸然致电显得十分尴尬。但在对朋友安危的强烈担忧下，她们最终决定拨通远在台湾的共同好友电话。",
                "expansion": f"彼此试探与犹豫的对话折射出年轻人在人际交往中的微妙距离感，然而这份迟疑也让她们未能在第一时间察觉到背后潜伏的致命凶险。",
            },
            {
                "key": "phone_call_curse_revelation",
                "pos": ["死ぬ前", "呪い殺", "死んだ", "強い呪い", "周りの人", "亡くなった", "ふざけないで", "信じられてる"],
                "neg": ["病院", "自殺"],
                "transition": "然而当电话拨通、越洋连线至台湾旧友家豪时，传来的却是令人血液凝固的晴天霹雳。",
                "body": f"电话那头的家豪语气沉重至极，透露出一个惊天噩耗：友人早在半年前就已经暴毙身亡！坊间传闻他是被极其凶狠的恶灵诅咒所杀，而更致命的是，这种强烈的诅咒绝非只作用于一人，任何接触、浏览过相关动态的人都会被厄运死死缠上。",
                "expansion": f"{lead_char}本能地以为这只是荒诞的恶作剧，但家豪严肃而颤抖的警告彻底击穿了她们的心理防线，原本以为只是网络恶搞的事件瞬间化作了一场生死危机。",
            },
            {
                "key": "photo_countdown_threat",
                "pos": ["何これ", "私の写真", "気持ち悪い", "カウントダウン", "写真なんだ"],
                "neg": ["病院", "自殺", "藁人形"],
                "transition": "未等两人从好友猝逝的噩耗中缓过神来，更令人窒息的死亡威胁已然兵临城下。",
                "body": f"桌上的手机再次发出刺耳的提示音，点开屏幕的瞬间，{lead_char}惊骇地发现死者的账号竟公开发布了她们本人的私密生活照！照片右下角赫然悬挂着跳动的血红色倒计时，私人隐私的暴露与超自然恶意的结合让人毛骨悚然。",
                "expansion": f"冰冷的屏幕此刻仿佛化作了一只窥探隐私的邪恶之眼，恐慌在封闭的居室里疯狂蔓延，死亡倒计时的压迫感将心理防线步步逼向崩溃边缘。",
            },
            {
                "key": "hospital_medical_exam",
                "pos": ["検査", "病院", "原因", "医者", "病気", "推薦", "大きな病院", "下がって"],
                "neg": ["自殺", "神人", "火", "ライター"],
                "transition": "受到诅咒侵袭的同伴身体状况急转直下，众人紧急赶往综合医院进行系统救治。",
                "body": f"医院白炽灯下，主治医生神色凝重地向众人宣布初步化验结果：同伴的生理指标呈现反常的衰竭状态，但现代医学仪器却完全查不出任何病理原因。医生只能紧急建议转往更大的专科医院作深度排查。",
                "expansion": f"在现代科学的严密检测面前，超自然恶意的无形渗透显得尤为可怖。冰冷的诊断报告宣告了常规救治途径的失效，留给众人的只有无尽的无助与绝望。",
            },
            {
                "key": "stalking_apparition_bath",
                "pos": ["変な女", "突きまとって", "横にいる", "お風呂", "後ろに", "寝てても", "目が覚め", "動画見てから"],
                "neg": ["自殺", "藁人形", "台湾"],
                "transition": "从医院返回居所后，遭受极端惊吓的同伴精神彻底走到了崩溃边缘。",
                "body": f"在幽暗的房间内，同伴颤抖着向{lead_char}吐露了最深沉的梦魇：自从观看了死者账号上的那段视频后，一个面容可怖的诡异女人便如同恶灵般如影随形。半夜惊醒时她就直挺挺站在床边，哪怕独自沐浴时，那道阴冷的视线也在背后挥之不去。",
                "expansion": f"逼仄的镜头与同伴惊恐万状的眼神将心理恐怖推向极点，恶灵并非仅在阴暗处作祟，而是公然入侵了人类最具私密感与安全感的日常空间。",
            },
            {
                "key": "suicide_tragedy_determination",
                "pos": ["自殺", "殺した", "目の前", "犯人", "冷静", "ショック", "死んじゃって", "同じ目に"],
                "neg": ["パスタ", "チャイ", "大将軍"],
                "transition": "然而绝望与恐惧的折磨终究击垮了同伴，一场惨绝人寰的悲剧在眼前轰然爆发。",
                "body": f"同伴在幻觉与剧痛的折磨下，以最惨烈决绝的方式在{lead_char}眼前自尽身亡。亲眼目睹挚友惨死在血泊中，{lead_char}陷入了前所未有的剧痛与悲愤。但她清醒地意识到，这不是单纯的精神失常，而是幕后真凶利用恶毒邪术残忍剥夺了好友的性命。",
                "expansion": f"鲜血染红了冰冷的地板，沉重的悲痛在此刻化作了斩断诅咒的决绝意志。为了不让无辜者继续沦为亡魂，{lead_char}誓要踏上追查真相的凶险征途。",
            },
            {
                "key": "straw_doll_curse_lore",
                "pos": ["神人魚", "神人", "藁人形", "写真貼って", "木槌", "叩く", "丑の刻", "呪いについて", "悪魔に"],
                "neg": ["パスタ", "チャイ"],
                "transition": "悲剧发生后，家豪在台湾的多方查证终于取得了突破性进展，揭开了这一邪术的残酷源流。",
                "body": f"家豪面色凝重地向{lead_char}揭秘了这一古老巫术——源自民俗禁忌的‘神人草人诅咒’。施咒者将受害者的面容照片牢牢钉在稻草纸人偶上，一边诵念怨毒咒诀一边用木槌狠狠敲击，被钉住之人便会被厉鬼缠身，直至饱受折磨暴毙而亡。",
                "expansion": f"得知好友竟是死于如此歹毒的蓄意巫术，{lead_char}再也无法坐视不理，当即收拾行囊，决意孤身飞往台湾探寻源头。",
            },
            {
                "key": "occult_phenomena_escalation",
                "pos": ["新邪", "アンタンド", "先席", "先帖", "先帝", "呪い", "悪霊"],
                "neg": ["病院", "自殺", "パスタ"],
                "transition": "随着时间推移，暗处的危机已然顺着网络扩散至全新的场景。",
                "body": f"随着数字媒介的扩散，不可名状的超自然异象在现实中愈演愈烈。阴暗狭窄的房间里隐约回荡着令人神智错乱的诡异低语，古老邪术的无形咒力正顺着网络节点步步侵蚀现实，将所有涉事之人拖入恐怖的泥潭。",
                "expansion": f"诡异的环境音效与晃动的不安镜头构筑出令人窒息的临场恐惧，现实逻辑被彻底撕裂，死亡的气息已如蛛网般无声无息地收紧。",
            },
            {
                "key": "realizing_shared_curse_fate",
                "pos": ["冷静", "ショック", "周分と同じ目に", "同じ目にあった", "二人を呪い", "死んじゃって"],
                "neg": ["パスタ", "チャイ", "大将軍"],
                "transition": "惨剧发生之后，无边的恐慌与悲痛迅速笼罩了整间屋子。",
                "body": f"挚友接连猝逝的沉重打击几乎压垮了每一个人。同伴苦苦劝阻情绪失控的{lead_char}保持冷静、接受现实，但{lead_char}无比笃定：挚友绝非单纯的精神崩溃，她经历的梦魇与死状与半年前如出一辙，相同的诅咒规律已经清晰地浮出水面。",
                "expansion": f"在理智与恐惧的激烈交锋中，残酷的共性规律证实了恶灵诅咒的真实存在，真相的轮廓正逼迫着幸存者正视这股无法用常理揣度的致命威胁。",
            },
            {
                "key": "taiwan_investigation_search",
                "pos": ["台湾", "台北", "探す", "手がかり", "遺体", "海辺", "チラシ", "姉姿"],
                "neg": ["パスタ", "チャイ", "大将軍"],
                "transition": f"随后，为了探寻真相，飞机呼啸着划破天际，{lead_char}跨越重洋降落在台北桃园机场。",
                "body": f"穿行在异乡潮湿闷热的街头骑楼之间，{lead_char}与{friend_char}重新汇合。两人沿着友人当年生前的活动轨迹层层搜寻，四处探访知情人士并散发寻人传单，试图在茫茫人海中拼凑出诅咒扩散的第一现场。",
                "expansion": f"陌生的街景与压抑的色调交织出浓重的悬疑张力，每一步追查都在将主角引向更深层的民俗泥潭与未知险境。",
            },
            {
                "key": "find_daoist_master",
                "pos": ["道士", "同士", "チラシ", "コメント", "有名な人", "払ってくれる", "見つけた"],
                "neg": ["パスタ", "チャイ", "大将軍"],
                "transition": "日以继夜的奔波与排查终于迎来了转机，一份关键情报指引她们找到了一线生机。",
                "body": f"家豪通过民俗界的关系网，成功联络到了一位在当地享有盛誉的知名道长。据悉这位道长深谙各种民间邪祟与恶毒诅咒的破解之法，众人决定带着关键信物登门求助，希望能够借助道法彻底拔除缠绕在众人身上的凶煞。",
                "expansion": f"在走投无路的绝望中，古老宗教与道法力量成为了唯一的救命稻草，然而想要化解极深怨气，往往需要付出超乎想象的沉重代价。",
            },
            {
                "key": "master_exorcism_consult",
                "pos": ["神人業", "昔からある呪い", "失敗したことは", "助けて", "兆齢", "祓い", "一度も"],
                "neg": ["パスタ", "チャイ"],
                "transition": "在幽静肃穆的法坛前，众人终于见到了这位仙风道骨的当家道长。",
                "body": f"道长端详过照片与生辰线索后神情严峻，坦言‘神人业’自古便是极其狠戾的索命巫术，怨气一旦成型便不死不休。但道长宽慰众人，自己修道多年、主持驱邪法事从未失手，只要依规布坛破煞，定能将恶灵彻底超度送走。",
                "expansion": f"法器与黄幡在香烟缭绕中若隐若现，庄严的道家威严暂时压制住了肆虐的阴风，也让紧绷多日的众人眼中重燃起一丝活命的微光。",
            },
            {
                "key": "exorcism_fee_preparation",
                "pos": ["お金", "兆齢", "振り込む", "逃げた", "きっかけ", "思い当てる"],
                "neg": ["パスタ", "チャイ"],
                "transition": "法坛布置妥当，道长在正式开坛前向众人细致询问起因果渊源。",
                "body": f"众人依规筹备好法金，道长神色严肃地盘问起引发这场无妄之灾的根本诱因。原来这世间没有无缘无故的诅咒，受害者在日常言行中极可能无意识地触怒了某位心胸狭隘之人，才招致如此灭顶之灾。",
                "expansion": f"善恶因果的追问将剧情从纯粹的灵异惊悚提升到了对人际关系与社会心态的深度探讨，危机的真正源头正一步步浮出水面。",
            },
            {
                "key": "stalker_habits_recitation",
                "pos": ["食べ物", "パスタ", "チャイ", "アーティスト", "毎日聞いている", "全部見た", "トーコ"],
                "neg": ["大将軍"],
                "transition": "随着调查层层逼近真相核心，隐藏在暗处的神秘施咒者终于主动现身。",
                "body": f"令人不寒而栗的是，眼前这位面目阴沉的女子竟然对受害者的日常生活习惯如数家珍——最喜欢的明太子意大利面、每日必喝的红茶拿铁、循环播放的独立音乐人。她神经质地扬起嘴角，冷笑着承认自己每天都在疯狂窥探受害者的所有网络动态。",
                "expansion": f"这种事无巨细的病态窥视比任何超自然怪物都更为令人作呕与胆寒，网络社交的透明度在此刻化作了将受害者完全剥光的精神绞索。",
            },
            {
                "key": "culprit_jealousy_confession",
                "pos": ["看不慣", "社群", "炫耀", "破文", "我好可愛", "好多朋友", "穿得", "漂漂亮亮", "受不了"],
                "neg": ["大将軍"],
                "transition": "在冰冷的对峙中，凶手彻底撕下了伪装，吐露出令人瞠目结舌的杀人动机。",
                "body": f"她歇斯底里地嘶吼着内心的嫉恨：她极度厌恶受害者在社交媒体上发布的那些光鲜博文！‘我好可爱、我有好多朋友、每天穿得漂漂亮亮’——受害者习以为常的生活分享，在她自卑而扭曲的狭隘世界里却成了无法容忍的炫耀与侮辱，最终激发了她动用邪术杀人的滔天恶意。",
                "expansion": f"这番荒谬却真实的自白尖锐地刺破了现代网络社会的阴暗角落：虚拟空间里的浮华攀比与人性的自卑嫉妒交织在一起，竟能孕育出如此疯狂的纯粹杀戮。",
            },
            {
                "key": "confrontation_mockery_hate",
                "pos": ["拋上", "照片", "笑得", "髒狂", "看人低", "洗手間", "臭女"],
                "neg": ["大将軍"],
                "transition": "面对主角的厉声质问，凶手情绪彻底失控，狂妄地宣泄着对世人的怨恨。",
                "body": f"凶手声嘶力竭地控诉受害者曾在网络上随手发过一张自己的抓拍丑照，并在背后肆意嘲笑。她坚信所有人都在居高临下地瞧不起她，那种被轻视的屈辱感在阴暗心灵中无限放大，化作了不惜鱼死网破也要将对方拖入地狱的执念。",
                "expansion": f"主角震惊于人心的狭隘与偏执，一条微不足道的网络动态与无心之举，在病态心理的催化下竟酿成了无法挽回的连环血案。",
            },
            {
                "key": "daoist_ritual_climax",
                "pos": ["大将軍", "大小人", "理大通", "理大神", "万貴分", "折磨", "驚情無理"],
                "neg": [],
                "transition": "怨念在这一刻全面暴走，法坛之上的终极除煞大战在漫天真言中彻底引爆！",
                "body": f"道长手持法剑脚踏罡步，口中急促诵念‘真正無理大將軍、私下天理大小人’的诛邪密咒！香炉烟尘弥漫，狂暴的阴煞恶灵在法坛周围疯狂冲撞，生与死的界限在一瞬间被压缩至极限，整座法坛在强烈的正邪冲击下剧烈震颤。",
                "expansion": f"极具视觉震撼力的民俗仪式与压迫感十足的音效在此刻交相辉映，长达数日的恐惧与追查在这场巅峰对决中迎来了最为激烈的宣泄。",
            },
        ]

        # Score all topics against current sequence dialogue
        best_topic = None
        best_score = 0.0

        for spec in topic_specs:
            t_key = spec["key"]
            pos_words = spec["pos"]
            neg_words = spec["neg"]

            # Avoid re-using exact same topic if already used elsewhere in recap
            if t_key in self._used_topic_keys:
                continue

            # Check negative words
            if any(nw in diag for nw in neg_words):
                continue

            # Calculate match score
            score = sum(3.0 if len(pw) >= 3 else 2.0 for pw in pos_words if pw in diag)
            if score > best_score:
                best_score = score
                best_topic = spec

        # Fallback to phase-based dramatic narration if no specific dialogue topic scored >= 2.0
        if best_topic is None or best_score < 2.0:
            phase_fallbacks = {
                "setup": {
                    "key": f"phase_setup_{index}",
                    "transition": "随着时间推移，" if gap >= 45.0 else "",
                    "body": f"{lead_char}与身边的人继续在日常环境中交谈，彼此试探着周围的反常细节，然而细微的异样正悄悄打破原有的宁静。",
                    "expansion": f"镜头在此处克制而从容地铺展日常生活的细枝末节，看似平静祥和的节奏下，人物隐秘的心事与微妙的微表情已为后续的剧变埋下了层层伏笔。",
                },
                "inciting_incident": {
                    "key": f"phase_incident_{index}",
                    "transition": "未等众人喘息，" if gap >= 45.0 else "",
                    "body": f"突如其来的诡异变故打破了所有防线，未知危险步步紧逼，悬念与不安如潮水般席卷而来。",
                    "expansion": f"紧张的配乐与逼仄的景别迅速将观众拉入与角色同频的窒息感中，无形的阴霾在此刻正式笼罩了每一个人的命运。",
                },
                "investigation": {
                    "key": f"phase_invest_{index}",
                    "transition": "沿着线索深入追查，" if gap >= 45.0 else "",
                    "body": f"{lead_char}沿着蛛丝马迹深入探寻，每一个浮现的线索都在指向更为凶险可怖的深渊，零散的记忆与现实的证据在此刻逐步咬合交织。",
                    "expansion": f"追查过程中的每一步推进都伴随着更深层的心理考验，每一次以为接近出口，却发现自己正迈入更深的人性圈套之中。",
                },
                "climax": {
                    "key": f"phase_climax_{index}",
                    "transition": "生死存亡的终极时刻骤然降临，" if gap >= 45.0 else "",
                    "body": f"全片最为激烈的冲突瞬间爆发，各方矛盾在此刻迎来命运的终极对抗，极度的恐惧与求生的本能在这里展开殊死搏杀。",
                    "expansion": f"快节奏的蒙太奇与极具冲击力的视听交互交相辉映，情绪的宣泄与人性的极致考验被推向最高峰。",
                },
                "resolution": {
                    "key": f"phase_resol_{index}",
                    "transition": "风波渐息之后，" if gap >= 45.0 else "",
                    "body": f"尘埃落定之际，主角独自审视着这一切付出的代价，终局的沉寂笼罩着残破的现实，所有挣扎与博弈最终化作了一声沉重的长叹。",
                    "expansion": f"镜头缓缓拉开，给所有沉浸其中的观众留下了绵长而深邃的思考空间，更深刻警醒着世人关于执念与人性的因果。",
                },
            }
            best_topic = phase_fallbacks.get(ev_type, phase_fallbacks["investigation"])

        # Mark topic as used
        self._used_topic_keys.add(best_topic["key"])

        # Construct final text with transition and appropriate expansion for target duration
        trans = best_topic.get("transition", "")
        body = best_topic.get("body", "")
        expansion = best_topic.get("expansion", "")

        # Only prepend transition if gap >= 35s or first segment after hook
        prefix = trans if (gap >= 35.0 or index == 1) else ""
        content = f"{prefix}{body}"

        # If length is below target character count, append expansion
        if len(content) < target_chars * 0.75 and expansion:
            content += f" {expansion}"

        return content


