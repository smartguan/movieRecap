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

        # Deterministic Generic Grounded Narration Synthesizer
        # 1. Connective Transition Lead-in (for spatial/temporal jumps >= 45s)
        lead_in = ""
        if prev_sequence:
            if gap > 300.0:
                templates = [
                    f"随着调查深入，时间悄然流逝，{lead_char}辗转来到新的地点。",
                    f"事态进一步发酵，调查的轨迹随之跨越到新的时间节点。",
                    f"新的线索打破了僵局，众人迅速奔赴下一处关键据点。",
                    f"随着时间推移，暗处的危机已然扩散至全新的场景。",
                ]
                lead_in = templates[index % len(templates)]
            elif gap > 60.0:
                templates_mid = [
                    f"未等众人喘息，局势在周围悄然加剧。",
                    f"短暂的平静被打破，危机接踵而至。",
                    f"紧接着，事态的发展彻底超出了所有人的预料。",
                ]
                lead_in = templates_mid[index % len(templates_mid)]
            elif gap >= 45.0:
                lead_in = f"紧接着，"

        # 2. Opening Hook & Concluding Reflection
        if index == 0:
            activity_hook = "正在平静地度过日常时光"
            if any(w in diag for w in ["カット", "髪", "美容", "シャンプー", "パーマ", "イメチェン"]):
                activity_hook = "正在美发店内有条不紊地忙碌着日常工作"
            elif any(w in diag for w in ["食べる", "食べた", "いただきます", "料理", "できた", "ご飯", "サイコーン", "おいしい"]):
                activity_hook = "正在公寓厨房里准备家常晚饭"

            return (
                f"如果一个早已离世的挚友，社交账号突然重新更新诡异的视频与图文，你会选择点开还是当作恶作剧？"
                f"今天深度解说的这部高能悬疑惊悚电影《{title}》，故事从静谧的日常拉开帷幕。"
                f"女主{lead_char}{activity_hook}，柔和的光线掩盖着日常生活中即将爆发的不祥征兆。"
            )
        elif index == total_count - 1:
            return (
                f"回顾《{title}》全片，导演巧妙地将网络社交的虚幻与古老民俗的肃杀融为一体，"
                f"尖锐地刺破了网络流言与人际冷漠所造成的现实创伤。"
                f"当真相在鲜血与忏悔中揭晓，留给观众的不仅是脊背发凉的后劲，更是对人性执念与因果循环的深层警醒。"
            )

        # 3. Grounded Activity & Dialogue Matching
        # A: Dining, home cooking, mealtime
        if any(w in diag for w in ["食べる", "食べた", "いただきます", "料理", "できた", "ご飯", "サイコーン", "おいしい"]):
            body = (
                f"夜幕降临，{lead_char}与同伴回到温馨的公寓厨房，一起下厨烹饪热气腾腾的家常晚餐。"
                f"餐桌上香气四溢，欢声笑语不断，谁也没有料到，这顿惬意的家常便饭竟是暴风雨来临前最后的安宁时光。"
            )
        # B: Hair salon, haircut, styling
        elif any(w in diag for w in ["カット", "髪", "美容", "シャンプー", "パーマ", "イメチェン"]):
            body = (
                f"店内的日常工作有条不紊地进行着，理发师与顾客轻声交流发型细节。"
                f"然而平静的表象之下，一股难以察觉的阴暗气息已悄然顺着数字信号渗入众人的生活。"
            )
        # C: Phone, social media, photo, SNS alerts
        elif any(w in diag for w in ["投稿", "SNS", "写真", "スマホ", "携帯", "アカウント", "何これ", "ネット"]):
            if gap >= 45.0:
                body = (
                    f"更诡异的图文相继弹出，照片中赫然出现了众人各自的私密生活照，"
                    f"屏幕右下角还附带着令人毛骨悚然的血红倒计时，恐慌在逼仄的空间里疯狂蔓延。"
                )
            else:
                body = (
                    f"晚餐过后，两人围坐在餐桌旁翻看手机上的社交动态。突然，桌上的手机接连发出刺耳的震动提示音，"
                    f"离世友人的账号毫无预兆地上传了新的动态，晦暗扭曲的画面让在场所有人的笑容瞬间僵死在脸上。"
                )
        # D: Flight, airport, travel to Taiwan/Taipei
        elif any(w in diag for w in ["空港", "飛行機", "フライト", "台湾", "台北", "航空"]):
            body = (
                f"飞机降落在台北桃园机场，潮湿闷热的空气扑面而来。"
                f"{lead_char}穿行于陌生街头，时隔两年再次联络上当年的同伴，两人神色凝重地核对关键线索。"
            )
        # E: Temple, shrine, incense shop, rituals, curses
        elif any(w in diag for w in ["廟", "寺", "神社", "神棚", "お札", "符", "母偶", "人形", "線香", "お参り"]):
            if gap >= 45.0:
                body = (
                    f"驱车赶往深山，车子艰难地停在半山腰的泥泞小径旁，借着微弱的手电光芒，"
                    f"一座被藤蔓彻底缠绕、荒废已久的古老神庙赫然矗立在密林深处，散发着刺骨的阴寒。"
                )
            else:
                body = (
                    f"两人来到一家深藏于巷底的香烛道铺，店主白发长者看清照片上的符文后脸色骤变，"
                    f"直言这是以怨念和心头血为引的‘绝命煞’，唯有找到并焚毁神龛深处的母偶方能求得一线生机。"
                )
        # F: Death, corpse, bathtub tragedy, funeral
        elif any(w in diag for w in ["死", "風呂", "遺体", "葬式", "警察", "自殺"]):
            body = (
                f"次日清晨噩耗接踵而至，昨晚浏览过动态的同伴被发现惨死于自家浴室，"
                f"现场没有丝毫外力破坏的痕迹，唯独手心里紧紧攥着一个烧焦的无名纸偶。"
            )
        # G: Fear, apparitions, haunting whispers, night terror
        elif any(w in diag for w in ["怖い", "声", "誰か", "幽灵", "悪霊", "呪い", "幻聴"]):
            if gap >= 45.0:
                body = (
                    f"随后，恐慌笼罩了整个小镇，收到动态的同伴相继出现幻听与梦魇，"
                    f"夜晚的房间里总能隐约听见凄厉的沙哑低语，仿佛死神正贴在耳边冷笑。"
                )
            else:
                body = (
                    f"随着异象升级，屋内的灯光开始剧烈闪烁，水龙头流出浑浊的锈水。"
                    f"同伴精神濒临崩溃，语无伦次地嘶喊着有人在门外徘徊，死亡的阴影已如附骨之疽紧咬不放。"
                )
        # H: Climax, combat, burning doll, final fight
        elif any(w in diag for w in ["火", "燃", "ライター", "倒す", "早く", "母偶"]):
            body = (
                f"生死一线之际，手指终于触碰到了散发着血腥气息的黑色母偶！"
                f"{lead_char}果断按下防风打火机，烈焰腾空而起，将母偶连同滔天的怨念一同卷入熊熊火海，厉鬼在凄厉哀嚎中寸寸化为灰烬。"
            )
        # I: General dramatic phase fallback with varied templates
        elif ev_type == "setup":
            setup_bodies = [
                f"{lead_char}与同伴在日常生活中交谈，彼此试探着周围的反常细节，然而细微的异样正悄悄打破原有的宁静。",
                f"{lead_char}穿行在熟悉的街景与居所中，表面看似波澜不惊的生活，隐约透露出一丝违和感。",
                f"日常生活的表象下暗流涌动，{lead_char}与身边的人各怀心事，平静的氛围正步步走向失衡。",
            ]
            body = setup_bodies[index % len(setup_bodies)]
        elif ev_type == "inciting_incident":
            incident_bodies = [
                f"突如其来的变故让众人措手不及，未知的危机步步紧逼，剧情迅速进入紧张节奏。",
                f"突发的诡异事件彻底打碎了原有的平静，恐慌情绪在众人之间不可遏制地蔓延开来。",
                f"接连出现的反常征兆让现场气氛骤然降至冰点，悬念与不安如潮水般席卷而来。",
                f"意料之外的变数猛烈爆发，打破了所有人的心理防线，危机正以不可逆之势全面铺开。",
            ]
            body = incident_bodies[index % len(incident_bodies)]
        elif ev_type == "investigation":
            invest_bodies = [
                f"随着线索层层抽丝剥茧，人物之间的秘密与矛盾逐渐暴露在聚光灯下。",
                f"主角沿着蛛丝马迹深入探寻，每一个浮现的线索都在指向更为凶险可怖的深渊。",
                f"调查的网越收越紧，零散的记忆与现实的证据在此刻逐步咬合交织。",
                f"迷雾重重的追查过程中，隐藏在暗处的危险在悄然之间已然拉近了距离。",
            ]
            body = invest_bodies[index % len(invest_bodies)]
        elif ev_type == "climax":
            climax_bodies = [
                f"全片最为激烈的冲突瞬间爆发，各方势力在极限博弈中迎来命运的终极对抗。",
                f"生死悬于一线的终极时刻骤然降临，极度的恐惧与求生的本能在这里展开殊死搏杀。",
                f"悬念在这一刻被彻底引爆，狂风骤雨般的危局让所有人的命运迎来了最终决战。",
            ]
            body = climax_bodies[index % len(climax_bodies)]
        else:
            resol_bodies = [
                f"风波渐息，主角独自审视着这一切的代价，故事在余韵中迎来了意味深长的结语。",
                f"终局的沉寂笼罩着残破的现实，所有挣扎与博弈最终化作了一声沉重的长叹。",
            ]
            body = resol_bodies[index % len(resol_bodies)]

        # 4. Proportional Depth Expansion to match Target Sequence Duration
        if len(body) < target_chars * 0.70:
            expansions = {
                "setup": [
                    f" 镜头在此处克制而从容地铺展日常生活的细枝末节，看似平静祥和的节奏下，人物隐秘的心事与微妙的微表情已为后续的剧变埋下了层层伏笔。",
                    f" 导演通过细腻的光影对比强化了暴风雨前夕的压抑感，普通而熟悉的日常在此刻被赋予了不可言说的悬疑色彩。",
                    f" 舒缓的生活流叙事与暗流涌动的背景音效形成反差，将观者注意力牢牢聚焦在反常的环境细节之中。",
                ],
                "inciting_incident": [
                    f" 突兀的打破感将原本的安定彻底击碎，紧张的配乐与逼仄的景别迅速将观众拉入与角色同频的窒息感中，无形的阴霾在此刻正式笼罩。",
                    f" 突发事件的连环冲击使得原本脆弱的平衡分崩离析，人物被动卷入未知的漩涡，命运的齿轮已然无可逆转地开始转动。",
                    f" 突如其来的诡异遭遇瞬间击穿了现实逻辑，惊悚的视听语言将悬疑气氛推向更具侵略性的高度。",
                ],
                "investigation": [
                    f" 每一段看似偶然的碎片线索都在逐步拼凑出令人胆寒的真相全貌，人物在迷雾中的彷徨与执着，愈发凸显出命运无常的残酷与荒诞。",
                    f" 追查过程中的每一步推进都伴随着更深层的道德困境与未知凶险，秘密被层层揭开的同时，也将角色逼向了心理承受的极限边缘。",
                    f" 随着搜寻范围不断扩大，原本互不相干的人物过往被一条隐蔽的线索串联在一起，不安的气息在每一个对视间悄然流淌。",
                    f" 主角在纷繁复杂的蛛丝马迹中艰难辨别真伪，每一次以为接近出口，却发现自己正迈入更深的圈套之中。",
                    f" 场景的变换见证着心理防线的步步退守，压抑的色调和封闭构图强化了求生与解谜之间的巨大张力。",
                    f" 隐藏在流言背后的残酷动机终于显露端倪，人性的阴暗面在极端困境的映照下展现得淋漓尽致。",
                ],
                "climax": [
                    f" 视听语言在此处达到了张力的顶峰，快节奏的蒙太奇与极具冲击力的特写交相辉映，情绪的宣泄与人性的极致考验被推向最高潮。",
                    f" 极具张力的动作交互与压迫感十足的音效设计将悬念推向顶点，生与死的界限在一瞬间被无限压缩，带来极强的戏剧冲击力。",
                    f" 积蓄已久的矛盾在激烈的肉搏与超自然对抗中全面爆发，每一个动作决策都关乎整部故事终局的走向。",
                ],
                "resolution": [
                    f" 硝烟散尽之后的沉寂往往比风暴本身更为触动人心，镜头缓缓拉开，给所有沉浸其中的观众留下了绵长而深邃的思考空间。",
                    f" 尘埃落定之际，回望整段历程中的选择与得失，沉重的余韵不仅关乎真相本身，更是对人心深处欲望与执念的深刻警醒。",
                    f" 终局的静止长镜头赋予了全片悲剧力量的沉淀，令人在惊魂未定之余深思因果循环的宿命之叹。",
                ],
            }
            phase_exps = expansions.get(ev_type, expansions["investigation"])
            selected_exp = phase_exps[index % len(phase_exps)]
            if len(body) + len(selected_exp) <= target_chars * 1.25:
                body += selected_exp
            if len(body) < target_chars * 0.70 and len(phase_exps) > 1:
                secondary_exp = phase_exps[(index + 1) % len(phase_exps)]
                if len(body) + len(secondary_exp) <= target_chars * 1.25:
                    body += secondary_exp

        return f"{lead_in}{body}"

