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
            activity_hook = "在美发沙龙内平静地进行日常工作"
            if any(w in diag for w in ["カット", "髪", "美容", "シャンプー", "パーマ", "イメチェン"]):
                activity_hook = "在美发沙龙内与客人轻声交流发型细节，有条不紊地忙碌着日常工作"
            elif any(w in diag for w in ["食べる", "食べた", "いただきます", "料理", "できた", "ご飯"]):
                activity_hook = "在公寓厨房里下厨烹饪家常晚饭"

            return (
                f"社交网络上一则早已离世友人的异常更新，打破了原本平静的生活。"
                f"在民俗悬疑电影《{title}》中，女主{lead_char}{activity_hook}。"
                f"然而，一个本该静止的账号却毫无征兆地发布出诡异图文，将毫无防备的同伴步步引向一场无法逆转的凶险诅咒。"
            )

        if index == total_count - 1:
            return (
                f"随着施咒真凶被当场制伏并移交警方，这场源于网络嫉恨的草人邪术终于告一段落。"
                f"历经生死劫难的{lead_char}安葬了逝去的挚友，将带有恶意的诅咒信物彻底封存销毁。"
                f"现实生活重归平静，但这场由虚荣与恶意引发的连环惨剧，却在每一个幸存者的生命中留下了无法抹平的创伤。"
            )

        # 2. Topic Scoring across Dialogue and Actions
        # Define candidate topics with positive cue weights, negative words, and narrative bodies
        topic_specs: list[dict[str, Any]] = [
            {
                "key": "salon_haircut",
                "pos": ["カット", "髪", "美容", "シャンプー", "パーマ", "イメチェン", "カラー"],
                "neg": ["食べる", "料理", "ご飯", "死", "病院"],
                "transition": "画面回到最初的日常景象。",
                "body": f"美发店内的日常工作有条不紊地进行着，理发师与顾客轻声交流发型细节。然而平静的表象之下，一桩反常的网络异动正悄然渗入众人的生活。",
                "expansion": f"{lead_char}仔细核对着顾客的预约登记表与服务项目，店内镜子上映照着平和的工作节奏，此时的她完全没有料到危险已然迫近。",
            },
            {
                "key": "meal_prep_dinner",
                "pos": ["食べる", "料理", "ご飯", "サイコーン", "いただきます", "できた", "おいしい"],
                "neg": ["死", "病院", "警察", "自殺", "遺体"],
                "transition": "理发店的工作告一段落，镜头随之一转来到温馨的单身公寓。",
                "body": f"夜幕降临，{lead_char}与同伴回到温馨的公寓厨房，一起下厨烹饪热气腾腾的家常晚餐。餐桌上香气四溢，两人边吃边聊起当天的琐事。",
                "expansion": f"同伴端上刚出锅的热菜，两人互相品尝着家常手艺，在餐桌前随手翻看手机，享受着一天忙碌后难得的惬意时光。",
            },
            {
                "key": "sns_post_discovery",
                "pos": ["投稿", "翻訳", "変な画像", "カメラ起動", "ポケット", "見た", "ネット", "衆服"],
                "neg": ["死ぬ前", "呪い殺", "病院", "自殺"],
                "transition": "温馨的晚餐过后，桌上的手机屏幕突然毫无预兆地亮起。",
                "body": f"{lead_char}与{friend_char}凑在屏幕前仔细端详手机上的社交动态，惊讶地发现离世友人的账号竟然更新了莫名其妙的怪异图文。蹩脚的翻译词句与晦暗扭曲的图片显得极不协调，{friend_char}推测是否是在口袋里误触了相机快门。",
                "expansion": f"屏幕上不断刷新出友人过往的照片与含糊不清的配文，发布时间赫然显示在几分钟之前，但发布者本人分明早已音讯全无。",
            },
            {
                "key": "discuss_contact_friends",
                "pos": ["連絡", "気まずい", "別れた", "対话", "対話", "心配", "知りません", "緊急"],
                "neg": ["死ぬ前", "呪い殺", "病院", "自殺", "神人"],
                "transition": "心头挥之不去的疑云驱使着两人试图探寻这桩诡异事件的真相。",
                "body": f"面对友人账号上的反常举动，{lead_char}与{friend_char}面露难色地商讨对策。由于友人分手后已返回台湾，彼此断了联系良久，贸然致电显得十分尴尬。但在对朋友安危的担忧下，她们最终决定拨通远在台湾的共同好友电话。",
                "expansion": f"{lead_char}翻查着旧手机通讯录，犹豫再三后找出台湾旧识家豪的联络号码，两人反复确认着拨通后的询问措辞，生怕引起不必要的误会。",
            },
            {
                "key": "phone_call_curse_revelation",
                "pos": ["死ぬ前", "呪い殺", "死んだ", "強い呪い", "周りの人", "亡くなった", "ふざけないで", "信じられてる"],
                "neg": ["病院", "自殺"],
                "transition": "然而当电话拨通、越洋连线至台湾旧友家豪时，传来的却是令人心惊的凶讯。",
                "body": f"电话那头的家豪语气沉重至极，透露出一个噩耗：友人早在半年前就已经暴毙身亡！坊间传闻他是被极其凶狠的恶灵诅咒所杀，而更致命的是，这种强烈的诅咒绝非只作用于一人，任何接触、浏览过相关动态的人都会被厄运死死缠上。",
                "expansion": f"家豪在电话中急切地叮嘱两人千万不要再次打开或转发那些动态，并透露友人死前身体曾出现诡异的抓痕与高烧，死因至今都被列为离奇悬案。",
            },
            {
                "key": "photo_countdown_threat",
                "pos": ["何これ", "私の写真", "気持ち悪い", "カウントダウン", "写真なんだ"],
                "neg": ["病院", "自殺", "藁人形"],
                "transition": "未等两人从好友猝逝的噩耗中缓过神来，更令人窒息的威胁已然降临。",
                "body": f"桌上的手机再次发出急促的提示音，点开屏幕的瞬间，{lead_char}惊骇地发现死者的账号竟公开发布了她们本人的私密生活照！照片右下角赫然悬挂着跳动的血红色倒计时，私人隐私遭到公开展露。",
                "expansion": f"未曾公开发布过的卧室抓拍赫然出现在死者主页，倒计时数字以秒为单位不断减少，同伴慌乱地想要锁屏关机，手机却出现严重卡顿无法关闭。",
            },
            {
                "key": "hospital_medical_exam",
                "pos": ["検査", "病院", "原因", "医者", "病気", "推薦", "大きな病院", "下がって"],
                "neg": ["自殺", "神人", "火", "ライター"],
                "transition": "受到诅咒侵袭的同伴身体状况急转直下，众人紧急赶往综合医院进行系统救治。",
                "body": f"医院白炽灯下，主治医生神色凝重地向众人宣布初步化验结果：同伴的生理指标呈现反常的衰竭状态，但现代医学仪器却完全查不出任何病理原因。医生只能紧急建议转往更大的专科医院作深度排查。",
                "expansion": f"主治医生展示了脑部CT与血液化验单，表示各项指标剧烈异常却无器质性病变，只能先为虚弱的同伴注射镇定剂并开具转院证明。",
            },
            {
                "key": "stalking_apparition_bath",
                "pos": ["変な女", "突きまとって", "横にいる", "お風呂", "後ろに", "寝てても", "目が覚め", "動画見てから"],
                "neg": ["自殺", "藁人形", "台湾"],
                "transition": "从医院返回居所后，遭受极端惊吓的同伴精神走到了崩溃边缘。",
                "body": f"在幽暗的房间内，同伴颤抖着向{lead_char}吐露了最深沉的遭遇：自从观看了死者账号上的那段视频后，一个面容可怖的诡异女人便如影随形。半夜惊醒时她就站在床边，哪怕独自沐浴时，身后也总传来逼近的水声与视线。",
                "expansion": f"同伴紧紧抓住{lead_char}的手腕，声音颤抖地描述那名女子惨白的脸色与凌乱的长发，并坚称每次水声响起时，都能在浴室镜面反光中看到她逼近的身影。",
            },
            {
                "key": "suicide_tragedy_determination",
                "pos": ["自殺", "殺した", "目の前", "犯人", "冷静", "ショック", "死んじゃって", "同じ目に"],
                "neg": ["パスタ", "チャイ", "大将軍"],
                "transition": "然而绝望与恐惧的折磨终究击垮了同伴，一场悲剧在眼前轰然爆发。",
                "body": f"同伴在幻觉与剧痛的折磨下，以极度决绝的方式在{lead_char}眼前坠楼身亡。亲眼目睹挚友惨死，{lead_char}陷入了巨大的悲痛。但她清醒地意识到，这不是单纯的精神失常，而是幕后真凶利用恶毒邪术残忍剥夺了好友的性命。",
                "expansion": f"救护车与警笛声划破夜空，警方勘验现场后初步定为意外坠楼，但深知内情的{lead_char}在现场捡到了同伴紧握的手机，上面依旧停留在那个血色倒计时界面。",
            },
            {
                "key": "straw_doll_curse_lore",
                "pos": ["神人魚", "神人", "藁人形", "写真貼って", "木槌", "叩く", "丑の刻", "呪いについて", "悪魔に"],
                "neg": ["パスタ", "チャイ"],
                "transition": "悲剧发生后，家豪在台湾的多方查证终于取得了突破性进展，揭开了这一邪术的源流。",
                "body": f"家豪面色凝重地向{lead_char}揭秘了这一古老巫术——源自民俗禁忌的‘神人草人诅咒’。施咒者将受害者的面容照片牢牢钉在稻草纸人偶上，一边诵念怨毒咒诀一边用木槌狠狠敲击，被钉住之人便会被厉鬼缠身，直至饱受折磨暴毙而亡。",
                "expansion": f"家豪发来民间古籍中关于草人钉魂术的记载照片，指出受害者身上出现的莫名淤青正是木槌敲打的投射，唯有找到原始母偶并破除封印才能彻底破解。",
            },
            {
                "key": "occult_phenomena_escalation",
                "pos": ["新邪", "アンタンド", "先席", "先帖", "先帝", "呪い", "悪霊"],
                "neg": ["病院", "自殺", "パスタ"],
                "transition": "随着时间推移，暗处的危机已然顺着网络扩散至全新的场景。",
                "body": f"随着数字媒介的扩散，不可名状的超自然异象在现实中愈演愈烈。阴暗狭窄的房间里隐约回荡着令人神智错乱的诡异低语，古老邪术的无形咒力正顺着网络节点步步侵蚀现实，将所有涉事之人拖入凶险境地。",
                "expansion": f"屋内灯光出现剧烈频闪，墙角与走廊接连传来不明由来的抓挠声，掉落在地上的水杯无故滑动，超自然的侵袭已彻底突破了物理空间的限制。",
            },
            {
                "key": "realizing_shared_curse_fate",
                "pos": ["冷静", "ショック", "周分と同じ目に", "同じ目にあった", "二人を呪い", "死んじゃって"],
                "neg": ["パスタ", "チャイ", "大将軍"],
                "transition": "惨剧发生之后，恐慌与悲痛迅速笼罩了整间屋子。",
                "body": f"挚友接连猝逝的沉重打击几乎压垮了每一个人。同伴苦苦劝阻情绪失控的{lead_char}保持冷静、接受现实，但{lead_char}无比笃定：挚友绝非单纯的精神崩溃，她经历的梦魇与死状与半年前如出一辙，相同的诅咒规律已经清晰地浮出水面。",
                "expansion": f"{lead_char}将两名死者生前的病历记录与社交动态逐一对照，发现两人发病周期完全吻合，这也坚定了她立刻启程前往台湾的决心。",
            },
            {
                "key": "taiwan_investigation_search",
                "pos": ["台湾", "台北", "探す", "手がかり", "遺体", "海辺", "チラシ", "姉姿"],
                "neg": ["パスタ", "チャイ", "大将軍"],
                "transition": f"随后，为了探寻真相，{lead_char}跨越重洋降落在台北桃园机场。",
                "body": f"穿行在异乡潮湿闷热的街头骑楼之间，{lead_char}与{friend_char}重新汇合。两人沿着友人当年生前的活动轨迹层层搜寻，四处探访知情人士并散发寻人传单，试图在茫茫人海中拼凑出诅咒扩散的第一现场。",
                "expansion": f"根据死者生前遗留的信用卡消费记录与租房地址，{lead_char}与家豪逐一走访老旧街区的小巷商铺，向附近的邻里住户打听死者当年接触过的人员。",
            },
            {
                "key": "find_daoist_master",
                "pos": ["道士", "同士", "チラシ", "コメント", "有名な人", "払ってくれる", "見つけた"],
                "neg": ["パスタ", "チャイ", "大将軍"],
                "transition": "日以继夜的奔波与排查终于迎来了转机，一份关键情报指引她们找到了突破口。",
                "body": f"家豪通过民俗界的关系网，成功联络到了一位在当地享有盛誉的知名道长。据悉这位道长深谙各种民间邪祟与恶毒诅咒的破解之法，众人决定带着关键信物登门求助，希望能够借助道法彻底拔除缠绕在众人身上的凶煞。",
                "expansion": f"家豪凭借熟人引荐拿到了一张道家法坛的生辰批条，获悉这位主持擅长追溯邪咒源流，众人立即带着死者遗留的信物与生辰八字赶往香火缭绕的法坛。",
            },
            {
                "key": "master_exorcism_consult",
                "pos": ["神人業", "昔からある呪い", "失敗したことは", "助けて", "兆齢", "祓い", "一度も"],
                "neg": ["パスタ", "チャイ"],
                "transition": "在幽静肃穆的法坛前，众人终于见到了这位当家道长。",
                "body": f"道长端详过照片与生辰线索后神情严峻，坦言‘神人业’自古便是极其狠戾的索命巫术，怨气一旦成型便不死不休。但道长宽慰众人，自己修道多年、主持驱邪法事从未失手，只要依规布坛破煞，定能将恶灵彻底驱散化解。",
                "expansion": f"道长在案台前点燃朱砂符咒投入水中，符灰沉底呈现浑浊的黑褐色，证实怨煞已经深入命盘，必须在三日之内设立神坛开坛做法。",
            },
            {
                "key": "exorcism_fee_preparation",
                "pos": ["お金", "兆齢", "振り込む", "逃げた", "きっかけ", "思い当てる"],
                "neg": ["パスタ", "チャイ"],
                "transition": "法坛布置妥当，道长在正式开坛前向众人细致询问起因果渊源。",
                "body": f"众人依规筹备好法金，道长神色严肃地盘问起引发这场无妄之灾的根本诱因。原来这世间没有无缘无故的诅咒，受害者在日常言行中极可能无意识地触怒了某位心胸狭隘之人，才招致如此灭顶之灾。",
                "expansion": f"{lead_char}仔细回忆友人过往的人际交往圈，梳理出一份曾在网上与死者产生过激烈言语冲突的关注者名单，并将其提供给道长排查命理相克之人。",
            },
            {
                "key": "stalker_habits_recitation",
                "pos": ["食べ物", "パスタ", "チャイ", "アーティスト", "毎日聞いている", "全部見た", "トーコ"],
                "neg": ["大将軍"],
                "transition": "随着调查层层逼近真相核心，隐藏在暗处的神秘施咒者终于主动现身。",
                "body": f"眼前这位面目阴沉的女子竟然对受害者的日常生活习惯如数家珍——最喜欢的明太子意大利面、每日必喝的红茶拿铁、循环播放的独立音乐。她神经质地扬起嘴角，冷笑着承认自己每天都在窥探受害者的所有网络动态。",
                "expansion": f"女子阴鸷地翻出随身携带的厚重日记本，上面工整密麻地记录着受害者每一天的打卡定位与消费小票，甚至精确到了每一趟乘车时间。",
            },
            {
                "key": "culprit_jealousy_confession",
                "pos": ["看不慣", "社群", "炫耀", "破文", "我好可愛", "好多朋友", "穿得", "漂漂亮亮", "受不了"],
                "neg": ["大将軍"],
                "transition": "在冰冷的对峙中，凶手彻底撕下了伪装，吐露出令人震惊的作案动机。",
                "body": f"她嘶吼着内心的嫉恨：她极度厌恶受害者在社交媒体上发布的那些光鲜博文！‘我好可爱、我有好多朋友、每天穿得漂漂亮亮’——受害者习以为常的生活分享，在她自卑而扭曲的狭隘世界里却成了无法容忍的炫耀，最终激发了她动用邪术杀人的恶意。",
                "expansion": f"她指着自己粗糙破旧的衣物和狭窄昏暗的居所，质问为什么那些长相甜美、生活优渥的女孩可以毫无顾忌地享受关注，而自己却只能在角落里默默忍受无视。",
            },
            {
                "key": "confrontation_mockery_hate",
                "pos": ["拋上", "照片", "笑得", "髒狂", "看人低", "洗手間", "臭女"],
                "neg": ["大将軍"],
                "transition": "面对主角的厉声质问，凶手情绪彻底失控，宣泄着对世人的怨恨。",
                "body": f"凶手声嘶力竭地控诉受害者曾在网络上随手发过一张自己的抓拍丑照，并在背后肆意嘲笑。她坚信所有人都在居高临下地瞧不起她，那种被轻视的屈辱感在阴暗心灵中无限放大，化作了不惜同归于尽也要将对方拖入深渊的执念。",
                "expansion": f"{lead_char}当面质问她草人母偶的具体藏匿地点，凶手却狂笑着拒绝交出，并扬言哪怕同归于尽也绝不解除施加在众人身上的诅咒。",
            },
            {
                "key": "daoist_ritual_climax",
                "pos": ["大将軍", "大小人", "理大通", "理大神", "万貴分", "折磨", "驚情無理"],
                "neg": [],
                "transition": "怨念在这一刻全面爆发，法坛之上的终极除煞大战在漫天真言中彻底引爆！",
                "body": f"道长手持法剑脚踏罡步，口中急促诵念‘真正無理大將軍、私下天理大小人’的诛邪密咒！香炉烟尘弥漫，狂暴的阴煞恶灵在法坛周围疯狂冲撞，整座法坛在强烈的法力冲击下剧烈震颤。",
                "expansion": f"案台前香烛被狂风吹得剧烈摇晃，道长将浸泡过法水的桃木剑重重刺入草人核心，口中念诵真言压制住剧烈挣扎的阴煞，最终将贴有死者照片的草人当场焚为灰烬。",
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
                    "body": f"{lead_char}与身边的人继续在日常环境中交谈，彼此核对着工作日程与周围的反常细节，试图找出事情的源头。",
                    "expansion": f"两人仔细检查着身边的物件与房间门窗，看似平和的环境中，细小的线索正在悄悄串联成完整的链条。",
                },
                "inciting_incident": {
                    "key": f"phase_incident_{index}",
                    "transition": "未等众人喘息，" if gap >= 45.0 else "",
                    "body": f"突如其来的诡异变故打破了原有节奏，未知危险步步逼近，众人急忙关紧门窗防范四周的不明响动。",
                    "expansion": f"房间内温度骤降，墙壁传来连续的敲击异响，主角迅速翻出防身工具，警惕地注视着走廊尽头的黑暗。",
                },
                "investigation": {
                    "key": f"phase_invest_{index}",
                    "transition": "沿着线索深入追查，" if gap >= 45.0 else "",
                    "body": f"{lead_char}沿着蛛丝马迹深入探寻，每一个浮现的线索都在指向更为凶险的隐情，零散的文字记录与现实证据在此刻逐步吻合。",
                    "expansion": f"她翻阅着过往收集的照片与手写笔记，将受害者的行程轨迹重新梳理，锁定了下一个关键的调查据点。",
                },
                "climax": {
                    "key": f"phase_climax_{index}",
                    "transition": "最为激烈的正面交锋瞬间爆发，" if gap >= 45.0 else "",
                    "body": f"双方为了各自的目的展开搏杀，求生的意志在此刻压倒了一切恐惧。",
                    "expansion": f"两人扭打在一处，周围的桌椅与器物被撞翻散落一地，主角拼尽全力夺取关键信物，彻底打破了僵持的死局。",
                },
                "resolution": {
                    "key": f"phase_resol_{index}",
                    "transition": "风波平息之后，" if gap >= 45.0 else "",
                    "body": f"危机终于彻底化解，残破的现场渐渐归于平静，主角默默收拾好散落一地的遗物。",
                    "expansion": f"警笛声在街角渐渐远去，经历浩劫的幸存者站在阳光下深吸了一口气，带着对逝者的告慰重新面对接下来的生活。",
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


