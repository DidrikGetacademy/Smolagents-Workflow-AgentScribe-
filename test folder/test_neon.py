
import gc
from neon.log import log
from neon.Modes import modes
from neon.utils import _normalize_roman_unicode, _replace_number_word, Text_symbols, _build_grouped_sequence, add_subtitles_to_frames,_project_centers_from_base,_evenly_spaced_centers


def _auto_adjust_counts_for_pauses(words_raw, start_idx, base_counts, avg_gap):
    """
    Auto-adjusterer lane counts basert på pause-deteksjon mellom ord.

    Args:
        words_raw: Liste med ord-objekter
        start_idx: Startindeks i words_raw
        base_counts: Original counts fra mode config
        avg_gap: Gjennomsnittlig gap mellom ord

    Returns:
        Justerte counts dict
    """
    # Threshold for å detektere pauser - justert for å fange 0.18s pauser
    # For disse dataene er 0.18s en naturlig pause å bryte på
    pause_threshold = 0.17  # Fast threshold som fanger 0.18s gaps

    remaining_words = len(words_raw) - start_idx
    if remaining_words <= 0:
        return {ln: 0 for ln in base_counts.keys()}

    # Start med original counts
    adjusted_counts = dict(base_counts)

    # Finn største lane med count > 1 (vanligvis middle lane)
    main_lane = None
    max_count = 0
    for ln, count in base_counts.items():
        if count > max_count:
            max_count = count
            main_lane = ln

    if main_lane is None or max_count <= 1:
        return adjusted_counts

    # Analyser gaps for de neste ordene som ville bli brukt
    total_needed = sum(base_counts.values())
    words_to_check = words_raw[start_idx:start_idx + min(total_needed, remaining_words)]

    if len(words_to_check) < 2:
        return adjusted_counts

    # Sjekk om det er en stor pause innenfor denne gruppen
    has_significant_pause = False
    pause_position = -1

    # Debug: log gaps for this chunk
    gap_info = []
    for i in range(len(words_to_check) - 1):
        current_end = float(words_to_check[i].get("rel_end", 0))
        next_start = float(words_to_check[i + 1].get("rel_start", 0))
        gap = max(0.0, next_start - current_end)
        w1 = words_to_check[i].get("text", "")
        w2 = words_to_check[i + 1].get("text", "")
        gap_info.append(f"{w1}->{w2}:{gap:.2f}s")

        if gap >= pause_threshold:
            has_significant_pause = True
            pause_position = i
            break

    log(f"[pause_detect] threshold={pause_threshold:.2f}s gaps=[{', '.join(gap_info)}] pause_found={has_significant_pause} at_pos={pause_position}")    # Hvis vi finner en betydelig pause, reduser count for hovedlinjen
    if has_significant_pause and pause_position >= 0:
        # Reduser main_lane count til posisjonen før pausen + 1
        words_before_pause = pause_position + 1
        if words_before_pause < adjusted_counts[main_lane]:
            adjusted_counts[main_lane] = max(1, words_before_pause)
            log(f"[pause_detect] Found pause after word {pause_position}, reduced {main_lane} count from {base_counts[main_lane]} to {adjusted_counts[main_lane]}")

    return adjusted_counts


def _adjust_gap_for_word_length(word1, word2, base_gap):
    """
    Adjust gap based on combined word length to maintain visual consistency.

    Args:
        word1: First word text
        word2: Second word text
        base_gap: Original gap ratio from config

    Returns:
        Adjusted gap ratio
    """
    combined_length = len(word1) + len(word2)

    # Shorter word pairs need smaller gaps, longer pairs can handle larger gaps
    if combined_length <= 8:  # Short pairs like "being busy" (5+4=9, but "we all"=5)
        adjusted = base_gap * 0.7  # Reduce gap by 30%
        log(f"[gap_adjust] SHORT pair '{word1}+{word2}' len={combined_length} gap={base_gap:.3f}→{adjusted:.3f}")
        return adjusted
    elif combined_length <= 12:  # Medium pairs
        log(f"[gap_adjust] MED pair '{word1}+{word2}' len={combined_length} gap={base_gap:.3f} (unchanged)")
        return base_gap
    else:  # Long pairs like "psychology known" (10+5=15)
        adjusted = base_gap * 1.5  # Increase gap by 50%
        log(f"[gap_adjust] LONG pair '{word1}+{word2}' len={combined_length} gap={base_gap:.3f}→{adjusted:.3f}")
        return adjusted


def build_random_mode_pairs_from_words(frames, subtitle_text, duration, fps, mode_name: str | None = None):
    """Bygger stiliserte undertekst overlay-par fra ordnivå tidsstempler med forhåndsdefinerte moduser.

    Args:
        frames: Liste med video-rammer
        subtitle_text: Undertekst med ord-tidsstempler
        duration: Varighet i sekunder
        fps: Bilder per sekund
        mode_name: Spesifikk modus å bruke (valgfri)

    Returns:
        Liste med rammer med påført undertekst
    """
    import random

    # Safety checks
    if not frames or not isinstance(frames, list):
        log("[build_pairs] No frames or invalid frames list; returning input")
        return frames
    if not subtitle_text:
        log("[build_pairs] No subtitle_text provided; returning frames unchanged")
        return frames

    try:
        log(f"[build_pairs] frames={len(frames)} duration={duration:.2f}s fps={fps} forced_mode={mode_name if mode_name in modes else None} words_in={len(subtitle_text) if isinstance(subtitle_text, list) else 'str'}")
    except Exception:
        pass

    # Normalize input format (expect list of {word,start,end})
    words_raw = []
    if isinstance(subtitle_text, list):
        for w in subtitle_text:
            if isinstance(w, dict) and "word" in w and "start" in w and "end" in w:
                txt = str(w.get("word", "")).strip()
                if not txt:
                    continue
                try:
                    s = float(w.get("start", 0.0))
                    e = float(w.get("end", s))
                    if e < s:
                        e = s
                    words_raw.append({"text": txt, "start": s, "end": e})
                except Exception:
                    continue
    else:
        tokens = str(subtitle_text).strip().split()
        if not tokens:
            return frames
        step = max(0.08, duration / max(1, len(tokens)))
        t = 0.0
        for tok in tokens:
            s, e = t, min(duration, t + step * 0.9)
            words_raw.append({"text": tok, "start": s, "end": e})
            t += step

    if not words_raw:
        log("[build_pairs] No usable words after normalization")
        return frames


    first_start = min(w["start"] for w in words_raw)
    for w in words_raw:
        w["rel_start"] = max(0.0, float(w["start"]) - first_start)
        w["rel_end"] = max(w["rel_start"], float(w["end"]) - first_start)
    for w in words_raw:
        w["rel_start"] = min(duration, w["rel_start"])
        w["rel_end"] = min(duration, w["rel_end"]) if duration is not None else w["rel_end"]

    log(f"[build_pairs] first_start(orig)={first_start:.2f}s rel_range={words_raw[0]['rel_start']:.2f}-{words_raw[-1]['rel_end']:.2f} n_words={len(words_raw)}")


    gaps = []
    for i in range(len(words_raw) - 1):
        gaps.append(max(0.0, words_raw[i + 1]["rel_start"] - words_raw[i]["rel_start"]))
    avg_gap = (sum(gaps) / len(gaps)) if gaps else 0.25
    log(f"[build_pairs] avg_gap={avg_gap:.3f}s")

    mode_names = list(modes.keys())
    weights = []

    weights.append(2.0)

    rng = random.Random(int((duration or 0) * 1000) + len(words_raw))

    def _make_group(mode_name: str, lane_name: str, lane_cfg: dict, global_cfg: dict, words_texts: list[str]):
        """Kopierer linje/global konfigurasjon til en gruppe-dict forventet av _build_grouped_sequence.

        Args:
            mode_name: Navn på modusen
            lane_name: Navn på linjen (top/middle/bottom)
            lane_cfg: Linjekonfigurasjon
            global_cfg: Global konfigurasjon
            words_texts: Liste med ord-tekster

        Returns:
            Gruppeobjekt eller None hvis ingen ord
        """
        if not words_texts:
            return None
        # 1) Convert number words ("one".."ten") → ASCII Roman sequences ("I".."X")
        # 2) Normalize any Unicode Number Form numerals (e.g., "Ⅳ") → ASCII ("IV")
        transformed_words = [_normalize_roman_unicode(_replace_number_word(w)) for w in words_texts]
        try:
            lane_declared_single = int(lane_cfg.get("count", 0)) == 1
        except Exception:
            lane_declared_single = False
        if transformed_words and (lane_declared_single or len(transformed_words) == 1):
            sym = random.choice(Text_symbols)
            transformed_words[0] = f"{sym} {transformed_words[0]} {sym}"
        # Require explicit color configuration
        if "glow_color" not in lane_cfg and "color" not in lane_cfg:
            raise KeyError("'glow_color' or 'color' missing in lane_cfg – must be defined explicitly!")
        if "letter_color" not in lane_cfg:
            raise KeyError("'letter_color' missing in lane_cfg – must be defined explicitly!")
        if "stroke_1_color" not in lane_cfg:
            raise KeyError("'stroke_1_color' missing in lane_cfg – must be defined explicitly!")
        if "stroke_2_color" not in lane_cfg:
            raise KeyError("'stroke_2_color' missing in lane_cfg – must be defined explicitly!")

        resolved_glow = lane_cfg.get("glow_color", lane_cfg["color"])
        resolved_fill = lane_cfg["letter_color"]
        resolved_stroke1 = lane_cfg["stroke_1_color"]
        resolved_stroke2 = lane_cfg["stroke_2_color"]

        # Keep original y_factor (0.59) but center single words horizontally
        y_pos = lane_cfg.get("y_factor")
        x_left = lane_cfg.get("x_left")
        x_right = lane_cfg.get("x_right")
        inner_pad = global_cfg.get("inner_pad")

        # For single words in middle lane, center horizontally by setting tight x bounds
        if len(transformed_words) == 1 and lane_name == "middle":
            # Center the single word by creating a narrow centered band
            center_width = 0.4  # 40% of screen width centered
            x_left = 0.5 - center_width/2  # 0.3
            x_right = 0.5 + center_width/2  # 0.7
            inner_pad = 0.0  # No additional padding for single words

        # Apply dynamic gap adjustment for 2-word pairs
        gap_by_count_config = lane_cfg.get("gap_by_count", {})
        if len(transformed_words) == 2 and 2 in gap_by_count_config:
            original_gap = gap_by_count_config[2]
            adjusted_gap = _adjust_gap_for_word_length(transformed_words[0], transformed_words[1], original_gap)
            gap_by_count_config = dict(gap_by_count_config)  # Copy to avoid modifying original
            gap_by_count_config[2] = adjusted_gap

        # Special font for single words (pause-detected words)
        font_choice = lane_cfg.get("font_pick")
        if len(transformed_words) == 1 and lane_name == "middle":
            font_choice = "Chrusty Rock"
            log(f"[single_word_font] Using 'Chrusty Rock' for single word: '{transformed_words[0]}'")

        g = {
            "mode": mode_name,
            "lane": lane_name,
            "words": transformed_words,
            "font_size": lane_cfg.get("font_size"),
            "letter_spacing": lane_cfg.get("letter_spacing"),
            "y_factor": y_pos,
            "style": global_cfg.get("style", "cinematic_soft"),
            "glow_gain": global_cfg.get("glow_gain"),
            "x_left": x_left,
            "x_right": x_right,
            "inner_pad": inner_pad,
            "pair_gap_ratio": global_cfg.get("pair_gap_ratio_default"),
            "gap_by_count": gap_by_count_config,
            "slot_ratio": lane_cfg.get("slot_ratio"),  # optional override
            "font": font_choice,
            "glow_color": resolved_glow,
            "fill_color": resolved_fill,
            "stroke_1_color": resolved_stroke1,
            "stroke_2_color": resolved_stroke2,
            "inner_boldness": float(lane_cfg.get("inner_boldness", 0.0)),
            "uniform_all_lanes": bool(global_cfg.get("uniform_size_all_lanes", False)),
        }
        if lane_name == "middle" and "char_by_char" in lane_cfg:
            g["middle_char_by_char"] = bool(lane_cfg.get("char_by_char"))
            g["middle_char_stagger"] = lane_cfg.get("middle_char_stagger", 0.08)
        return g

    frame_h, frame_w = frames[0].shape[:2]
    all_pairs = []

    idx = 0
    while idx < len(words_raw):
        # Pick mode: testing override if valid, else random choice
        chosen_mode = mode_name if (mode_name and mode_name in modes) else rng.choices(mode_names, weights=weights, k=1)[0]
        mode_cfg = modes[chosen_mode]
        global_cfg = mode_cfg.get("global", {})
        lanes_cfg = mode_cfg.get("lanes", {})
        draw_order = list(global_cfg.get("draw_order", ["top", "middle", "bottom"]))

        # Original counts from mode configuration
        base_counts = {ln: max(0, int(cfg.get("count", 0))) for ln, cfg in lanes_cfg.items()}

        # Auto-adjust counts based on pause detection
        counts = _auto_adjust_counts_for_pauses(words_raw, idx, base_counts, avg_gap)

        total_needed = sum(counts.values())
        if total_needed <= 0:
            log(f"[build_pairs] Mode {chosen_mode} has zero total count; stopping")
            break

        remaining_total = len(words_raw) - idx
        if remaining_total < total_needed:
            used = 0
            for ln in ("top", "middle", "bottom"):
                c = counts.get(ln, 0)
                take = min(c, max(0, remaining_total - used))
                counts[ln] = take
                used += take

        # Log både original og justerte counts
        orig_str = f"orig=top:{base_counts.get('top',0)} mid:{base_counts.get('middle',0)} bot:{base_counts.get('bottom',0)}"
        adj_str = f"adj=top:{counts.get('top',0)} mid:{counts.get('middle',0)} bot:{counts.get('bottom',0)}"
        log(f"[build_pairs] choose_mode={chosen_mode} want={total_needed} remaining={remaining_total} {orig_str} {adj_str}")

        # Create lane groups and collect words in builder order: top -> middle -> bottom
        groups_this = []
        words_for_pairs_this = []
        lane_words_map = {"top": [], "middle": [], "bottom": []}
        start_idx_snapshot = idx
        for lane_name in ("top", "middle", "bottom"):
            c = counts.get(lane_name, 0)
            if c <= 0:
                continue
            take_slice = words_raw[idx: idx + c]
            if not take_slice:
                continue
            words_for_pairs_this.extend(take_slice)
            lane_cfg = lanes_cfg.get(lane_name, {})
            g = _make_group(
                chosen_mode,
                lane_name,
                lane_cfg,
                global_cfg,
                [w["text"] for w in take_slice]
            )
            if g is not None:
                groups_this.append(g)
                lane_words_map[lane_name] = list(g.get("words", []))
            idx += len(take_slice)
        consumed = idx - start_idx_snapshot
        if not groups_this or consumed == 0:
            log("[build_pairs] No groups formed in this iteration; stopping")
            break

        # Log preview-style summary of this chunk (words + centers)
        try:
            log(f"[preview] mode={chosen_mode} expect top:{counts.get('top',0)} mid:{counts.get('middle',0)} bot:{counts.get('bottom',0)} total={sum(counts.values())}")
            # Compute centers for preview logging (mirrors _build_grouped_sequence)
            shared_column_alignment = bool(global_cfg.get("shared_column_alignment", False))
            base_centers = None
            if shared_column_alignment:
                base_n = max(len(lane_words_map.get(ln) or []) for ln in ("top","middle","bottom"))
                ref_ln = None
                for ln in ("top","middle","bottom"):
                    if len(lane_words_map.get(ln) or []) == base_n and (ref_ln is None or ln == "middle"):
                        ref_ln = ln
                if ref_ln and base_n > 0:
                    ref_cfg = lanes_cfg.get(ref_ln, {})
                    base_centers = _evenly_spaced_centers(
                        base_n,
                        ref_cfg.get("x_left", 0.0),
                        ref_cfg.get("x_right", 1.0),
                        global_cfg.get("inner_pad", 0.06),
                        global_cfg.get("pair_gap_ratio_default", 0.25),
                        ref_cfg.get("gap_by_count"),
                    )
            for ln in ("bottom","middle","top"):
                lw = lane_words_map.get(ln) or []
                if not lw:
                    continue
                cfg = lanes_cfg.get(ln, {})
                if base_centers is not None:
                    centers = _project_centers_from_base(base_centers, len(lw))
                else:
                    centers = _evenly_spaced_centers(
                        len(lw),
                        cfg.get("x_left", 0.0),
                        cfg.get("x_right", 1.0),
                        global_cfg.get("inner_pad", 0.06),
                        global_cfg.get("pair_gap_ratio_default", 0.25),
                        cfg.get("gap_by_count"),
                    )
                log(f"[preview] lane={ln} words={lw} centers={[round(c,3) for c in centers]} shared={bool(global_cfg.get('shared_column_alignment', False))}")
        except Exception:
            pass

        # Per-mode timing parameters (used as defaults; will realign to word times below)
        start_t = float(global_cfg.get("start_time", 0.0))
        per_word_stagger = float(global_cfg.get("per_word_stagger", 0.0))
        hold_after_last = float(global_cfg.get("hold_after_last", 0.0))
        overlap_next = float(global_cfg.get("overlap_next", 0.0))
        word_in = float(global_cfg.get("fade_in", 0.0))
        fade_out = float(global_cfg.get("fade_out", 0.0))
        middle_lane_cfg = lanes_cfg.get("middle", {})
        middle_char_stagger = float(middle_lane_cfg.get("middle_char_stagger", 0.08))

        pairs_chunk, _ = _build_grouped_sequence(
            groups_this,
            start_time=start_t,
            per_word_stagger=per_word_stagger,
            hold_after_last=hold_after_last,
            overlap_next=overlap_next,
            word_in=word_in,
            fade_out=fade_out,
            middle_char_stagger=middle_char_stagger,
            frame_width=frame_w,
            shared_column_alignment=bool(global_cfg.get("shared_column_alignment", False)),
        )
        log(f"[build_pairs] chunk_pairs={len(pairs_chunk)} consumed={consumed}")

        # Realign each pair (including char-by-char) to the actual word timings
        # We tagged pairs with a per-chunk word index (widx) so we can group them safely
        by_widx = {}
        for p in pairs_chunk:
            wi = int(p.get("widx", -1))
            by_widx.setdefault(wi, []).append(p)
        disable_fades = bool(global_cfg.get("no_per_word_fades", False))
        for local_idx, word_info in enumerate(words_for_pairs_this):
            if local_idx not in by_widx:
                continue
            s = max(0.0, min(duration, word_info["rel_start"]))
            e = max(s, min(duration, word_info["rel_end"]))
            per_list = by_widx[local_idx]
            per_list.sort(key=lambda q: q.get("start", 0.0))
            planned_first = per_list[0].get("start", s)
            shift = planned_first - s
            span = max(0.01, e - s)
            # Derive fade-in/out from the word span unless disabled for this mode
            fade_in_word = 0.0 if disable_fades else max(0.04, min(0.30, span * 0.22))
            fade_out_word = fade_in_word
            for k, p in enumerate(per_list):
                # Shift start into alignment with real word start
                p["start"] = max(0.0, p.get("start", s) - shift)
                if bool(p.get("dynamic_duration", True)):
                    # Hard clamp duration to the spoken span for this word.
                    p["duration"] = max(0.02, e - p["start"])
                    p["fade_in"] = fade_in_word
                    p["fade_out"] = fade_out_word
                if disable_fades:
                    p["alpha_static"] = True
                if duration is not None:
                    # Ensure we never exceed clip boundary but do not extend beyond the word.
                    p["duration"] = min(p["duration"], max(0.0, duration - p["start"]))

        # Compute helpers used by both policies
        # Map lane -> list of local word indices in this chunk, in order
        lane_word_local = {"top": [], "middle": [], "bottom": []}
        ptr = 0
        for ln in ("top","middle","bottom"):
            c = counts.get(ln, 0)
            if c > 0:
                lane_word_local[ln] = list(range(ptr, ptr + c))
                ptr += c
        # Chunk end time = latest rel_end among words in this chunk
        chunk_end_t = 0.0
        for w in words_for_pairs_this:
            chunk_end_t = max(chunk_end_t, float(w.get("rel_end", 0.0)))

        # Extend words within the same lane to end together at that lane's last word end,
        # but do not spill into the next chunk. This makes, e.g., 2-word middle lane hold
        # the first word visible until the second word's end.
        # Compute per-lane last end within this chunk
        lane_last_end = {}
        for ln, idx_list in lane_word_local.items():
            if not idx_list:
                continue
            lane_last_end[ln] = max(float(words_for_pairs_this[j]["rel_end"]) for j in idx_list)

        # Optional: synchronize pairs per lane so both words fade in/out together.
        # Controlled by mode global flag `fadein_out_together`.
        try:
            if bool(global_cfg.get("fadein_out_together", False)):
                for ln, idx_list in lane_word_local.items():
                    if len(idx_list) >= 2:  # Handle pairs or more words in a lane
                        # Get timing info for all words in this lane
                        word_timings = []
                        for i in idx_list:
                            s = float(words_for_pairs_this[i].get("rel_start", 0.0))
                            e = float(words_for_pairs_this[i].get("rel_end", s))
                            word_timings.append((i, s, e))

                        # Find the earliest start and latest end for the pair/group
                        pair_start = min(s for _, s, _ in word_timings)
                        pair_end = max(e for _, _, e in word_timings)
                        pair_span = max(0.01, pair_end - pair_start)

                        # Calculate fade duration based on pair span
                        fade_duration = 0.0 if disable_fades else max(0.04, min(0.30, pair_span * 0.22))

                        # Apply synchronized timing to all words in this lane
                        for p in pairs_chunk:
                            if p.get("lane", ln) == ln and int(p.get("widx", -1)) in idx_list:
                                p["start"] = pair_start
                                if bool(p.get("dynamic_duration", True)):
                                    p["duration"] = max(0.02, pair_end - pair_start)
                                p["fade_in"] = fade_duration
                                p["fade_out"] = fade_duration
                                if disable_fades:
                                    p["alpha_static"] = True
        except Exception:
            pass

        # Next chunk start (to avoid overlap)
        next_chunk_start = None
        if idx < len(words_raw):
            try:
                next_chunk_start = float(words_raw[idx]["rel_start"])  # relative timeline
            except Exception:
                next_chunk_start = None
        elif duration is not None:
            next_chunk_start = float(duration)

        for p in pairs_chunk:
            if not bool(p.get("dynamic_duration", True)):
                continue
            ln = p.get("lane", "middle")
            lane_end_target = lane_last_end.get(ln, chunk_end_t)
            # Extend to lane end
            desired = max(0.02, lane_end_target - float(p["start"]))
            # Clamp to next chunk start if available
            if next_chunk_start is not None:
                desired = min(desired, max(0.0, next_chunk_start - float(p["start"])) )
            # Also clamp to clip duration if provided
            if duration is not None:
                desired = min(desired, max(0.0, float(duration) - float(p["start"])) )
            p["duration"] = desired

        # Tag rendering layer based on draw_order
        layer_map = {ln: i for i, ln in enumerate(draw_order)}
        for p in pairs_chunk:
            p["layer"] = layer_map.get(p.get("lane", "middle"), 0)

        all_pairs.extend(pairs_chunk)

    # Ensure front-most lanes are drawn last for overlapping frames
    all_pairs.sort(key=lambda q: (q.get("layer", 0), q.get("start", 0.0)))
    log(f"[build_pairs] total_pairs={len(all_pairs)} -> render")
    return add_subtitles_to_frames(frames, all_pairs, fps)
























if __name__ == "__main__":
    # import cv2
    # from moviepy import VideoFileClip
    # import torch
    # import gc
    # gc.collect()
    # torch.cuda.empty_cache()

    # input_video = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\blender_test_output\enhanced_detail.png"
    # output_video = r""

    # subtitle_text = [{'word': 'we', 'start': 0.0, 'end': 0.16}, {'word': 'mistake', 'start': 0.16, 'end': 0.6}, {'word': 'being', 'start': 0.6, 'end': 1.0}, {'word': 'busy', 'start': 1.0, 'end': 1.5}, {'word': 'for', 'start': 1.5, 'end': 2.08}, {'word': 'being', 'start': 2.08, 'end': 2.38}, {'word': 'valuable', 'start': 2.38, 'end': 2.88}, {'word': 'this', 'start': 3.06, 'end': 3.58}, {'word': 'is', 'start': 3.58, 'end': 3.72}, {'word': 'something', 'start': 3.72, 'end': 3.98}, {'word': 'in', 'start': 3.98, 'end': 4.16}, {'word': 'psychology', 'start': 4.16, 'end': 4.72}, {'word': 'known', 'start': 4.72, 'end': 5.14}, {'word': 'as', 'start': 5.14, 'end': 5.5}, {'word': 'the', 'start': 5.5, 'end': 5.66}, {'word': 'effort', 'start': 5.66, 'end': 6.16}, {'word': 'heuristic', 'start': 6.16, 'end': 6.88}, {'word': 'we', 'start': 7.06, 'end': 7.46}, {'word': 'all', 'start': 7.46, 'end': 7.66}, {'word': 'know', 'start': 7.66, 'end': 7.82}, {'word': 'what', 'start': 7.82, 'end': 7.96}, {'word': 'it', 'start': 7.96, 'end': 8.06}, {'word': 'feels', 'start': 8.06, 'end': 8.32}, {'word': 'like', 'start': 8.32, 'end': 8.62}, {'word': 'we', 'start': 8.78, 'end': 9.18}, {'word': 'think', 'start': 9.18, 'end': 9.44}, {'word': 'if', 'start': 9.44, 'end': 9.6}, {'word': "we're", 'start': 9.6, 'end': 9.7}, {'word': 'working', 'start': 9.7, 'end': 10.06}, {'word': '12', 'start': 10.06, 'end': 10.34}, {'word': 'hours', 'start': 10.34, 'end': 10.66}, {'word': 'a', 'start': 10.66, 'end': 10.8}, {'word': 'day', 'start': 10.8, 'end': 11.02}, {'word': "we're", 'start': 11.16, 'end': 11.52}, {'word': 'winning', 'start': 11.52, 'end': 11.78}, {'word': "we're", 'start': 11.96, 'end': 12.08}, {'word': 'moving', 'start': 12.08, 'end': 12.32}, {'word': 'forward', 'start': 12.32, 'end': 12.68}]
    # clip_duration = 12.829999999999984
    # clip_fps = 29.97002997002997
    # mode_name = "0-2-0-word-by-word"

    # if mode_name not in modes:
    #     raise SystemExit(f"Mode '{mode_name}' not found. Available: {list(modes.keys())}")

    # # Load and extract frames from video
    # print(f"Loading video: {input_video}")
    # full_video = VideoFileClip(input_video)
    # start_time = 0.0
    # end_time = clip_duration
    # clip = full_video.subclipped(start_time, end_time)

    # frames = []
    # print("Extracting frames...")
    # for frame in clip.iter_frames():
    #     frames.append(frame)
    #     frame_height, frame_width = frame.shape[:2]




    # frames_bgr = []
    # for frame in frames:
    #     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     frames_bgr.append(frame_bgr)


    # print(f"Applying subtitles with mode: {mode_name}")
    # rendered_frames = build_random_mode_pairs_from_words(
    #     frames=frames_bgr,
    #     subtitle_text=subtitle_text,
    #     duration=clip_duration,
    #     fps=clip_fps,
    #     mode_name=mode_name,
    # )
    # frames_rgb = []
    # for frames in rendered_frames:
    #     frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
    #     frames_rgb.append(frames)
    # from moviepy import VideoFileClip, ImageSequenceClip,  CompositeVideoClip

    # try:
    #    processed_clip = ImageSequenceClip(frames_rgb, fps=clip.fps).with_duration(clip.duration)

    # except Exception as e:
    #      log(f"[processed_clip] ERROR: {str(e)}")
    # final_clip = CompositeVideoClip(
    #             [processed_clip.with_position('center')],
    #             size=processed_clip.size
    #             )
    # final_clip.audio = clip.audio

    # final_clip.write_videofile(
    #     output_video,
    #     logger='bar',
    #     codec="libx264",
    #     preset="fast",
    #     audio_codec="aac",
    #     threads=8,
    #     ffmpeg_params=[
    #         "-crf", "8",
    #         "-pix_fmt", "yuv420p",
    #         "-profile:v", "high",
    #         "-ar", "48000",
    #         "-color_primaries", "bt709",
    #         "-color_trc", "bt709",
    #         "-colorspace", "bt709",
    #         "-movflags", "+faststart",
    #     ],
    #     audio_bitrate="384k",
    #     remove_temp=True
    #         )


    # full_video.close()
    # clip.close()
    import cv2
    import torch
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    input_image = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\blender_test_output\enhanced_detail.png"
    output_image = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\blender_test_output\font_test_output.png"

    # Simple subtitle text with just "we" and "mistake" for font testing
    # Give them small durations so they're always visible on the static image
    subtitle_text = [
        {'word': 'we', 'start': 0.0, 'end': 1.0},
        {'word': 'mistake', 'start': 0.0, 'end': 1.0}
    ]
    clip_duration = 1.0
    clip_fps = 1
    mode_name = "0-2-0-word-by-word"

    if mode_name not in modes:
        raise SystemExit(f"Mode '{mode_name}' not found. Available: {list(modes.keys())}")

    # Load single image
    print(f"Loading image: {input_image}")
    img = cv2.imread(input_image)
    if img is None:
        raise SystemExit(f"Could not load image: {input_image}")

    # Convert to RGB (since frames are expected to be RGB in your pipeline)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_height, frame_width = img_rgb.shape[:2]

    # Create a single frame list
    frames = [img_rgb]

    print(f"Applying subtitles with mode: {mode_name}")
    print(f"Debug: Input frame shape: {frames[0].shape}")
    print(f"Debug: Input frame type: {frames[0].dtype}")

    rendered_frames = build_random_mode_pairs_from_words(
        frames=frames,
        subtitle_text=subtitle_text,
        duration=clip_duration,
        fps=clip_fps,
        mode_name=mode_name,
    )

    # Save the result
    if rendered_frames:
        print(f"Debug: Output frame shape: {rendered_frames[0].shape}")
        print(f"Debug: Frame changed: {not (frames[0] == rendered_frames[0]).all()}")

        # Convert back to BGR for saving with cv2
        output_frame_bgr = cv2.cvtColor(rendered_frames[0], cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image, output_frame_bgr)
        print(f"Successfully created image with subtitles: {output_image}")
        print(f"Image specs: {frame_width}x{frame_height}")

        # Also save the original for comparison
        original_bgr = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
        original_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\blender_test_output\original_for_comparison.png"
        cv2.imwrite(original_path, original_bgr)
        print(f"Original saved for comparison: {original_path}")
    else:
        print("No frames were rendered")
