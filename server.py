"""PDF Note Head Detector - Flask backend using OpenCV"""
import os, io, base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def estimate_staff_space_full_page(gray):
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj = np.sum(bw, axis=1) / 255
    threshold = np.max(proj) * 0.3
    staff_rows = np.where(proj > threshold)[0]
    if len(staff_rows) < 4:
        return 15
    groups = []
    cur = [staff_rows[0]]
    for i in range(1, len(staff_rows)):
        if staff_rows[i] - staff_rows[i-1] <= 2:
            cur.append(staff_rows[i])
        else:
            groups.append(cur)
            cur = [staff_rows[i]]
    groups.append(cur)
    line_centers = [int(np.mean(g)) for g in groups]
    gaps = np.diff(line_centers)
    small_gaps = gaps[(gaps > 3) & (gaps < 30)]
    if len(small_gaps) == 0:
        return 15
    small_gaps.sort()
    return int(np.median(small_gaps))


def remove_staff_lines(bw, staff_space):
    h, w = bw.shape
    cleaned = bw.copy()
    proj = np.sum(bw, axis=1) / 255
    if np.max(proj) == 0:
        return cleaned
    threshold = np.max(proj) * 0.3
    staff_rows = np.where(proj > threshold)[0]
    groups = []
    cur = [staff_rows[0]]
    for i in range(1, len(staff_rows)):
        if staff_rows[i] - staff_rows[i-1] <= 2:
            cur.append(staff_rows[i])
        else:
            groups.append(cur)
            cur = [staff_rows[i]]
    groups.append(cur)
    for group in groups:
        for r in range(min(group), max(group)+1):
            if r >= h:
                continue
            for c in range(w):
                if not cleaned[r, c]:
                    continue
                top, bot = r, r
                while top > 0 and cleaned[top-1, c]: top -= 1
                while bot < h-1 and cleaned[bot+1, c]: bot += 1
                if bot - top + 1 <= max(3, staff_space // 5):
                    cleaned[r, c] = 0
    return cleaned


def classify_contour(cnt, staff_space, region_width, cleaned_img=None, cleaned_preclose=None):
    area = cv2.contourArea(cnt)
    x, y, bw_c, bh_c = cv2.boundingRect(cnt)
    aspect = bw_c / max(bh_c, 1)
    ns = staff_space

    if bw_c < ns * 0.3 or bh_c < ns * 0.3:
        return False, 'tiny'
    if bw_c > ns * 3.5 or bh_c > ns * 5.0:
        return False, 'toobig'

    min_area = (ns * 0.2) ** 2
    max_area = (ns * 3.0) ** 2
    if area < min_area:
        return False, f'small({area:.0f}<{min_area:.0f})'
    if area > max_area:
        return False, f'big({area:.0f}>{max_area:.0f})'

    if aspect > 2.5 or aspect < 0.25:
        return False, f'aspect({aspect:.2f})'

    # Skip if in the leftmost few staff-spaces AND very tall (clef/brace span multiple systems)
    left_margin = staff_space * 4
    if x < left_margin and bh_c > staff_space * 5:
        return False, f'left-clef'
        return False, f'leftmargin(x={x}<{left_margin})'

    # Text/dynamic markings: narrower than note heads
    # Note heads are typically ~0.8-1.3 staff_space wide AND tall
    # Letters/words are either: narrow (< 0.7 ss) OR short (< 0.8 ss) with width < 1.5 ss
    if bw_c < ns * 0.7:
        return False, f'text-like(w={bw_c}<{ns*0.7:.0f})'
    if bh_c < ns * 0.8 and bw_c < ns * 1.5:
        return False, f'text-short(h={bh_c}<{ns*0.8:.0f})'

    # Circularity
    perimeter = cv2.arcLength(cnt, True)
    if perimeter < 1:
        return False, 'noperim'
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    if circularity < 0.1:
        # Low circularity could be a note head + stem + flag
        # Check if the blob has a dense region of note-head size
        mask = np.zeros_like(cleaned_img, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        if bw_c >= ns * 0.5 and bw_c <= ns * 2.5 and bh_c >= ns * 0.5 and bh_c <= ns * 5:
            # Check row density to find a note-head-like dense region
            roi = mask[y:y+bh_c, x:x+bw_c]
            row_sums = np.sum(roi > 0, axis=1)
            # A note head occupies ~1 staff_space vertically with high width
            window = max(int(ns * 0.8), 3)
            for row_start in range(0, max(1, bh_c - window + 1)):
                window_sum = np.mean(row_sums[row_start:row_start+window])
                if window_sum >= bw_c * 0.4 and bw_c >= ns * 0.5:
                    print(f'    stem_note: circ={circularity:.2f} but dense head region found at row {row_start}, width_fill={window_sum/bw_c:.2f}', flush=True)
                    # Accept as note with stem
                    return True, 'NOTE_WITH_STEM'
        return False, f'circ({circularity:.2f})'

    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    if hull_area < 1:
        return False, 'nohull'
    solidity = area / hull_area

    # ──── Hollow whole-note detection (before solidity/density filters) ────
    # Whole notes are open ovals: thin ring → very low density/solidity
    # but they are still valid note heads. Detect by shape.
    if solidity < 0.3 and circularity >= 0.35:
        ring_mask = np.zeros((bh_c + 2, bw_c + 2), np.uint8)
        cv2.drawContours(ring_mask, [cnt], -1, 255, -1)
        filled_area = cv2.countNonZero(ring_mask)
        if filled_area >= 1:
            ring_fill = area / filled_area
            if ring_fill < 0.55 and bw_c >= ns * 0.5 and bh_c >= ns * 0.4:
                if 0.5 <= aspect <= 2.0:
                    return True, 'WHOLE_NOTE'
        # Not a whole note → reject for low solidity
        return False, f'sol({solidity:.2f})'

    if solidity < 0.3:
        return False, f'sol({solidity:.2f})'

    density = area / (bw_c * bh_c)
    if density < 0.15:
        return False, f'dense({density:.2f})'
    # Very high density = likely text/solid shape, not note head
    # Note heads with stem have density ~0.2-0.5, text letters ~0.6-0.9
    # BUT: whole notes after close can have density ~0.77 too
    # Whole note exception: roughly elliptical, size ~1 staff_space
    if density > 0.75:
        # Check if it's a whole-note-sized ellipse
        if ns * 0.7 <= bw_c <= ns * 2.0 and ns * 0.6 <= bh_c <= ns * 1.5:
            if 0.5 <= aspect <= 2.0 and circularity >= 0.5:
                return True, 'WHOLE_NOTE(filled)'
        return False, f'text-dense({density:.2f})'

    # Rest detection via pre-close image: rests are sparse strokes
    # Check consecutive dense rows in the image BEFORE morphological close
    if cleaned_preclose is not None:
        roi_check = cleaned_preclose[y:y+bh_c, x:x+bw_c]
        if roi_check.size > 0:
            row_fills = np.sum(roi_check > 0, axis=1)
            max_consec = 0
            cur_consec = 0
            for rf in row_fills:
                if rf > bw_c * 0.4:
                    cur_consec += 1
                    if cur_consec > max_consec:
                        max_consec = cur_consec
                else:
                    cur_consec = 0
            if max_consec < ns * 0.25:
                return False, f'rest(consec={max_consec}<{ns*0.3:.0f})'

    return True, 'NOTE'


def find_note_head_position(cleaned, cnt, staff_space):
    """Find the note head center within a contour (head vs stem).
    Returns (cx, head_size) where cx,cy is head center and head_size is head diameter."""
    x, y, bw_c, bh_c = cv2.boundingRect(cnt)
    
    # If height is close to width, it's already just a head
    if bh_c <= bw_c * 1.5:
        # Use the widest row as center for better accuracy
        roi_h = cleaned[y:y+bh_c, x:x+bw_c]
        row_widths = np.sum(roi_h > 0, axis=1)
        max_row = np.argmax(row_widths)
        return x + bw_c/2, y + max_row + 0.5, max(bw_c, bh_c)
    
    # Find the row range with the highest density = the head
    mask = np.zeros_like(cleaned)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    # Also use the actual cleaned pixels
    roi = cleaned[y:y+bh_c, x:x+bw_c]
    
    # Sliding window of height ~ staff_space to find densest region
    window_h = max(staff_space, bw_c)
    best_y = y
    best_sum = 0
    for row_start in range(0, max(1, bh_c - window_h + 1)):
        row_end = row_start + window_h
        if row_end > bh_c:
            break
        s = np.sum(roi[row_start:row_end, :])
        if s > best_sum:
            best_sum = s
            best_y = row_start
    
    head_cy_density = y + best_y + window_h / 2
    
    # Find the row with maximum width (most dark pixels)
    row_sums = np.array([np.sum(roi[r, :] > 0) for r in range(bh_c)])
    max_row = np.argmax(row_sums)
    head_cy_width = y + max_row + 0.5
    
    # For notes with stems going UP, the head is at the BOTTOM of the blob
    # For notes with stems going DOWN, the head is at the TOP
    # Determine stem direction: if top of blob is narrow and bottom is wide -> stem up (head at bottom)
    # For tall blobs, use the contour's horizontal extent per row to find the head,
    # since hollow half notes have low density in cleaned image but full outline in contour.
    third = bh_c // 3
    top_avg = np.mean(row_sums[:third]) if third > 0 else 0
    bot_avg = np.mean(row_sums[2*third:]) if 2*third < bh_c else 0
    
    if bh_c > staff_space * 2.5:
        # Tall blob with stem
        if bot_avg > top_avg * 1.2:
            head_cy = y + bh_c - staff_space/2
        elif top_avg > bot_avg * 1.2:
            head_cy = y + staff_space/2
        else:
            head_cy = (head_cy_density + head_cy_width) / 2
    elif bot_avg > top_avg * 1.2:
        # Head is at bottom of blob (stem goes up)
        head_cy = y + bh_c - staff_space/2
    elif top_avg > bot_avg * 1.2:
        # Head is at top of blob (stem goes down)
        head_cy = y + staff_space/2
    else:
        # Ambiguous - use average of density and width
        head_cy = (head_cy_density + head_cy_width) / 2
    
    head_size = max(bw_c, min(bh_c, staff_space * 1.2))
    
    print(f'    head_detect: blob={bw_c}x{bh_c} at ({x},{y}), top_avg={top_avg:.1f} bot_avg={bot_avg:.1f} head_cy={head_cy:.1f} density_cy={head_cy_density:.1f} width_cy={head_cy_width:.1f}', flush=True)
    
    return x + bw_c/2, head_cy, head_size


def find_staff_lines_in_region(bw_img, staff_space):
    """Find staff line y-positions in a binary image region.
    Returns list of staff line y-positions."""
    h, w = bw_img.shape
    proj = np.sum(bw_img, axis=1) / 255
    if np.max(proj) == 0:
        return []
    threshold = np.max(proj) * 0.3
    staff_rows = np.where(proj > threshold)[0]
    if len(staff_rows) == 0:
        return []
    # Group consecutive rows into individual lines
    groups = []
    cur = [staff_rows[0]]
    for i in range(1, len(staff_rows)):
        if staff_rows[i] - staff_rows[i-1] <= 2:
            cur.append(staff_rows[i])
        else:
            groups.append(cur)
            cur = [staff_rows[i]]
    groups.append(cur)
    line_centers = [int(np.mean(g)) for g in groups]
    
    # Filter: find the best group of 5 evenly-spaced lines
    # Slurs/ties can be misdetected as extra lines
    if len(line_centers) > 5:
        best_group = None
        best_score = float('inf')
        from itertools import combinations
        for combo in combinations(range(len(line_centers)), 5):
            lines = [line_centers[i] for i in combo]
            gaps = [lines[i+1] - lines[i] for i in range(4)]
            if min(gaps) < staff_space * 0.5:
                continue  # Skip if any gap is too small
            # Score: variance of gaps (lower = more uniform)
            gap_var = np.var(gaps)
            if gap_var < best_score:
                best_score = gap_var
                best_group = lines
        if best_group:
            return best_group
    
    return line_centers


def find_nearest_staff_group(note_cy, staff_line_ys):
    """Find the 5-line staff group closest to the note's y position."""
    if len(staff_line_ys) < 5:
        return staff_line_ys
    
    # Group lines into sets of 5 separated by large gaps
    # Gap between systems is much larger than gap between lines in a system
    gaps = []
    for i in range(1, len(staff_line_ys)):
        gaps.append(staff_line_ys[i] - staff_line_ys[i-1])
    
    # Typical staff: 4 gaps of ~staff_space, then large gap to next system
    # Find natural breaks (gaps > median * 2)
    if not gaps:
        return staff_line_ys
    
    median_gap = sorted(gaps)[len(gaps)//2]
    break_threshold = max(median_gap * 2, staff_line_ys[-1] - staff_line_ys[0])  # large threshold
    
    # Split into groups at large gaps
    line_groups = []
    current = [staff_line_ys[0]]
    for i in range(1, len(staff_line_ys)):
        if staff_line_ys[i] - staff_line_ys[i-1] > break_threshold:
            line_groups.append(current)
            current = [staff_line_ys[i]]
        else:
            current.append(staff_line_ys[i])
    line_groups.append(current)
    
    # Find group closest to note
    best_group = line_groups[0]
    best_dist = float('inf')
    for g in line_groups:
        if len(g) < 5:
            continue
        # Distance from note to center of group
        center = (g[0] + g[-1]) / 2
        dist = abs(note_cy - center)
        if dist < best_dist:
            best_dist = dist
            best_group = g
    
    return best_group


def y_to_pitch(note_cy, staff_line_ys, staff_space, clef='bass', do_mode='fixed', do_key=1):
    """Map a y-coordinate to a pitch number.
    
    do_mode='fixed': Fixed do (固定调) - C is always 1
    do_mode='movable': Movable do (首调) - do_key is the keynote (1=C,2=D,...7=B)
    
    do_key: the note that becomes "1" in movable do
      1=C, 2=D, 3=E, 4=F, 5=G, 6=A, 7=B
    """
    nearest_group = find_nearest_staff_group(note_cy, staff_line_ys)
    if len(nearest_group) < 5:
        return '?'
    
    bottom_line_y = nearest_group[-1]
    # Use actual median gap from detected lines for more accurate pitch
    if len(nearest_group) >= 2:
        gaps = [nearest_group[i+1] - nearest_group[i] for i in range(len(nearest_group)-1)]
        actual_half = np.median(gaps) / 2.0
        if actual_half > 0:
            half_space = actual_half
        else:
            half_space = staff_space / 2.0
    else:
        half_space = staff_space / 2.0
    
    raw_pos = (bottom_line_y - note_cy) / half_space
    pos = round(raw_pos)
    
    if clef == 'treble':
        # Treble: bottom line = E
        # Map position to chromatic note: E=3, F=4, G=5, A=6, B=7, C=1, D=2
        fixed_notes = [3, 4, 5, 6, 7, 1, 2]
    else:
        # Bass: bottom line = G
        # G=5, A=6, B=7, C=1, D=2, E=3, F=4
        fixed_notes = [5, 6, 7, 1, 2, 3, 4]
    
    fixed_pitch = fixed_notes[pos % 7]
    
    if do_mode == 'movable' and do_key >= 1 and do_key <= 7:
        # Movable do: transpose so that do_key maps to 1
        # shift = (1 - do_key) mod 7, but we need to handle the cycle
        # e.g. if key=G(5): G->1, A->2, B->3, C->4, D->5, E->6, F->7
        # shift = (1 - 5) mod 7 = -4 mod 7 = 3
        shift = (1 - do_key) % 7
        movable_pitch = ((fixed_pitch - 1 + shift) % 7) + 1
        return str(movable_pitch)
    
    return str(fixed_pitch)


def process_regions(img_bytes, regions, annotate=False, calibration=None):
    print(f'Regions received: {regions}', flush=True)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return [] if not annotate else img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    staff_space = estimate_staff_space_full_page(gray)
    print(f'Image: {w}x{h}, staff_space={staff_space}', flush=True)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # If calibration provided, use it to validate staff line detection
    cal_line_gap = None
    if calibration and 'lineGap' in calibration:
        cal_line_gap = calibration['lineGap']
        print(f'Using calibration: line_gap={cal_line_gap}', flush=True)

    all_notes = []
    total_accepted = 0

    for region in regions:
        rx0 = max(0, int(region.get('x', 0) * w))
        rx1 = min(w, int((region.get('x', 0) + region.get('w', 1)) * w))
        y0 = max(0, int(region['y'] * h))
        y1 = min(h, int((region['y'] + region['h']) * h))
        rw = rx1 - rx0
        rh = y1 - y0
        if rh < 5 or rw < 5:
            continue

        roi_bw = bw[y0:y1, rx0:rx1]
        cleaned = remove_staff_lines(roi_bw, staff_space)
        # Also keep pre-close version for rest detection
        cleaned_preclose = cleaned.copy()

        ns = staff_space

        # ──── Detect hollow notes BEFORE morphological close ────
        # Close fills in hollow ovals, making them look like text
        preclose_contours, _ = cv2.findContours(cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        whole_note_rects = set()  # (x,y,w,h) of detected whole notes
        half_note_head_rects = {}  # (x,y,w,h) -> head_cy for half notes with stem
        print(f'  Preclose contours: {len(preclose_contours)}', flush=True)
        for wcnt in preclose_contours:
            wx, wy, ww, wh = cv2.boundingRect(wcnt)
            warea = cv2.contourArea(wcnt)
            print(f'  preclose raw: pos=({wx},{wy}) size=({ww}x{wh}) area={warea:.0f}', flush=True)
            if ww < ns * 0.3 or wh < ns * 0.3:
                continue
            wperim = cv2.arcLength(wcnt, True)
            if wperim < 1:
                continue
            wcirc = 4 * np.pi * warea / (wperim * wperim)
            ring_mask = np.zeros((wh + 2, ww + 2), np.uint8)
            # Shift contour to mask coordinates
            shifted = wcnt - np.array([[[wx, wy]]])
            cv2.drawContours(ring_mask, [shifted], -1, 255, -1)
            filled_area = cv2.countNonZero(ring_mask)
            if filled_area < 1:
                continue
            ring_fill = warea / filled_area
            waspect = ww / max(wh, 1)
            print(f'  preclose candidate: pos=({wx},{wy}) size=({ww}x{wh}) area={warea:.0f} circ={wcirc:.2f} ring_fill={ring_fill:.2f} aspect={waspect:.2f}', flush=True)
            is_hollow = ring_fill < 0.55 and wcirc >= 0.3 and 0.5 <= waspect <= 2.0
            if is_hollow:
                whole_note_rects.add((wx, wy, ww, wh))
                print(f'  -> WHOLE_NOTE (preclose)!', flush=True)
            elif wh > ns * 2.0 and waspect < 0.5:
                # Tall thin blob - could be half note (hollow head + stem)
                # Check the BOTTOM region of the blob for hollow head pattern
                roi_prec = cleaned_preclose[wy:wy+wh, wx:wx+ww]
                bot_start = max(0, wh - int(ns * 1.5))
                bot_roi = roi_prec[bot_start:, :]
                if bot_roi.size > 0:
                    # Count pixels in bottom region
                    bot_pixels = np.count_nonzero(bot_roi)
                    bot_area = bot_roi.size
                    bot_fill = bot_pixels / bot_area if bot_area > 0 else 1.0
                    # For a hollow half note head, bottom region has:
                    # - some pixels (outline) but fill < 0.4
                    # - the outline creates a distinctive pattern
                    print(f'  -> half_note_bot: bot_fill={bot_fill:.2f} bot_pixels={bot_pixels} bot_area={bot_area}', flush=True)
                    # Disabled: half note detection from preclose bottom fill
                    # This was incorrectly triggered for regular notes
                    # if bot_fill < 0.35:
                    #     head_cy_prec = wy + wh - ns
                    #     half_note_head_rects[(wx, wy, ww, wh)] = head_cy_prec
                    #     print(f'  -> HALF_NOTE_BOT head_cy={head_cy_prec:.1f}', flush=True)

        kern_size = max(2, staff_space // 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern_size, kern_size))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find staff lines in original (not cleaned) ROI for pitch detection
        staff_line_ys = find_staff_lines_in_region(roi_bw, staff_space)
        # If calibration exists, use it to select best 5 lines
        if cal_line_gap and len(staff_line_ys) >= 5:
            from itertools import combinations
            best_group = None
            best_score = float('inf')
            for combo in combinations(range(len(staff_line_ys)), 5):
                lines = [staff_line_ys[i] for i in combo]
                gaps = [lines[i+1] - lines[i] for i in range(4)]
                avg_g = sum(gaps) / 4
                gap_dev = abs(avg_g - cal_line_gap)
                gap_var = np.var(gaps)
                score = gap_dev * 10 + gap_var
                if score < best_score:
                    best_score = score
                    best_group = lines
            if best_group:
                staff_line_ys = best_group
        print(f'  Staff lines in region: {staff_line_ys}', flush=True)

        # Split large blobs that may be multiple notes connected by beams/ties
        all_contours = []
        for idx, cnt in enumerate(contours):
            cx2, cy2, cw2, ch2 = cv2.boundingRect(cnt)
            if cw2 > ns * 2.0 and ch2 > ns * 0.3 and ch2 < ns * 6:
                erode_sz = max(2, ns // 3)
                ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_sz, erode_sz))
                mask = np.zeros_like(cleaned)
                mask[cy2:cy2+ch2, cx2:cx2+cw2] = cleaned[cy2:cy2+ch2, cx2:cx2+cw2]
                eroded = cv2.erode(mask, ek, iterations=2)
                dilated = cv2.dilate(eroded, ek, iterations=2)
                sub_ct, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid = [sc for sc in sub_ct if cv2.contourArea(sc) > (ns * 0.2) ** 2]
                if len(valid) >= 2:
                    all_contours.extend(valid)
                    continue
            all_contours.append(cnt)

        for cnt in all_contours:
            x, y, bw_c, bh_c = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            # Check if this contour matches a preclose-detected whole note
            is_whole = False
            half_note_cy = None
            for (wx, wy, ww, wh) in whole_note_rects:
                margin = ns * 0.5
                if (abs(x - wx) < margin and abs(y - wy) < margin and
                    abs(bw_c - ww) < margin and abs(bh_c - wh) < margin):
                    is_whole = True
                    break
            if not is_whole:
                for (hx, hy, hw, hh), hcy in half_note_head_rects.items():
                    margin = ns * 0.5
                    if (abs(x - hx) < margin and abs(y - hy) < margin and
                        abs(bw_c - hw) < margin and abs(bh_c - hh) < margin):
                        half_note_cy = hcy
                        break

            if is_whole:
                reason = 'WHOLE_NOTE(matched)'
                ok = True
            else:
                ok, reason = classify_contour(cnt, staff_space, rw, cleaned, cleaned_preclose)
            cx = x + bw_c / 2 + rx0
            cy = y + bh_c / 2 + y0
            print(f'  contour: pos=({x},{y}) size=({bw_c}x{bh_c}) area={area:.0f} -> {reason}', flush=True)

            if ok:
                cx, cy, head_size = find_note_head_position(cleaned, cnt, staff_space)
                cx += rx0
                # Override head position for half notes detected in preclose
                if half_note_cy is not None:
                    cy = half_note_cy
                    print(f'    -> HALF_NOTE override head_cy={cy:.1f}', flush=True)
                cy_abs = cy + y0
                clef = region.get('clef', 'bass')
                pitch = y_to_pitch(cy_abs, [sly + y0 for sly in staff_line_ys], staff_space, clef, region.get('doMode','fixed'), region.get('doKey',1))
                print(f'    -> NOTE at ({cx:.0f},{cy_abs:.0f}) pitch={pitch} clef={clef}', flush=True)
                all_notes.append({'cx': float(cx), 'cy': float(cy_abs), 'size': float(head_size), 'pitch': pitch, 'clef': clef})

            if annotate:
                if ok:
                    total_accepted += 1
                    cx_a, cy_a, hs_a = find_note_head_position(cleaned, cnt, staff_space)
                    cx_a += rx0; cy_a += y0
                    half = hs_a / 2
                    clef = region.get('clef', 'bass')
                    pitch = y_to_pitch(cy_a, [sly + y0 for sly in staff_line_ys], staff_space, clef, region.get('doMode','fixed'), region.get('doKey',1))
                    if clef == 'treble':
                        color = (233, 144, 74)  # blue for treble (BGR)
                    else:
                        color = (96, 69, 233)  # red for bass (BGR)
                    cv2.rectangle(img, (int(cx_a-half), int(cy_a-half)), (int(cx_a+half), int(cy_a+half)), color, 2)
                    cv2.putText(img, pitch, (int(cx_a-half-8), int(cy_a+half*0.3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    color = (0, 180, 0)
                    cv2.rectangle(img, (x+rx0, y+y0), (x+bw_c+rx0, y+bh_c+y0), color, 2)
                    cv2.putText(img, reason, (x+rx0, y+y0-3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        if annotate:
            cv2.putText(img, f"ss={staff_space} accepted={total_accepted}",
                        (rx0, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    if annotate:
        return img
    return all_notes


@app.route('/calibrate', methods=['POST'])
def calibrate():
    data = request.json
    region = data.get('region')
    image_bytes = base64.b64decode(data.get('image', ''))
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None or not region:
        return jsonify({'success': False, 'error': 'invalid input'})
    
    h, w = img.shape
    x0, y0 = int(region['x'] * w), int(region['y'] * h)
    x1, y1 = int((region['x'] + region['w']) * w), int((region['y'] + region['h']) * h)
    roi = img[y0:y1, x0:x1]
    
    if roi.size == 0:
        return jsonify({'success': False, 'error': 'empty region'})
    
    # Estimate staff space in this region
    _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    staff_space = estimate_staff_space_full_page(roi)
    
    lines = find_staff_lines_in_region(bw, staff_space)
    print(f'Calibration: detected {len(lines)} lines: {lines}, staff_space={staff_space}', flush=True)
    
    if len(lines) != 5:
        return jsonify({'success': False, 'error': f'detected {len(lines)} lines (need 5)'})
    
    # Calculate average line gap
    gaps = [lines[i+1] - lines[i] for i in range(4)]
    avg_gap = sum(gaps) / len(gaps)
    
    return jsonify({
        'success': True,
        'staff_space': int(staff_space),
        'line_gap': avg_gap,
        'lines': lines
    })


@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    regions = data.get('regions', [])
    calibration = data.get('calibration')
    image_bytes = base64.b64decode(data.get('image', ''))
    notes = process_regions(image_bytes, regions, annotate=False, calibration=calibration)
    return jsonify({'notes': notes})


@app.route('/debug', methods=['POST'])
def debug():
    data = request.json
    regions = data.get('regions', [])
    calibration = data.get('calibration')
    image_bytes = base64.b64decode(data.get('image', ''))
    img = process_regions(image_bytes, regions, annotate=True, calibration=calibration)
    if img is None:
        return jsonify({'error': 'bad image'}), 400
    _, buf = cv2.imencode('.png', img)
    return send_file(io.BytesIO(buf.tobytes()), mimetype='image/png')


@app.route('/')
def index():
    return open(os.path.join(BASE_DIR, 'index.html')).read()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=False)
