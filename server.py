"""PDF Note Head Detector - Flask backend using OpenCV"""
import os, io, json, time, uuid, base64, logging
from collections import Counter
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from threading import Lock
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
EVENT_LOG_PATH = os.path.join(LOG_DIR, 'events.jsonl')
SERVER_LOG_PATH = os.path.join(LOG_DIR, 'server.log')
EVENT_LOG_LOCK = Lock()

os.makedirs(LOG_DIR, exist_ok=True)

LOGGER = logging.getLogger('note_marker')
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    handler = RotatingFileHandler(SERVER_LOG_PATH, maxBytes=2_000_000, backupCount=3, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    LOGGER.addHandler(handler)


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def sanitize_log_value(value, depth=0):
    if depth >= 4:
        return '<max-depth>'
    if isinstance(value, np.generic):
        return sanitize_log_value(value.item(), depth=depth)
    if isinstance(value, dict):
        items = list(value.items())
        sanitized = {str(k): sanitize_log_value(v, depth=depth + 1) for k, v in items[:40]}
        if len(items) > 40:
            sanitized['_truncated'] = len(items) - 40
        return sanitized
    if isinstance(value, (list, tuple)):
        sanitized = [sanitize_log_value(v, depth=depth + 1) for v in value[:40]]
        if len(value) > 40:
            sanitized.append(f'...({len(value) - 40} more)')
        return sanitized
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (int, bool)) or value is None:
        return value
    if isinstance(value, str):
        return value if len(value) <= 500 else value[:500] + '...'
    return str(value)


def log_event(source, event, level='INFO', **payload):
    level_name = str(level or 'INFO').upper()
    entry = {
        'ts': utc_now_iso(),
        'source': source,
        'event': event,
        'level': level_name,
    }
    entry.update({key: sanitize_log_value(value) for key, value in payload.items()})

    line = json.dumps(entry, ensure_ascii=False)
    with EVENT_LOG_LOCK:
        with open(EVENT_LOG_PATH, 'a', encoding='utf-8') as fp:
            fp.write(line + '\n')

    log_level = getattr(logging, level_name, logging.INFO)
    LOGGER.log(log_level, line)
    return entry


def request_meta():
    return {
        'path': request.path,
        'method': request.method,
        'remote_addr': request.headers.get('X-Forwarded-For', request.remote_addr),
        'user_agent': request.headers.get('User-Agent', ''),
    }


def current_session_id(data=None):
    if isinstance(data, dict):
        session_id = data.get('sessionId')
        if session_id:
            return session_id
    return request.headers.get('X-Session-Id') or request.headers.get('X-Session-ID')


def summarize_region(region):
    if not isinstance(region, dict):
        return {}
    summary = {}
    for key in ('x', 'y', 'w', 'h'):
        if key in region:
            try:
                summary[key] = round(float(region[key]), 4)
            except (TypeError, ValueError):
                summary[key] = region[key]
    for key in ('clef', 'doMode', 'doKey'):
        if key in region:
            summary[key] = region[key]
    return summary


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


def pick_staff_space(primary_space, fallback_space):
    """Choose a stable staff-space value, preferring primary when it agrees with fallback."""
    primary = int(primary_space) if primary_space else 0
    fallback = int(fallback_space) if fallback_space else 0
    if primary <= 0:
        return fallback or 15
    if fallback <= 0:
        return primary
    if primary < fallback * 0.6 or primary > fallback * 1.7:
        return fallback
    return int(round(np.median([primary, fallback])))


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


def is_compact_eighth_rest_shape(width, height, aspect, density, circularity, staff_space):
    """Detect compact eighth-rest glyphs that otherwise resemble small note blobs."""
    if not (staff_space * 0.75 <= width <= staff_space * 1.45):
        return False
    if not (staff_space * 1.55 <= height <= staff_space * 2.65):
        return False
    if not (0.35 <= aspect <= 0.78):
        return False
    if not (0.22 <= density <= 0.62):
        return False
    if circularity > 0.65:
        return False
    return True


def has_straight_stem_run(binary_img, x, y, width, height, staff_space):
    """Return True when a narrow contour contains a long vertical stem."""
    if binary_img is None:
        return False
    roi = binary_img[y:y+height, x:x+width]
    if roi.size == 0:
        return False

    mask = roi > 0
    min_run = max(int(staff_space * 2.05), int(height * 0.62), 3)
    long_col_count = 0
    max_run_seen = 0

    for col in range(mask.shape[1]):
        cur_run = 0
        col_max = 0
        for is_on in mask[:, col]:
            if is_on:
                cur_run += 1
                if cur_run > col_max:
                    col_max = cur_run
            else:
                cur_run = 0
        if col_max >= min_run:
            long_col_count += 1
        if col_max > max_run_seen:
            max_run_seen = col_max

    max_stem_cols = max(5, int(width * 0.25))
    return max_run_seen >= min_run and 0 < long_col_count <= max_stem_cols


def is_dotted_stemmed_note_shape(width, height, aspect, density, staff_space,
                                 x, y, cleaned_preclose=None):
    """Detect a stemmed note whose dot widens the contour enough to look rest-like."""
    if cleaned_preclose is None:
        return False
    if not (staff_space * 1.35 <= width <= staff_space * 2.85):
        return False
    if not (staff_space * 3.3 <= height <= staff_space * 4.8):
        return False
    if not (0.32 <= aspect <= 0.75):
        return False
    if not (0.11 <= density <= 0.34):
        return False
    if not has_straight_stem_run(cleaned_preclose, x, y, width, height, staff_space):
        return False

    roi = cleaned_preclose[y:y + height, x:x + width]
    if roi.size == 0:
        return False

    row_counts = np.sum(roi > 0, axis=1)
    dense_threshold = max(staff_space * 0.6, width * 0.28)
    dense_rows = np.where(row_counts >= dense_threshold)[0]
    if len(dense_rows) < max(5, int(round(staff_space * 0.28))):
        return False

    dense_span = int(dense_rows[-1] - dense_rows[0] + 1)
    if dense_span < staff_space * 0.45 or dense_span > staff_space * 1.8:
        return False

    return True


def is_bar_rest_shape(width, height, aspect, density, staff_space):
    """Detect whole/half-rest horizontal bars before they enter note fallback."""
    if not (staff_space * 0.95 <= width <= staff_space * 1.75):
        return False
    if not (staff_space * 0.38 <= height <= staff_space * 0.78):
        return False
    if not (1.55 <= aspect <= 3.1):
        return False
    if density < 0.45:
        return False
    return True


def is_preclose_stemmed_note_candidate(width, height, aspect, density, circularity,
                                       staff_space, x, y, cleaned_preclose=None):
    """Detect note-head+stem blobs before closing merges them with nearby slurs."""
    if cleaned_preclose is None:
        return False
    if not (staff_space * 1.18 <= width <= staff_space * 2.85):
        return False
    if not (staff_space * 2.65 <= height <= staff_space * 5.35):
        return False
    if not (0.26 <= aspect <= 0.82):
        return False
    if not (0.08 <= density <= 0.76):
        return False
    if is_quarter_rest_shape(width, height, aspect, density, circularity, staff_space):
        return False
    if is_compact_eighth_rest_shape(width, height, aspect, density, circularity, staff_space):
        return False
    if not has_straight_stem_run(cleaned_preclose, x, y, width, height, staff_space):
        return False
    return True


def is_quarter_rest_shape(width, height, aspect, density, circularity, staff_space,
                          cnt=None, cleaned_preclose=None):
    """Detect narrow quarter-rest glyphs before they are treated as stemmed notes."""
    if not (staff_space * 0.85 <= width <= staff_space * 1.35):
        return False
    if not (staff_space * 2.45 <= height <= staff_space * 3.35):
        return False
    if not (0.28 <= aspect <= 0.55):
        return False
    if not (0.38 <= density <= 0.72):
        return False
    if circularity > 0.90:
        return False

    return True


def is_flat_accidental_fragment_shape(width, height, aspect, density, staff_space):
    """Detect the lower bulb of a flat accidental after contour splitting."""
    if not (staff_space * 0.72 <= width <= staff_space * 1.05):
        return False
    if not (staff_space * 0.75 <= height <= staff_space * 1.25):
        return False
    if not (0.70 <= aspect <= 1.10):
        return False
    if not (0.55 <= density <= 0.76):
        return False
    return True


def is_bass_clef_fragment_shape(width, height, aspect, density, staff_space):
    """Detect dense narrow fragments from bass-clef curls that mimic a note head."""
    if not (staff_space * 0.68 <= width <= staff_space * 0.92):
        return False
    if not (staff_space * 1.25 <= height <= staff_space * 1.70):
        return False
    if not (0.42 <= aspect <= 0.68):
        return False
    if not (0.62 <= density <= 0.78):
        return False
    return True


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
        density = area / max(bw_c * bh_c, 1)
        if is_bar_rest_shape(bw_c, bh_c, aspect, density, ns):
            return False, f'rest-bar(w={bw_c},h={bh_c},asp={aspect:.2f},dense={density:.2f})'
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

    if is_compact_eighth_rest_shape(bw_c, bh_c, aspect, density, circularity, ns):
        return False, f'rest-eighth(w={bw_c},h={bh_c},asp={aspect:.2f})'
    if is_quarter_rest_shape(bw_c, bh_c, aspect, density, circularity, ns, cnt, cleaned_preclose):
        return False, f'rest-quarter(w={bw_c},h={bh_c},asp={aspect:.2f},dense={density:.2f})'
    if is_flat_accidental_fragment_shape(bw_c, bh_c, aspect, density, ns):
        return False, f'accidental-flat-fragment(w={bw_c},h={bh_c},asp={aspect:.2f},dense={density:.2f})'
    if is_bass_clef_fragment_shape(bw_c, bh_c, aspect, density, ns):
        return False, f'bass-clef-fragment(w={bw_c},h={bh_c},asp={aspect:.2f},dense={density:.2f})'
    if is_dotted_stemmed_note_shape(bw_c, bh_c, aspect, density, ns, x, y, cleaned_preclose):
        return True, 'NOTE_WITH_STEM_DOT'

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


def find_compact_head_candidate(binary_img, x, y, width, height, staff_space, label):
    """Find a note-head-sized compact blob inside a tall connected contour."""
    if binary_img is None:
        return None

    roi = binary_img[y:y+height, x:x+width]
    if roi.size == 0:
        return None

    ns = staff_space
    k = max(3, int(round(ns * 0.16)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)
    comp_contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for comp in comp_contours:
        cx0, cy0, cw, ch = cv2.boundingRect(comp)
        if cw < ns * 0.55 or ch < ns * 0.62:
            continue
        if cw > ns * 2.65 or ch > ns * 1.85:
            continue
        aspect = cw / max(ch, 1)
        if aspect < 0.55 or aspect > 2.75:
            continue

        comp_roi = opened[cy0:cy0+ch, cx0:cx0+cw]
        ink = cv2.countNonZero(comp_roi)
        fill = ink / max(cw * ch, 1)
        if fill < 0.18:
            continue

        max_col_run = 0
        for col in range(comp_roi.shape[1]):
            cur = 0
            for is_on in comp_roi[:, col] > 0:
                if is_on:
                    cur += 1
                    if cur > max_col_run:
                        max_col_run = cur
                else:
                    cur = 0

        # The target is a compact oval-ish head. Long stems and slurs may survive
        # opening, so down-rank components that are much more line-like.
        head_w_target = ns * 1.25
        head_h_target = ns * 1.05
        size_score = 1.0 / (1.0 + abs(cw - head_w_target) / max(ns, 1))
        height_score = 1.0 / (1.0 + abs(ch - head_h_target) / max(ns, 1))
        compact_score = min(fill, 0.85) + size_score + height_score
        if max_col_run > ns * 1.65 and cw < ns * 0.9:
            compact_score -= 1.0

        moments = cv2.moments(comp)
        if moments['m00']:
            local_cx = moments['m10'] / moments['m00']
            local_cy = moments['m01'] / moments['m00']
        else:
            local_cx = cx0 + cw / 2.0
            local_cy = cy0 + ch / 2.0

        candidates.append({
            'cx': x + float(local_cx),
            'cy': y + float(local_cy),
            'size': float(max(cw, ch, ns * 0.9)),
            'score': float(compact_score),
            'bbox': (cx0, cy0, cw, ch),
            'fill': float(fill),
            'kind': 'component',
        })

    scan_img = opened if cv2.countNonZero(opened) > 0 else roi
    window_h = max(int(round(ns * 1.12)), 5)
    window_h = min(window_h, height)
    for row_start in range(0, max(1, height - window_h + 1)):
        row_end = row_start + window_h
        band = scan_img[row_start:row_end, :]
        ys, xs = np.nonzero(band > 0)
        if len(xs) == 0:
            continue

        bx = int(xs.min())
        by = int(ys.min())
        bw = int(xs.max() - xs.min() + 1)
        bh = int(ys.max() - ys.min() + 1)
        if bw < ns * 0.65 or bh < ns * 0.55:
            continue
        if bw > ns * 2.75 or bh > ns * 1.65:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 0.55 or aspect > 2.85:
            continue

        tight_band = band[by:by+bh, bx:bx+bw]
        ink = cv2.countNonZero(tight_band)
        fill = ink / max(bw * bh, 1)
        if fill < 0.16:
            continue

        row_counts = np.sum(tight_band > 0, axis=1)
        max_row_width = float(np.max(row_counts)) if len(row_counts) else 0.0
        wide_threshold = max(ns * 0.45, bw * 0.34)
        max_wide_run = 0
        cur_wide_run = 0
        for row_count in row_counts:
            if row_count >= wide_threshold:
                cur_wide_run += 1
                if cur_wide_run > max_wide_run:
                    max_wide_run = cur_wide_run
            else:
                cur_wide_run = 0

        if max_row_width < ns * 0.72:
            continue
        if max_wide_run < ns * 0.24:
            continue

        weights = (tight_band > 0).astype(np.uint8)
        moments = cv2.moments(weights)
        if moments['m00']:
            local_cx = bx + moments['m10'] / moments['m00']
            local_cy = row_start + by + moments['m01'] / moments['m00']
        else:
            local_cx = bx + bw / 2.0
            local_cy = row_start + by + bh / 2.0

        head_w_target = ns * 1.25
        head_h_target = ns * 1.05
        size_score = 1.0 / (1.0 + abs(bw - head_w_target) / max(ns, 1))
        height_score = 1.0 / (1.0 + abs(bh - head_h_target) / max(ns, 1))
        density_score = min(fill / 0.48, 1.0)
        wide_score = min(max_wide_run / max(ns * 0.75, 1), 1.0)
        compact_score = density_score + wide_score + size_score + height_score

        candidates.append({
            'cx': x + float(local_cx),
            'cy': y + float(local_cy),
            'size': float(max(bw, bh, ns * 0.9)),
            'score': float(compact_score),
            'bbox': (bx, row_start + by, bw, bh),
            'fill': float(fill),
            'kind': 'window',
        })

    if not candidates:
        return None

    candidates.sort(key=lambda item: item['score'], reverse=True)
    best = candidates[0]
    bx, by, bw, bh = best['bbox']
    print(
        f"    head_candidate[{label}/{best['kind']}]: bbox=({bx},{by},{bw},{bh}) "
        f"cy={best['cy']:.1f} fill={best['fill']:.2f} score={best['score']:.2f}",
        flush=True
    )
    return best['cx'], best['cy'], best['size'], f'{label}_compact_head'


def find_note_head_position(cleaned, cnt, staff_space, cleaned_preclose=None):
    """Find the note head center within a contour (head vs stem).
    Returns (cx, cy, head_size, source) where cx,cy is head center."""
    x, y, bw_c, bh_c = cv2.boundingRect(cnt)
    
    # If height is close to width, it's already just a head
    if bh_c <= bw_c * 1.5:
        # Beamed eighth-note heads are often slanted, so the widest row sits
        # slightly above the visual center and can cross a pitch boundary.
        return x + bw_c/2, y + bh_c/2.0, max(bw_c, bh_c), 'compact_bbox_center'

    if bh_c > staff_space * 2.35:
        candidate = find_compact_head_candidate(
            cleaned_preclose, x, y, bw_c, bh_c, staff_space, 'preclose'
        )
        if candidate is None:
            candidate = find_compact_head_candidate(
                cleaned, x, y, bw_c, bh_c, staff_space, 'closed'
            )
        if candidate is not None:
            return candidate
    
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
    
    return x + bw_c/2, head_cy, head_size, 'contour'


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

    filtered = []
    min_sep = max(2, int(round(staff_space * 0.35)))
    for line_y in line_centers:
        if not filtered or line_y - filtered[-1] >= min_sep:
            filtered.append(line_y)
        else:
            filtered[-1] = int(round((filtered[-1] + line_y) / 2))

    return filtered


def select_best_five_lines(line_centers, expected_gap=None, anchor_y=None):
    """Pick the most staff-like consecutive window of five lines."""
    if len(line_centers) <= 5:
        return line_centers

    best_group = None
    best_score = float('inf')

    for start in range(0, len(line_centers) - 4):
        lines = line_centers[start:start + 5]
        gaps = np.diff(lines)
        if len(gaps) != 4:
            continue
        median_gap = float(np.median(gaps))
        if median_gap <= 0:
            continue
        if np.min(gaps) < median_gap * 0.55:
            continue

        score = float(np.var(gaps))
        if expected_gap and expected_gap > 0:
            score += abs(median_gap - expected_gap) * 8
        if anchor_y is not None:
            center = (lines[0] + lines[-1]) / 2.0
            score += abs(center - anchor_y) / max(expected_gap or median_gap, 1)

        if score < best_score:
            best_score = score
            best_group = lines

    return best_group or line_centers[:5]


def split_staff_line_groups(staff_line_ys, expected_gap=None):
    if not staff_line_ys:
        return []

    sorted_lines = sorted(staff_line_ys)
    if len(sorted_lines) < 2:
        return [sorted_lines]

    gaps = [sorted_lines[i + 1] - sorted_lines[i] for i in range(len(sorted_lines) - 1) if sorted_lines[i + 1] > sorted_lines[i]]
    median_gap = float(np.median(gaps)) if gaps else 0.0
    base_gap = float(expected_gap or median_gap or 0)
    break_threshold = max(base_gap * 1.8, 6)

    groups = [[sorted_lines[0]]]
    for line_y in sorted_lines[1:]:
        if line_y - groups[-1][-1] > break_threshold:
            groups.append([line_y])
        else:
            groups[-1].append(line_y)

    return groups


def find_nearest_staff_group(note_cy, staff_line_ys, expected_gap=None):
    """Find the 5-line staff group closest to the note's y position."""
    if len(staff_line_ys) < 5:
        return staff_line_ys

    line_groups = split_staff_line_groups(staff_line_ys, expected_gap=expected_gap)
    best_group = None
    best_dist = float('inf')

    for group in line_groups:
        candidate = select_best_five_lines(group, expected_gap=expected_gap, anchor_y=note_cy)
        if len(candidate) < 5:
            continue
        center = (candidate[0] + candidate[-1]) / 2
        dist = abs(note_cy - center)
        if dist < best_dist:
            best_dist = dist
            best_group = candidate

    if best_group:
        return best_group
    return select_best_five_lines(sorted(staff_line_ys), expected_gap=expected_gap, anchor_y=note_cy)


def choose_staff_lines_for_region(region_staff_lines, page_staff_lines, y0, y1, expected_gap=None):
    """Prefer full-page staff lines, because tight ROIs can mistake notes/ledger lines for staff lines."""
    if not page_staff_lines:
        return region_staff_lines, 'region'

    anchor_y = (y0 + y1) / 2.0
    page_group = find_nearest_staff_group(anchor_y, page_staff_lines, expected_gap=expected_gap)
    if len(page_group) >= 5:
        group_center = (page_group[0] + page_group[-1]) / 2.0
        max_dist = max((y1 - y0) * 1.2, (expected_gap or 0) * 6.0, 80.0)
        if abs(group_center - anchor_y) <= max_dist:
            return [int(round(line_y - y0)) for line_y in page_group], 'page'

    return region_staff_lines, 'region'


def y_to_pitch(note_cy, staff_line_ys, staff_space, clef='bass', do_mode='fixed', do_key=1, expected_gap=None):
    """Map a y-coordinate to a pitch number.
    
    do_mode='fixed': Fixed do (固定调) - C is always 1
    do_mode='movable': Movable do (首调) - do_key is the keynote (1=C,2=D,...7=B)
    
    do_key: the note that becomes "1" in movable do
      1=C, 2=D, 3=E, 4=F, 5=G, 6=A, 7=B
    """
    nearest_group = find_nearest_staff_group(note_cy, staff_line_ys, expected_gap=expected_gap)
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


def whole_note_bbox_to_note(x, y, bw_c, bh_c, rx0, y0, staff_line_ys, staff_space, clef, do_mode, do_key, expected_gap=None):
    cx = x + bw_c / 2 + rx0
    cy = y + bh_c / 2 + y0
    size = float(max(bw_c, bh_c))
    pitch = y_to_pitch(cy, [sly + y0 for sly in staff_line_ys], staff_space, clef, do_mode, do_key, expected_gap=expected_gap)
    return {
        'cx': float(cx),
        'cy': float(cy),
        'size': size,
        'pitch': pitch,
        'clef': clef,
        'head_source': 'preclose_round_fallback',
        'bbox': {'x': float(x + rx0), 'y': float(y + y0), 'w': float(bw_c), 'h': float(bh_c)},
    }


def preclose_stemmed_note_to_note(cleaned_preclose, cnt, rx0, y0, staff_line_ys,
                                  staff_space, clef, do_mode, do_key, expected_gap=None):
    x, y, bw_c, bh_c = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    cx, cy, head_size, head_method = find_note_head_position(
        cleaned_preclose, cnt, staff_space, cleaned_preclose
    )
    cx += rx0
    cy_abs = cy + y0
    bbox_abs = {'x': float(x + rx0), 'y': float(y + y0), 'w': float(bw_c), 'h': float(bh_c)}
    cy_abs, head_refinement = refine_stemmed_head_to_staff_line(
        cy_abs, bbox_abs, [sly + y0 for sly in staff_line_ys], staff_space, head_method
    )
    if head_refinement is None:
        cy_abs, head_refinement = refine_likely_half_note_to_staff_line(
            cy_abs, bbox_abs, area, [sly + y0 for sly in staff_line_ys],
            staff_space, head_method
        )
    pitch = y_to_pitch(
        cy_abs, [sly + y0 for sly in staff_line_ys], staff_space, clef,
        do_mode, do_key, expected_gap=expected_gap
    )
    note = {
        'cx': float(cx),
        'cy': float(cy_abs),
        'size': float(head_size),
        'pitch': pitch,
        'clef': clef,
        'head_source': 'preclose_stemmed_fallback',
        'head_method': head_method,
        'shape_density': round(float(area) / max(float(bw_c * bh_c), 1.0), 3),
        'bbox': bbox_abs,
    }
    if head_refinement:
        note['head_refinement'] = head_refinement
    return note


def refine_stemmed_head_to_staff_line(cy_abs, bbox_abs, staff_line_ys_abs, staff_space, head_source):
    """Recover a line-centered note head when staff-line removal left only its lower half."""
    if head_source != 'preclose_compact_head':
        return cy_abs, None
    if not staff_line_ys_abs:
        return cy_abs, None

    bbox_h = float(bbox_abs.get('h', 0))
    if bbox_h < staff_space * 3.0:
        return cy_abs, None

    bbox_y = float(bbox_abs.get('y', 0))
    best_line = None
    best_score = float('inf')
    for line_y in staff_line_ys_abs:
        top_gap = float(line_y) - bbox_y
        center_gap = float(cy_abs) - float(line_y)
        if not (staff_space * 0.25 <= top_gap <= staff_space * 0.68):
            continue
        if not (staff_space * 0.24 <= center_gap <= staff_space * 0.72):
            continue
        score = abs(top_gap - staff_space * 0.45) + abs(center_gap - staff_space * 0.5)
        if score < best_score:
            best_score = score
            best_line = float(line_y)

    if best_line is None:
        return cy_abs, None

    return best_line, {
        'type': 'staff_line_snap_after_line_removal',
        'from_cy': round(float(cy_abs), 2),
        'to_cy': round(best_line, 2),
    }


def refine_likely_half_note_to_staff_line(cy_abs, bbox_abs, contour_area,
                                          staff_line_ys_abs, staff_space,
                                          head_source, reason=''):
    """Stabilize hollow half-note heads whose compact center is pulled toward the line below."""
    if head_source != 'preclose_compact_head':
        return cy_abs, None
    if not staff_line_ys_abs:
        return cy_abs, None

    bbox_w = float(bbox_abs.get('w', 0))
    bbox_h = float(bbox_abs.get('h', 0))
    if not (staff_space * 1.85 <= bbox_w <= staff_space * 2.35):
        return cy_abs, None
    if not (staff_space * 3.75 <= bbox_h <= staff_space * 4.25):
        return cy_abs, None

    density = float(contour_area) / max(bbox_w * bbox_h, 1.0)
    if density < 0.27:
        return cy_abs, None
    if reason and reason not in {'NOTE', 'NOTE_WITH_STEM', 'NOTE_WITH_STEM_DOT'}:
        return cy_abs, None

    sorted_lines = sorted(float(line_y) for line_y in staff_line_ys_abs)
    nearest_index, nearest_line = min(
        enumerate(sorted_lines),
        key=lambda item: abs(item[1] - float(cy_abs))
    )
    distance = abs(nearest_line - float(cy_abs))
    if distance > staff_space * 0.32:
        return cy_abs, None
    if float(cy_abs) >= nearest_line:
        return cy_abs, None

    if nearest_index > 0:
        target_cy = (sorted_lines[nearest_index - 1] + nearest_line) / 2.0
    else:
        target_cy = nearest_line - staff_space / 2.0

    if abs(target_cy - float(cy_abs)) > staff_space * 0.42:
        return cy_abs, None

    return float(target_cy), {
        'type': 'half_note_space_above_line_snap',
        'from_cy': round(float(cy_abs), 2),
        'to_cy': round(float(target_cy), 2),
        'density': round(density, 3),
    }


def extract_tall_stack_notes(cleaned, cnt, rx0, y0, staff_line_ys, staff_space, clef, do_mode, do_key, expected_gap=None):
    """Extract multiple note heads from a tall narrow contour sharing the same x position."""
    x, y, bw_c, bh_c = cv2.boundingRect(cnt)
    ns = staff_space

    if not (ns * 0.75 <= bw_c <= ns * 2.45):
        return []
    if not (ns * 4.3 <= bh_c <= ns * 9.0):
        return []

    roi = cleaned[y:y + bh_c, x:x + bw_c]
    if roi.size == 0:
        return []

    row_counts = np.sum(roi > 0, axis=1).astype(float)
    if row_counts.size == 0 or np.max(row_counts) < ns * 0.45:
        return []

    smooth_window = max(3, int(round(ns * 0.18)))
    kernel = np.ones(smooth_window, dtype=float) / smooth_window
    smoothed = np.convolve(row_counts, kernel, mode='same')
    threshold = max(ns * 0.55, bw_c * 0.42)
    active_rows = np.where(smoothed >= threshold)[0]
    if len(active_rows) == 0:
        return []

    runs = []
    start = int(active_rows[0])
    prev = int(active_rows[0])
    for row in active_rows[1:]:
        row = int(row)
        if row - prev <= max(2, int(round(ns * 0.18))):
            prev = row
            continue
        runs.append((start, prev))
        start = prev = row
    runs.append((start, prev))

    candidates = []
    for run_start, run_end in runs:
        pad = max(2, int(round(ns * 0.25)))
        top = max(0, run_start - pad)
        bottom = min(bh_c, run_end + pad + 1)
        band = roi[top:bottom, :]
        ys, xs = np.nonzero(band > 0)
        if len(xs) == 0:
            continue

        band_w = int(xs.max() - xs.min() + 1)
        band_h = int(bottom - top)
        # Real stacked note heads both have note-head scale. Eighth-note flags can
        # create a second dense band, but it is noticeably narrower/shorter.
        if band_w < ns * 1.05:
            continue
        if band_h < ns * 0.90 or band_h > ns * 2.1:
            continue

        local_rows = np.arange(top, bottom)
        weights = row_counts[top:bottom]
        if np.sum(weights) > 0:
            cy_local = float(np.average(local_rows, weights=weights))
        else:
            cy_local = (top + bottom - 1) / 2.0

        cx_local = float((xs.min() + xs.max()) / 2.0)
        candidate = {
            'cx': x + cx_local + rx0,
            'cy': y + cy_local + y0,
            'size': float(max(band_w, min(band_h, ns * 1.4), ns * 1.05)),
            'bbox': {
                'x': float(x + xs.min() + rx0),
                'y': float(y + top + y0),
                'w': float(band_w),
                'h': float(band_h),
            },
            'score': float(np.max(smoothed[run_start:run_end + 1])),
        }
        candidates.append(candidate)

    candidates.sort(key=lambda item: item['cy'])
    merged = []
    for candidate in candidates:
        if merged and abs(candidate['cy'] - merged[-1]['cy']) < ns * 0.75:
            if candidate['score'] > merged[-1]['score']:
                merged[-1] = candidate
            continue
        merged.append(candidate)

    if len(merged) < 2:
        return []

    notes = []
    staff_abs = [sly + y0 for sly in staff_line_ys]
    for candidate in merged:
        pitch = y_to_pitch(candidate['cy'], staff_abs, ns, clef, do_mode, do_key, expected_gap=expected_gap)
        notes.append({
            'cx': float(candidate['cx']),
            'cy': float(candidate['cy']),
            'size': float(candidate['size']),
            'pitch': pitch,
            'clef': clef,
            'head_source': 'tall_stack_projection',
            'bbox': candidate['bbox'],
            'parent_bbox': {'x': float(x + rx0), 'y': float(y + y0), 'w': float(bw_c), 'h': float(bh_c)},
        })

    return notes


def extract_stacked_whole_notes(cnt, rx0, y0, staff_line_ys, staff_space, clef, do_mode, do_key, expected_gap=None):
    """Split two vertically merged whole-note heads into separate notes."""
    x, y, bw_c, bh_c = cv2.boundingRect(cnt)
    ns = staff_space
    aspect = bw_c / max(bh_c, 1)
    area = cv2.contourArea(cnt)
    density = area / max(bw_c * bh_c, 1)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 0.0 if perimeter < 1 else 4 * np.pi * area / (perimeter * perimeter)

    if not (ns * 1.35 <= bw_c <= ns * 2.35):
        return []
    if not (ns * 1.65 <= bh_c <= ns * 2.75):
        return []
    if not (0.65 <= aspect <= 1.35):
        return []
    if density < 0.68:
        return []
    if circularity < 0.45:
        return []

    staff_abs = [sly + y0 for sly in staff_line_ys]
    cx = x + bw_c / 2.0 + rx0
    head_h = min(max(ns * 1.05, bh_c * 0.56), ns * 1.45)
    top_cy = y + head_h / 2.0 + y0
    bottom_cy = y + bh_c - head_h / 2.0 + y0
    if bottom_cy - top_cy < ns * 0.65:
        return []

    notes = []
    for cy, name in ((top_cy, 'top'), (bottom_cy, 'bottom')):
        pitch = y_to_pitch(cy, staff_abs, ns, clef, do_mode, do_key, expected_gap=expected_gap)
        notes.append({
            'cx': float(cx),
            'cy': float(cy),
            'size': float(max(bw_c, head_h)),
            'pitch': pitch,
            'clef': clef,
            'head_source': 'stacked_whole_note',
            'bbox': {
                'x': float(x + rx0),
                'y': float((cy - head_h / 2.0)),
                'w': float(bw_c),
                'h': float(head_h),
            },
            'parent_bbox': {'x': float(x + rx0), 'y': float(y + y0), 'w': float(bw_c), 'h': float(bh_c)},
            'stack_position': name,
        })

    return notes


def bbox_intersection_area(a, b):
    ax0, ay0 = a.get('x', 0), a.get('y', 0)
    ax1, ay1 = ax0 + a.get('w', 0), ay0 + a.get('h', 0)
    bx0, by0 = b.get('x', 0), b.get('y', 0)
    bx1, by1 = bx0 + b.get('w', 0), by0 + b.get('h', 0)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return float((ix1 - ix0) * (iy1 - iy0))


def filter_region_note_artifacts(notes, staff_space):
    """Suppress duplicate note-head fragments and flag/stem pieces inside stemmed notes."""
    if len(notes) < 2:
        return notes, Counter()

    keep = [True] * len(notes)
    reasons = Counter()
    stack_sources = {
        'tall_stack_projection',
        'stacked_whole_note',
        'preclose_stacked_whole_fallback',
    }
    reliable_head_sources = stack_sources | {
        'preclose_compact_head',
        'compact_bbox_center',
        'preclose_round_candidate',
        'whole_note_bbox_center',
        'preclose_half_note',
        'contour',
    }

    def area(note):
        bbox = note.get('bbox') or {}
        return max(float(bbox.get('w', 0)) * float(bbox.get('h', 0)), 1.0)

    def source(note):
        return note.get('head_source') or ''

    def horizontal_overlap_ratio(a, b):
        ax0, ax1 = float(a.get('x', 0)), float(a.get('x', 0)) + float(a.get('w', 0))
        bx0, bx1 = float(b.get('x', 0)), float(b.get('x', 0)) + float(b.get('w', 0))
        overlap = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        return overlap / max(min(float(a.get('w', 0)), float(b.get('w', 0))), 1.0)

    def suppress(idx, reason):
        if keep[idx]:
            keep[idx] = False
            reasons[reason] += 1

    for i in range(len(notes)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(notes)):
            if not keep[j]:
                continue
            ni, nj = notes[i], notes[j]
            if ni.get('pitch') != nj.get('pitch'):
                continue
            if abs(ni.get('cx', 0) - nj.get('cx', 0)) > staff_space * 0.6:
                continue
            if abs(ni.get('cy', 0) - nj.get('cy', 0)) > staff_space * 0.6:
                continue
            remove_idx = i if area(ni) < area(nj) else j
            suppress(remove_idx, 'duplicate_same_head')

    # Beamed eighth notes often create a good primary head candidate plus a
    # later pre-close fallback from the same stem/beam blob. If another reliable
    # head is already sitting at that x-position, the fallback is a duplicate
    # even when its inferred pitch differs.
    for idx, note in enumerate(notes):
        if not keep[idx] or source(note) != 'preclose_stemmed_fallback':
            continue
        bbox = note.get('bbox') or {}
        for target_idx, target in enumerate(notes):
            if idx == target_idx or not keep[target_idx]:
                continue
            target_source = source(target)
            if target_source == 'preclose_stemmed_fallback' or target_source not in reliable_head_sources:
                continue
            target_bbox = target.get('bbox') or {}
            if abs(note.get('cx', 0) - target.get('cx', 0)) > staff_space * 0.55:
                continue
            if horizontal_overlap_ratio(bbox, target_bbox) < 0.55:
                continue
            smaller_area = min(area(note), area(target))
            if bbox_intersection_area(bbox, target_bbox) / smaller_area < 0.45:
                continue
            suppress(idx, 'stemmed_fallback_overlaps_existing_head')
            break

    # A compact contour inside the same tall stemmed-note bbox can be just a
    # piece of the already accepted head. Keep true stacked-note extractors, but
    # remove these near-center fragments before they become extra pitches.
    for idx, note in enumerate(notes):
        if not keep[idx] or source(note) != 'compact_bbox_center':
            continue
        bbox = note.get('bbox') or {}
        note_area = area(note)
        for parent_idx, parent in enumerate(notes):
            if idx == parent_idx or not keep[parent_idx]:
                continue
            if source(parent) != 'preclose_compact_head':
                continue
            parent_bbox = parent.get('bbox') or {}
            if parent_bbox.get('h', 0) < staff_space * 2.2:
                continue
            if area(parent) < note_area * 1.8:
                continue
            if abs(note.get('cx', 0) - parent.get('cx', 0)) > staff_space * 0.55:
                continue
            if abs(note.get('cy', 0) - parent.get('cy', 0)) > staff_space * 0.55:
                continue
            if horizontal_overlap_ratio(bbox, parent_bbox) < 0.6:
                continue
            if bbox_intersection_area(bbox, parent_bbox) / note_area < 0.65:
                continue
            suppress(idx, 'compact_fragment_inside_primary_head')
            break

    for idx, note in enumerate(notes):
        if not keep[idx]:
            continue
        bbox = note.get('bbox') or {}
        note_area = area(note)
        if bbox.get('w', 0) > staff_space * 1.05 or bbox.get('h', 0) > staff_space * 1.35:
            continue
        for target_idx, target in enumerate(notes):
            if idx == target_idx or not keep[target_idx]:
                continue
            target_bbox = target.get('bbox') or {}
            if target_bbox.get('x', 0) <= bbox.get('x', 0):
                continue
            gap = target_bbox.get('x', 0) - (bbox.get('x', 0) + bbox.get('w', 0))
            if gap < -staff_space * 0.15 or gap > staff_space * 0.75:
                continue
            dx = target.get('cx', 0) - note.get('cx', 0)
            if dx < staff_space * 0.85 or dx > staff_space * 2.0:
                continue
            if abs(target.get('cy', 0) - note.get('cy', 0)) > staff_space * 0.75:
                continue
            target_area = area(target)
            if note_area > target_area * 0.78:
                continue
            if bbox.get('w', 0) > target_bbox.get('w', 0) * 0.85:
                continue
            suppress(idx, 'accidental_left_of_note')
            break

    for idx, note in enumerate(notes):
        if not keep[idx]:
            continue
        bbox = note.get('bbox') or {}
        note_area = area(note)
        for parent_idx, parent in enumerate(notes):
            if idx == parent_idx or not keep[parent_idx]:
                continue
            parent_bbox = parent.get('bbox') or {}
            if parent_bbox.get('h', 0) < staff_space * 2.4:
                continue
            if bbox.get('w', 0) > staff_space * 1.6 or bbox.get('h', 0) > staff_space * 1.7:
                continue
            if abs(note.get('cx', 0) - parent.get('cx', 0)) > staff_space * 0.8:
                continue
            if note.get('cy', 0) - parent.get('cy', 0) < staff_space * 0.7:
                continue
            coverage = bbox_intersection_area(bbox, parent_bbox) / note_area
            if coverage < 0.65:
                continue
            suppress(idx, 'attached_flag_or_stem_fragment')
            break

    return [note for idx, note in enumerate(notes) if keep[idx]], reasons


def find_matching_round_note_rect(blob_rect, round_rects, staff_space):
    """Match a post-close contour to a pre-close round note-head candidate."""
    if not round_rects:
        return None

    bx, by, bw_c, bh_c = blob_rect
    blob_area = max(bw_c * bh_c, 1)
    blob_cx = bx + bw_c / 2.0
    blob_cy = by + bh_c / 2.0
    best_rect = None
    best_score = 0.0

    for rect in sorted(round_rects):
        rx, ry, rw_c, rh_c = rect
        rect_area = max(rw_c * rh_c, 1)
        ix0 = max(bx, rx)
        iy0 = max(by, ry)
        ix1 = min(bx + bw_c, rx + rw_c)
        iy1 = min(by + bh_c, ry + rh_c)
        if ix1 <= ix0 or iy1 <= iy0:
            continue

        inter = (ix1 - ix0) * (iy1 - iy0)
        rect_coverage = inter / rect_area
        blob_coverage = inter / blob_area
        rect_cx = rx + rw_c / 2.0
        rect_cy = ry + rh_c / 2.0
        dx = abs(blob_cx - rect_cx)
        dy = abs(blob_cy - rect_cy)
        center_limit_x = max(staff_space * 0.85, min(bw_c, rw_c) * 0.75)
        center_limit_y = max(staff_space * 0.85, min(bh_c, rh_c) * 0.75)

        if rect_coverage < 0.45 or blob_coverage < 0.12:
            continue
        if dx > center_limit_x or dy > center_limit_y:
            continue

        score = rect_coverage * 2.0 + blob_coverage - (dx + dy) / max(staff_space, 1) * 0.05
        if score > best_score:
            best_score = score
            best_rect = rect

    return best_rect


def process_regions(img_bytes, regions, annotate=False, calibration=None, request_id=None, session_id=None, endpoint='detect'):
    print(f'Regions received: {regions}', flush=True)
    process_start = time.perf_counter()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        log_event(
            'server', 'image_decode_failed', level='ERROR',
            request_id=request_id, session_id=session_id, endpoint=endpoint,
            region_count=len(regions)
        )
        return [] if not annotate else img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    page_staff_space = estimate_staff_space_full_page(gray)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # If calibration provided, use it to stabilize note sizing and staff-line selection.
    cal_staff_space = None
    cal_line_gap = None
    if calibration and 'staffSpace' in calibration:
        cal_staff_space = calibration['staffSpace']
    if calibration and 'lineGap' in calibration:
        cal_line_gap = calibration['lineGap']
    base_staff_space = pick_staff_space(cal_staff_space, page_staff_space)
    page_pitch_gap = cal_line_gap or base_staff_space
    page_staff_line_ys = find_staff_lines_in_region(bw, base_staff_space)
    print(
        f'Image: {w}x{h}, page_staff_space={page_staff_space}, '
        f'cal_staff_space={cal_staff_space}, cal_line_gap={cal_line_gap}, '
        f'base_staff_space={base_staff_space}, page_staff_lines={len(page_staff_line_ys)}',
        flush=True
    )

    all_notes = []
    total_accepted = 0

    for region_idx, region in enumerate(regions):
        region_start = time.perf_counter()
        rx0 = max(0, int(region.get('x', 0) * w))
        rx1 = min(w, int((region.get('x', 0) + region.get('w', 1)) * w))
        y0 = max(0, int(region['y'] * h))
        y1 = min(h, int((region['y'] + region['h']) * h))
        rw = rx1 - rx0
        rh = y1 - y0
        if rh < 5 or rw < 5:
            log_event(
                'server', 'region_skipped', level='WARNING',
                request_id=request_id, session_id=session_id, endpoint=endpoint,
                region_index=region_idx, region=summarize_region(region),
                reason='too_small', pixel_size={'w': rw, 'h': rh}
            )
            continue

        roi_gray = gray[y0:y1, rx0:rx1]
        roi_bw = bw[y0:y1, rx0:rx1]
        region_staff_space = pick_staff_space(estimate_staff_space_full_page(roi_gray), base_staff_space)
        cleaned = remove_staff_lines(roi_bw, region_staff_space)
        # Also keep pre-close version for rest detection
        cleaned_preclose = cleaned.copy()

        ns = region_staff_space

        # ──── Detect hollow notes BEFORE morphological close ────
        # Close fills in hollow ovals, making them look like text
        preclose_contours, _ = cv2.findContours(cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        whole_note_rects = set()  # (x,y,w,h) of detected whole notes
        preclose_round_note_rects = set()
        half_note_head_rects = {}  # (x,y,w,h) -> head_cy for half notes with stem
        preclose_stemmed_note_candidates = []
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
            bbox_fill = warea / max(ww * wh, 1)
            print(f'  preclose candidate: pos=({wx},{wy}) size=({ww}x{wh}) area={warea:.0f} circ={wcirc:.2f} ring_fill={ring_fill:.2f} aspect={waspect:.2f}', flush=True)
            if is_bar_rest_shape(ww, wh, waspect, bbox_fill, ns):
                print(
                    f'  -> REST_BAR skip fallback candidate (w={ww},h={wh},density={bbox_fill:.2f})',
                    flush=True
                )
                continue
            is_hollow = ring_fill < 0.55 and wcirc >= 0.3 and 0.5 <= waspect <= 2.0
            is_round_note_candidate = (
                ns * 0.9 <= ww <= ns * 2.4 and
                ns * 0.6 <= wh <= ns * 1.8 and
                0.45 <= waspect <= 2.4 and
                wcirc >= 0.45 and
                bbox_fill >= 0.45
            )
            if is_hollow:
                whole_note_rects.add((wx, wy, ww, wh))
                print(f'  -> WHOLE_NOTE (preclose)!', flush=True)
            if is_hollow or is_round_note_candidate:
                preclose_round_note_rects.add((wx, wy, ww, wh))
                if is_round_note_candidate and not is_hollow:
                    print(f'  -> ROUND_NOTE_CANDIDATE (preclose)!', flush=True)
            elif is_preclose_stemmed_note_candidate(
                ww, wh, waspect, bbox_fill, wcirc, ns, wx, wy, cleaned_preclose
            ):
                preclose_stemmed_note_candidates.append(wcnt.copy())
                print(f'  -> STEMMED_NOTE_CANDIDATE (preclose)!', flush=True)

            if wh > ns * 2.0 and waspect < 0.5:
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

        kern_size = max(2, ns // 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern_size, kern_size))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find staff lines in original (not cleaned) ROI for pitch detection
        raw_staff_line_ys = find_staff_lines_in_region(roi_bw, ns)
        pitch_gap = cal_line_gap or ns
        staff_line_ys, staff_line_source = choose_staff_lines_for_region(
            raw_staff_line_ys, page_staff_line_ys, y0, y1, expected_gap=page_pitch_gap
        )
        print(
            f'  Staff lines in region: {staff_line_ys}, raw={raw_staff_line_ys}, '
            f'source={staff_line_source}, region_staff_space={ns}',
            flush=True
        )

        # Split large blobs that may be multiple notes connected by beams/ties
        all_contours = []
        reject_counts = Counter()
        notes_before_region = len(all_notes)
        matched_whole_rects = set()
        matched_round_rects = set()
        fallback_whole_notes = 0
        tall_stack_contours = 0
        tall_stack_note_count = 0
        stacked_whole_contours = 0
        stacked_whole_note_count = 0
        fallback_stemmed_notes = 0
        preclose_stacked_whole_candidates = 0
        fallback_stacked_whole_notes = 0
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
            matched_whole_rect = None
            matched_round_rect = find_matching_round_note_rect(
                (x, y, bw_c, bh_c), preclose_round_note_rects, ns
            )
            half_note_cy = None
            for (wx, wy, ww, wh) in whole_note_rects:
                margin = ns * 0.5
                if (abs(x - wx) < margin and abs(y - wy) < margin and
                    abs(bw_c - ww) < margin and abs(bh_c - wh) < margin):
                    is_whole = True
                    matched_whole_rect = (wx, wy, ww, wh)
                    break
            if not is_whole:
                for (hx, hy, hw, hh), hcy in half_note_head_rects.items():
                    margin = ns * 0.5
                    if (abs(x - hx) < margin and abs(y - hy) < margin and
                        abs(bw_c - hw) < margin and abs(bh_c - hh) < margin):
                        half_note_cy = hcy
                        break

            clef = region.get('clef', 'bass')
            do_mode = region.get('doMode', 'fixed')
            do_key = region.get('doKey', 1)
            stack_notes = extract_tall_stack_notes(
                cleaned, cnt, rx0, y0, staff_line_ys, ns, clef, do_mode, do_key, expected_gap=pitch_gap
            )
            if stack_notes:
                tall_stack_contours += 1
                tall_stack_note_count += len(stack_notes)
                all_notes.extend(stack_notes)
                print(
                    f"  contour: pos=({x},{y}) size=({bw_c}x{bh_c}) area={area:.0f} "
                    f"-> TALL_STACK notes={[note.get('pitch') for note in stack_notes]}",
                    flush=True
                )
                if annotate:
                    total_accepted += len(stack_notes)
                    color = (233, 144, 74) if clef == 'treble' else (96, 69, 233)
                    for note in stack_notes:
                        half = note['size'] / 2
                        cv2.rectangle(
                            img,
                            (int(note['cx'] - half), int(note['cy'] - half)),
                            (int(note['cx'] + half), int(note['cy'] + half)),
                            color,
                            2
                        )
                        cv2.putText(
                            img,
                            note['pitch'],
                            (int(note['cx'] - half - 8), int(note['cy'] + half * 0.3)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2
                        )
                continue

            stacked_whole_notes = extract_stacked_whole_notes(
                cnt, rx0, y0, staff_line_ys, ns, clef, do_mode, do_key, expected_gap=pitch_gap
            )
            if stacked_whole_notes:
                stacked_whole_contours += 1
                stacked_whole_note_count += len(stacked_whole_notes)
                all_notes.extend(stacked_whole_notes)
                print(
                    f"  contour: pos=({x},{y}) size=({bw_c}x{bh_c}) area={area:.0f} "
                    f"-> STACKED_WHOLE notes={[note.get('pitch') for note in stacked_whole_notes]}",
                    flush=True
                )
                if annotate:
                    total_accepted += len(stacked_whole_notes)
                    color = (233, 144, 74) if clef == 'treble' else (96, 69, 233)
                    for note in stacked_whole_notes:
                        half = note['size'] / 2
                        cv2.rectangle(
                            img,
                            (int(note['cx'] - half), int(note['cy'] - half)),
                            (int(note['cx'] + half), int(note['cy'] + half)),
                            color,
                            2
                        )
                        cv2.putText(
                            img,
                            note['pitch'],
                            (int(note['cx'] - half - 8), int(note['cy'] + half * 0.3)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2
                        )
                continue

            if is_whole:
                reason = 'WHOLE_NOTE(matched)'
                ok = True
            else:
                ok, reason = classify_contour(cnt, ns, rw, cleaned, cleaned_preclose)
                if not ok:
                    reject_counts[reason] += 1
            cx = x + bw_c / 2 + rx0
            cy = y + bh_c / 2 + y0
            print(f'  contour: pos=({x},{y}) size=({bw_c}x{bh_c}) area={area:.0f} -> {reason}', flush=True)

            if ok:
                if matched_whole_rect is not None:
                    matched_whole_rects.add(matched_whole_rect)
                head_source = 'contour'
                if matched_round_rect is not None:
                    matched_round_rects.add(matched_round_rect)
                    wx, wy, ww, wh = matched_round_rect
                    cx = wx + ww / 2.0 + rx0
                    cy = wy + wh / 2.0
                    head_size = max(ww, wh)
                    head_source = 'preclose_round_candidate'
                    print(
                        f'    -> ROUND_CANDIDATE override head=({wx},{wy},{ww},{wh}) cy={cy:.1f}',
                        flush=True
                    )
                elif reason.startswith('WHOLE_NOTE'):
                    cx = x + bw_c / 2.0 + rx0
                    cy = y + bh_c / 2.0
                    head_size = max(bw_c, bh_c)
                    head_source = 'whole_note_bbox_center'
                    print(
                        f'    -> WHOLE_NOTE center override bbox=({x},{y},{bw_c},{bh_c}) cy={cy:.1f}',
                        flush=True
                    )
                else:
                    cx, cy, head_size, head_method = find_note_head_position(
                        cleaned, cnt, ns, cleaned_preclose
                    )
                    cx += rx0
                    head_source = head_method
                # Override head position for half notes detected in preclose
                if half_note_cy is not None and matched_round_rect is None:
                    cy = half_note_cy
                    head_source = 'preclose_half_note'
                    print(f'    -> HALF_NOTE override head_cy={cy:.1f}', flush=True)
                cy_abs = cy + y0
                head_refinement = None
                bbox_abs = {'x': float(x + rx0), 'y': float(y + y0), 'w': float(bw_c), 'h': float(bh_c)}
                if half_note_cy is None and matched_round_rect is None and not reason.startswith('WHOLE_NOTE'):
                    cy_abs, head_refinement = refine_stemmed_head_to_staff_line(
                        cy_abs,
                        bbox_abs,
                        [sly + y0 for sly in staff_line_ys],
                        ns,
                        head_source,
                    )
                    if head_refinement is None:
                        cy_abs, head_refinement = refine_likely_half_note_to_staff_line(
                            cy_abs,
                            bbox_abs,
                            area,
                            [sly + y0 for sly in staff_line_ys],
                            ns,
                            head_source,
                            reason,
                        )
                pitch = y_to_pitch(
                    cy_abs, [sly + y0 for sly in staff_line_ys], ns, clef,
                    do_mode, do_key, expected_gap=pitch_gap
                )
                if head_refinement:
                    print(
                        f"    -> HEAD_REFINED {head_refinement['type']} "
                        f"{head_refinement['from_cy']:.1f}->{head_refinement['to_cy']:.1f}",
                        flush=True
                    )
                print(f'    -> NOTE at ({cx:.0f},{cy_abs:.0f}) pitch={pitch} clef={clef}', flush=True)
                note = {
                    'cx': float(cx),
                    'cy': float(cy_abs),
                    'size': float(head_size),
                    'pitch': pitch,
                    'clef': clef,
                    'head_source': head_source,
                    'shape_density': round(float(area) / max(float(bw_c * bh_c), 1.0), 3),
                    'bbox': bbox_abs,
                }
                if head_refinement:
                    note['head_refinement'] = head_refinement
                all_notes.append(note)

            if annotate:
                if ok:
                    total_accepted += 1
                    if matched_round_rect is not None:
                        wx, wy, ww, wh = matched_round_rect
                        cx_a = wx + ww / 2.0 + rx0
                        cy_a = wy + wh / 2.0 + y0
                        hs_a = max(ww, wh)
                    elif reason.startswith('WHOLE_NOTE'):
                        cx_a = x + bw_c / 2.0 + rx0
                        cy_a = y + bh_c / 2.0 + y0
                        hs_a = max(bw_c, bh_c)
                    else:
                        cx_a, cy_a, hs_a, head_method_a = find_note_head_position(
                            cleaned, cnt, ns, cleaned_preclose
                        )
                        cx_a += rx0; cy_a += y0
                        if half_note_cy is not None:
                            cy_a = half_note_cy + y0
                        else:
                            cy_a, _head_refinement_a = refine_stemmed_head_to_staff_line(
                                cy_a,
                                {'x': float(x + rx0), 'y': float(y + y0), 'w': float(bw_c), 'h': float(bh_c)},
                                [sly + y0 for sly in staff_line_ys],
                                ns,
                                head_method_a,
                            )
                    half = hs_a / 2
                    pitch = y_to_pitch(
                        cy_a, [sly + y0 for sly in staff_line_ys], ns, clef,
                        do_mode, do_key, expected_gap=pitch_gap
                    )
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

        clef = region.get('clef', 'bass')
        do_mode = region.get('doMode', 'fixed')
        do_key = region.get('doKey', 1)
        for whole_cnt in preclose_contours:
            stacked_notes = extract_stacked_whole_notes(
                whole_cnt, rx0, y0, staff_line_ys, ns, clef,
                do_mode, do_key, expected_gap=pitch_gap
            )
            if not stacked_notes:
                continue
            preclose_stacked_whole_candidates += 1

            added_notes = []
            for fallback_note in stacked_notes:
                is_duplicate = False
                for note in all_notes[notes_before_region:]:
                    if abs(note['cx'] - fallback_note['cx']) <= ns * 0.9 and abs(note['cy'] - fallback_note['cy']) <= ns * 0.9:
                        is_duplicate = True
                        break
                if is_duplicate:
                    continue

                fallback_note = dict(fallback_note)
                fallback_note['head_source'] = 'preclose_stacked_whole_fallback'
                all_notes.append(fallback_note)
                added_notes.append(fallback_note)

            if not added_notes:
                continue

            fallback_stacked_whole_notes += len(added_notes)
            total_accepted += len(added_notes)
            sx, sy, sw, sh = cv2.boundingRect(whole_cnt)
            print(
                f"  fallback stacked whole: pos=({sx},{sy}) size=({sw}x{sh}) -> "
                f"notes={[note.get('pitch') for note in added_notes]}",
                flush=True
            )

            if annotate:
                color = (233, 144, 74) if clef == 'treble' else (96, 69, 233)
                for note in added_notes:
                    half = note['size'] / 2
                    cv2.rectangle(
                        img,
                        (int(note['cx'] - half), int(note['cy'] - half)),
                        (int(note['cx'] + half), int(note['cy'] + half)),
                        color,
                        2
                    )
                    cv2.putText(
                        img,
                        note['pitch'],
                        (int(note['cx'] - half - 8), int(note['cy'] + half * 0.3)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )

        for stem_cnt in preclose_stemmed_note_candidates:
            fallback_note = preclose_stemmed_note_to_note(
                cleaned_preclose, stem_cnt, rx0, y0, staff_line_ys, ns, clef,
                do_mode, do_key, expected_gap=pitch_gap
            )
            is_duplicate = False
            for note in all_notes[notes_before_region:]:
                if abs(note['cx'] - fallback_note['cx']) <= ns * 0.9 and abs(note['cy'] - fallback_note['cy']) <= ns * 0.9:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

            all_notes.append(fallback_note)
            fallback_stemmed_notes += 1
            total_accepted += 1
            bbox = fallback_note['bbox']
            print(
                f"  fallback stemmed note: pos=({bbox['x'] - rx0:.0f},{bbox['y'] - y0:.0f}) "
                f"size=({bbox['w']:.0f}x{bbox['h']:.0f}) -> NOTE at "
                f"({fallback_note['cx']:.0f},{fallback_note['cy']:.0f}) "
                f"pitch={fallback_note['pitch']} clef={clef}",
                flush=True
            )

            if annotate:
                half = fallback_note['size'] / 2
                color = (233, 144, 74) if clef == 'treble' else (96, 69, 233)
                cv2.rectangle(
                    img,
                    (int(fallback_note['cx'] - half), int(fallback_note['cy'] - half)),
                    (int(fallback_note['cx'] + half), int(fallback_note['cy'] + half)),
                    color,
                    2
                )
                cv2.putText(
                    img,
                    fallback_note['pitch'],
                    (int(fallback_note['cx'] - half - 8), int(fallback_note['cy'] + half * 0.3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        fallback_round_rects = preclose_round_note_rects or whole_note_rects
        for whole_rect in fallback_round_rects:
            if whole_rect in matched_whole_rects or whole_rect in matched_round_rects:
                continue
            wx, wy, ww, wh = whole_rect
            fallback_note = whole_note_bbox_to_note(
                wx, wy, ww, wh, rx0, y0, staff_line_ys, ns, clef, do_mode, do_key, expected_gap=pitch_gap
            )
            is_duplicate = False
            for note in all_notes[notes_before_region:]:
                if abs(note['cx'] - fallback_note['cx']) <= ns * 0.9 and abs(note['cy'] - fallback_note['cy']) <= ns * 0.9:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

            all_notes.append(fallback_note)
            fallback_whole_notes += 1
            total_accepted += 1
            print(
                f"  fallback round note: pos=({wx},{wy}) size=({ww}x{wh}) -> NOTE at "
                f"({fallback_note['cx']:.0f},{fallback_note['cy']:.0f}) pitch={fallback_note['pitch']} clef={clef}",
                flush=True
            )

            if annotate:
                half = fallback_note['size'] / 2
                color = (233, 144, 74) if clef == 'treble' else (96, 69, 233)
                cv2.rectangle(
                    img,
                    (int(fallback_note['cx'] - half), int(fallback_note['cy'] - half)),
                    (int(fallback_note['cx'] + half), int(fallback_note['cy'] + half)),
                    color,
                    2
                )
                cv2.putText(
                    img,
                    fallback_note['pitch'],
                    (int(fallback_note['cx'] - half - 8), int(fallback_note['cy'] + half * 0.3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        if annotate:
            cv2.putText(img, f"ss={ns} accepted={total_accepted}",
                        (rx0, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        raw_region_notes = all_notes[notes_before_region:]
        region_notes, suppressed_note_reasons = filter_region_note_artifacts(raw_region_notes, ns)
        if len(region_notes) != len(raw_region_notes):
            all_notes[notes_before_region:] = region_notes
            print(
                f'  suppressed artifacts: {len(raw_region_notes) - len(region_notes)} '
                f'{dict(suppressed_note_reasons)}',
                flush=True
            )
        log_event(
            'server', 'region_processed',
            request_id=request_id, session_id=session_id, endpoint=endpoint,
            region_index=region_idx, region=summarize_region(region),
            pixel_bounds={'x0': rx0, 'y0': y0, 'x1': rx1, 'y1': y1},
            region_staff_space=ns,
            staff_line_count=len(staff_line_ys),
            staff_lines=staff_line_ys,
            staff_line_source=staff_line_source,
            raw_staff_lines=raw_staff_line_ys,
            page_staff_line_count=len(page_staff_line_ys),
            preclose_contour_count=len(preclose_contours),
            preclose_round_candidate_count=len(preclose_round_note_rects),
            preclose_stemmed_candidate_count=len(preclose_stemmed_note_candidates),
            preclose_stacked_whole_candidate_count=preclose_stacked_whole_candidates,
            round_candidate_matched_count=len(matched_round_rects),
            tall_stack_contour_count=tall_stack_contours,
            tall_stack_note_count=tall_stack_note_count,
            stacked_whole_contour_count=stacked_whole_contours,
            stacked_whole_note_count=stacked_whole_note_count,
            contour_count=len(contours),
            split_contour_count=len(all_contours),
            accepted_note_count=len(region_notes),
            raw_accepted_note_count=len(raw_region_notes),
            suppressed_note_count=len(raw_region_notes) - len(region_notes),
            suppressed_note_reasons=dict(suppressed_note_reasons),
            fallback_whole_note_count=fallback_whole_notes,
            fallback_stemmed_note_count=fallback_stemmed_notes,
            fallback_stacked_whole_note_count=fallback_stacked_whole_notes,
            pitches=[note.get('pitch') for note in region_notes],
            head_sources=dict(Counter(note.get('head_source', 'unknown') for note in region_notes)),
            note_debug=[
                {
                    'pitch': note.get('pitch'),
                    'cx': round(note.get('cx', 0), 1),
                    'cy': round(note.get('cy', 0), 1),
                    'head_source': note.get('head_source'),
                    'head_refinement': note.get('head_refinement'),
                    'shape_density': note.get('shape_density'),
                    'bbox': note.get('bbox'),
                }
                for note in region_notes
            ],
            rejected_reasons=dict(reject_counts.most_common(12)),
            duration_ms=(time.perf_counter() - region_start) * 1000,
        )

    log_event(
        'server', 'process_regions_complete',
        request_id=request_id, session_id=session_id, endpoint=endpoint,
        annotate=annotate, region_count=len(regions), total_note_count=len(all_notes),
        duration_ms=(time.perf_counter() - process_start) * 1000,
    )
    if annotate:
        return img
    return all_notes


@app.route('/log/client', methods=['POST'])
def client_log():
    data = request.get_json(silent=True) or {}
    events = data.get('events') if isinstance(data.get('events'), list) else [data]
    stored = 0

    for item in events:
        if not isinstance(item, dict):
            continue
        event_name = item.get('event')
        if not event_name:
            continue
        log_event(
            'client', event_name, level=item.get('level', 'INFO'),
            session_id=item.get('sessionId') or current_session_id(item),
            client_ts=item.get('clientTs'),
            request=request_meta(),
            payload=item.get('payload', {}),
        )
        stored += 1

    return jsonify({'ok': True, 'stored': stored})


@app.route('/calibrate', methods=['POST'])
def calibrate():
    started = time.perf_counter()
    data = request.get_json(silent=True) or {}
    region = data.get('region')
    session_id = current_session_id(data)
    request_id = uuid.uuid4().hex[:12]
    image_bytes = base64.b64decode(data.get('image', ''))
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None or not region:
        log_event(
            'server', 'calibrate_failed', level='ERROR',
            request_id=request_id, session_id=session_id, reason='invalid_input',
            region=summarize_region(region), duration_ms=(time.perf_counter() - started) * 1000,
            request=request_meta(),
        )
        return jsonify({'success': False, 'error': 'invalid input', 'requestId': request_id})
    
    h, w = img.shape
    x0, y0 = int(region['x'] * w), int(region['y'] * h)
    x1, y1 = int((region['x'] + region['w']) * w), int((region['y'] + region['h']) * h)
    roi = img[y0:y1, x0:x1]
    
    if roi.size == 0:
        log_event(
            'server', 'calibrate_failed', level='ERROR',
            request_id=request_id, session_id=session_id, reason='empty_region',
            region=summarize_region(region), duration_ms=(time.perf_counter() - started) * 1000,
            request=request_meta(),
        )
        return jsonify({'success': False, 'error': 'empty region', 'requestId': request_id})
    
    # Estimate staff space in this region
    _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    staff_space = estimate_staff_space_full_page(roi)
    
    lines = find_staff_lines_in_region(bw, staff_space)
    best_lines = select_best_five_lines(lines, expected_gap=staff_space, anchor_y=roi.shape[0] / 2)
    print(
        f'Calibration: detected {len(lines)} lines: {lines}, '
        f'best_lines={best_lines}, staff_space={staff_space}',
        flush=True
    )
    
    if len(best_lines) != 5:
        log_event(
            'server', 'calibrate_failed', level='WARNING',
            request_id=request_id, session_id=session_id, reason='staff_line_count',
            region=summarize_region(region), line_count=len(lines), best_lines=best_lines,
            staff_space=staff_space, duration_ms=(time.perf_counter() - started) * 1000,
            request=request_meta(),
        )
        return jsonify({'success': False, 'error': f'detected {len(lines)} lines (need 5)', 'requestId': request_id})
    
    # Calculate average line gap
    gaps = [best_lines[i+1] - best_lines[i] for i in range(4)]
    avg_gap = sum(gaps) / len(gaps)
    
    response = {
        'success': True,
        'staff_space': int(staff_space),
        'line_gap': avg_gap,
        'lines': best_lines,
        'requestId': request_id
    }
    log_event(
        'server', 'calibrate_complete',
        request_id=request_id, session_id=session_id,
        region=summarize_region(region), staff_space=staff_space, line_gap=avg_gap,
        lines=best_lines, duration_ms=(time.perf_counter() - started) * 1000,
        request=request_meta(),
    )
    return jsonify(response)


@app.route('/detect', methods=['POST'])
def detect():
    started = time.perf_counter()
    data = request.get_json(silent=True) or {}
    regions = data.get('regions', [])
    calibration = data.get('calibration')
    session_id = current_session_id(data)
    request_id = uuid.uuid4().hex[:12]
    image_bytes = base64.b64decode(data.get('image', ''))
    notes = process_regions(
        image_bytes, regions, annotate=False, calibration=calibration,
        request_id=request_id, session_id=session_id, endpoint='detect'
    )
    log_event(
        'server', 'detect_complete',
        request_id=request_id, session_id=session_id,
        region_count=len(regions), calibration=calibration, note_count=len(notes),
        pitches=[note.get('pitch') for note in notes],
        duration_ms=(time.perf_counter() - started) * 1000, request=request_meta(),
    )
    return jsonify({'notes': notes, 'requestId': request_id})


@app.route('/debug', methods=['POST'])
def debug():
    started = time.perf_counter()
    data = request.get_json(silent=True) or {}
    regions = data.get('regions', [])
    calibration = data.get('calibration')
    session_id = current_session_id(data)
    request_id = uuid.uuid4().hex[:12]
    image_bytes = base64.b64decode(data.get('image', ''))
    img = process_regions(
        image_bytes, regions, annotate=True, calibration=calibration,
        request_id=request_id, session_id=session_id, endpoint='debug'
    )
    if img is None:
        log_event(
            'server', 'debug_failed', level='ERROR',
            request_id=request_id, session_id=session_id,
            region_count=len(regions), duration_ms=(time.perf_counter() - started) * 1000,
            request=request_meta(), reason='bad_image',
        )
        return jsonify({'error': 'bad image'}), 400
    _, buf = cv2.imencode('.png', img)
    log_event(
        'server', 'debug_complete',
        request_id=request_id, session_id=session_id,
        region_count=len(regions), calibration=calibration,
        image_bytes=len(buf), duration_ms=(time.perf_counter() - started) * 1000,
        request=request_meta(),
    )
    return send_file(io.BytesIO(buf.tobytes()), mimetype='image/png')


@app.route('/')
def index():
    return open(os.path.join(BASE_DIR, 'index.html')).read()


if __name__ == '__main__':
    log_event('server', 'startup', pid=os.getpid(), host='0.0.0.0', port=5555, log_dir=LOG_DIR)
    app.run(host='0.0.0.0', port=5555, debug=False)
