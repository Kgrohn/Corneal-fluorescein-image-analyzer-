"""Corneal fluorescein image quantification (batch analyzer).

Install: pip install opencv-python numpy pandas openpyxl matplotlib
Run:     python Auto_Corneal_Fluorescein.py

Put this program in a folder containing image files (including subfolders).
On first run: trace corneal ROI with left clicks; ENTER accepts, R clears, ESC skips.
ROI polygons are stored in corneal_rois.json and reused on later runs.
The original RGB image should be unannotated; a drawn white outline contaminates
its underlying pixels. If using annotated images, the program excludes near-white
outline pixels, but this cannot restore the overwritten image information.

Outputs in corneal_analysis_output/: results.xlsx with Summary, Histogram_Long,
Histogram_Wide, Parameters, and QC; PNG masks and per-image histogram plots.
This program measures *visible* corneal ROI; % positive cannot fully correct
for squinting/eyelid occlusion. Acquisition conditions must be standardized.
Thresholds are exploratory until validated for the study protocol.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ----------------------------- SETTINGS ------------------------------
def base_folder():
    return Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent

ROOT = base_folder()
OUTPUT = ROOT / 'corneal_analysis_output'
ROI_JSON = ROOT / 'corneal_rois.json'
FORMATS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
GREEN_MIN = 45                  # pilot threshold, not validated
GREEN_DOMINANCE = 10           # G-R and G-B must each be >= this
EXCLUDE_DRAWN_WHITE = True     # excludes near-white drawn ROI outlines from calculations
WHITE_RGB_MIN = 200
WHITE_RGB_SPREAD = 35
MAKE_HISTOGRAM_PLOTS = True
MIN_ROI_PIXELS = 100


def choose_polygon(img, title):
    """Interactively delineate the visible cornea; points in source-image coordinates."""
    h, w = img.shape[:2]
    scale = min(1.0, 1050.0 / max(w, h))
    canvas = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1 else img.copy()
    points = []
    window = 'CORNEA ROI | Left click: vertices | ENTER: save | R: redo | ESC: skip'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def mouse(event, x, y, flags, data):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([int(np.clip(round(x / scale), 0, w - 1)),
                           int(np.clip(round(y / scale), 0, h - 1))])

    cv2.setMouseCallback(window, mouse)
    while True:
        preview = canvas.copy()
        if points:
            shown = np.array(points, dtype=np.float32) * scale
            shown = np.rint(shown).astype(np.int32)
            for xy in shown:
                cv2.circle(preview, tuple(xy), 3, (0, 255, 255), -1)
            if len(shown) >= 2:
                cv2.polylines(preview, [shown], len(shown) >= 3, (0, 255, 255), 1)
        cv2.putText(preview, title[-75:], (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.imshow(window, preview)
        key = cv2.waitKey(40) & 0xFF
        if key in (13, 10) and len(points) >= 3:
            cv2.destroyWindow(window)
            return points
        if key in (ord('r'), ord('R')):
            points.clear()
        if key == 27:
            cv2.destroyWindow(window)
            return None


def read_bgr(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, 'Image could not be opened'
    if img.dtype != np.uint8:
        return None, 'Not 8-bit; skipped rather than rescaling each image (invalid intensity comparisons)'
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        return None, 'Not an RGB/BGR image'
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img, None


def measure(image, vertices):
    h, w = image.shape[:2]
    if not isinstance(vertices, list) or len(vertices) < 3:
        raise ValueError('Polygon needs at least 3 vertices')
    polygon = np.array(vertices, dtype=np.int32)
    if np.any(polygon[:, 0] < 0) or np.any(polygon[:, 0] >= w) or np.any(polygon[:, 1] < 0) or np.any(polygon[:, 1] >= h):
        raise ValueError('Saved polygon exceeds image bounds; delete its entry in corneal_rois.json and retrace')
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    original_n = int(mask.sum())
    if original_n < MIN_ROI_PIXELS:
        raise ValueError('ROI has too few pixels')
    b8, g8, r8 = cv2.split(image)
    b = b8.astype(np.int16)
    g = g8.astype(np.int16)
    r = r8.astype(np.int16)
    white = ((np.minimum(np.minimum(r, g), b) >= WHITE_RGB_MIN) &
             (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b) <= WHITE_RGB_SPREAD))
    included = mask.astype(bool)
    excluded_white = int((included & white).sum()) if EXCLUDE_DRAWN_WHITE else 0
    if EXCLUDE_DRAWN_WHITE:
        included &= ~white
    n = int(included.sum())
    if n < MIN_ROI_PIXELS:
        raise ValueError('ROI too small after excluding outline pixels')

    positive = (included & (g >= GREEN_MIN) & (g-r >= GREEN_DOMINANCE) & (g-b >= GREEN_DOMINANCE))
    count = int(positive.sum())
    vals = g[included]
    hist = np.bincount(vals, minlength=256).astype(int)
    excess = np.maximum(g - np.maximum(r, b), 0)
    excess_vals = excess[included]
    bc = np.maximum(vals.astype(np.int32) - int(np.percentile(vals, 5)), 0)
    # The P5 subtraction below is a descriptive internal reference only, not
    # a biological background control. Fixed/background controls are preferred.
    stats = {
        'roi_pixels_original': original_n,
        'roi_pixels_analyzed': n,
        'excluded_white_pixels': excluded_white,
        'positive_green_pixels': count,
        'positive_area_percent': round(100 * count / n, 5),
        'mean_green_roi': round(float(vals.mean()), 5),
        'mean_green_positive': round(float(g[positive].mean()), 5) if count else np.nan,
        'integrated_green_roi': int(vals.astype(np.int64).sum()),
        'mean_green_excess_roi': round(float(excess_vals.mean()), 5),
        'integrated_green_excess_roi': int(excess_vals.astype(np.int64).sum()),
        'intensity_p50': float(np.percentile(vals, 50)),
        'intensity_p75': float(np.percentile(vals, 75)),
        'intensity_p90': float(np.percentile(vals, 90)),
        'intensity_p95': float(np.percentile(vals, 95)),
        'roi_internal_p5_reference': float(np.percentile(vals, 5)),
        'internal_p5_corrected_mean': round(float(bc.mean()), 5),
        'saturated_green_pixels': int((vals == 255).sum()),
        'saturated_green_percent': round(float((vals == 255).mean()) * 100, 5),
    }
    return stats, hist, mask, included, positive


def save_qc(image, vertices, included, positive, path):
    qc = image.copy()
    overlay = image.copy()
    overlay[positive] = (15, 170, 255)  # orange in BGR
    qc = cv2.addWeighted(overlay, 0.40, qc, 0.60, 0)
    cv2.polylines(qc, [np.array(vertices, dtype=np.int32)], True, (255, 255, 255), 1)
    # Orange = threshold-positive; white = manually selected corneal border
    cv2.imwrite(str(path), qc)


def save_plot(hist, path, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(np.arange(256), hist, width=1)
    ax.axvline(GREEN_MIN, linestyle='--', label=f'Green minimum = {GREEN_MIN}')
    ax.set(xlabel='Green-channel intensity (0–255)', ylabel='ROI pixel count', title=name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    OUTPUT.mkdir(exist_ok=True)
    qc_dir = OUTPUT / 'QC_masks'
    plot_dir = OUTPUT / 'Histogram_plots'
    qc_dir.mkdir(exist_ok=True)
    if MAKE_HISTOGRAM_PLOTS:
        plot_dir.mkdir(exist_ok=True)
    rois = json.loads(ROI_JSON.read_text(encoding='utf-8')) if ROI_JSON.exists() else {}
    images = sorted(p for p in ROOT.rglob('*') if p.is_file() and p.suffix.lower() in FORMATS and OUTPUT not in p.parents)
    summary, wide, long, qcs = [], [], [], []
    print(f'Found {len(images)} image(s). ROI coordinates saved at {ROI_JSON}')
    for idx, path in enumerate(images, 1):
        key = path.relative_to(ROOT).as_posix()
        print(f'[{idx}/{len(images)}] {key}', flush=True)
        img, problem = read_bgr(path)
        if problem:
            qcs.append({'file': key, 'status': 'SKIPPED', 'details': problem})
            continue
        vertices = rois.get(key)
        if vertices is None:
            try:
                vertices = choose_polygon(img, key)
            except cv2.error as e:
                qcs.append({'file': key, 'status': 'SKIPPED', 'details': f'GUI unavailable; {e}'})
                continue
            if vertices is None:
                qcs.append({'file': key, 'status': 'SKIPPED', 'details': 'ROI selection canceled'})
                continue
            rois[key] = vertices
            ROI_JSON.write_text(json.dumps(rois, indent=2), encoding='utf-8')
        try:
            stats, hist, mask, included, positive = measure(img, vertices)
            stats = {'file_name': path.name, 'relative_path': key, 'image_width': img.shape[1], 'image_height': img.shape[0], **stats}
            summary.append(stats)
            wide.append({'relative_path': key, **{f'G_{i:03d}': int(v) for i, v in enumerate(hist)}})
            long.extend({'relative_path': key, 'intensity': int(i), 'pixel_count': int(v),
                         'fraction_of_roi': float(v / stats['roi_pixels_analyzed'])}
                        for i, v in enumerate(hist))
            # Stable file stem handles same filenames in different subfolders
            safe = f'{idx:04d}_{path.stem}'
            save_qc(img, vertices, included, positive, qc_dir / f'{safe}_QC.png')
            if MAKE_HISTOGRAM_PLOTS:
                save_plot(hist, plot_dir / f'{safe}_hist.png', key)
            notes = []
            if stats['excluded_white_pixels'] > 0:
                notes.append('Near-white pixels excluded; use clean originals for final reporting')
            if stats['saturated_green_percent'] > 1:
                notes.append('More than 1% of ROI green channel is saturated')
            qcs.append({'file': key, 'status': 'ANALYZED', 'details': '; '.join(notes) if notes else 'Review mask for correct ROI and threshold'})
        except Exception as exc:
            qcs.append({'file': key, 'status': 'ERROR', 'details': str(exc)})
            print(f'  ERROR: {exc}')

    parameters = pd.DataFrame([
        ('green_minimum', GREEN_MIN), ('green_dominance_over_red_blue', GREEN_DOMINANCE),
        ('exclude_near_white', EXCLUDE_DRAWN_WHITE), ('white_rgb_min', WHITE_RGB_MIN),
        ('white_rgb_max_spread', WHITE_RGB_SPREAD), ('roi_method', 'Manual polygon, saved/reused'),
        ('pixel_area_denominator', 'All included visible ROI pixels'),
        ('histogram', 'Raw G channel counts from 0 to 255, within analyzed ROI'),
        ('green_excess', 'max(G-max(R,B),0)'),
        ('internal_p5_correction', 'Exploratory only; use validated background for primary analysis'),
        ('source_image_requirement', 'Unannotated 8-bit BGR/RGB, comparable exposure and gain'),
        ('study_limitation', 'ROI normalization does not eliminate eyelid-occlusion/squint bias')
    ], columns=['parameter', 'value'])
    output_file = OUTPUT / 'corneal_fluorescein_results.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        pd.DataFrame(summary).to_excel(writer, sheet_name='Summary', index=False)
        pd.DataFrame(long).to_excel(writer, sheet_name='Histogram_Long', index=False)
        pd.DataFrame(wide).to_excel(writer, sheet_name='Histogram_Wide', index=False)
        parameters.to_excel(writer, sheet_name='Parameters', index=False)
        pd.DataFrame(qcs).to_excel(writer, sheet_name='QC', index=False)
        from openpyxl.styles import Font, PatternFill
        for ws in writer.book.worksheets:
            ws.freeze_panes = 'B2'
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(color='FFFFFF', bold=True)
                cell.fill = PatternFill('solid', fgColor='164B59')
            for col in ws.columns:
                letter = col[0].column_letter
                # Avoid enormous memory/iteration for histogram_long
                if ws.title == 'Histogram_Long':
                    widths = {'A': 40, 'B': 14, 'C': 18, 'D': 20}
                    ws.column_dimensions[letter].width = widths.get(letter, 15)
                elif ws.title == 'Histogram_Wide':
                    ws.column_dimensions[letter].width = 39 if letter == 'A' else 13
                else:
                    ws.column_dimensions[letter].width = min(50, max(15, len(str(col[0].value)) + 4))
    print(f'Finished: {len(summary)} image(s) analyzed, {len(qcs)-len(summary)} skipped/errors.')
    print(f'Excel workbook: {output_file}')


if __name__ == '__main__':
    main()
