"""
Taswira Analytics: midwife visit demo (Streamlit)
=================================================
Upload a blind sweep (frames, a ZIP of frames, or a video). The app:

  1. shows danger signs first (they always override the scan),
  2. judges scan quality (good scan / please scan again),
  3. finds the best frame for each of the five standard planes,
  4. flags a possible breech (only when a head view was captured),
  5. runs the exploratory brain check (only if its model was exported),
  6. shows a guidance card and a referral note.

Model files are produced by the notebook (see export_artifacts_cell.py) and live in ./artifacts:
  config.json, heads.npz, taswira_features.tflite

DECISION SUPPORT ONLY. Study prototype: not clinically validated, never a diagnosis.
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import os
import pathlib
import re
import tempfile
import threading
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
ART_DIR = HERE / "artifacts"

# --------------------------------------------------------------------------------------
# Constants (mirror the notebook)
# --------------------------------------------------------------------------------------
CLASS_NAMES = ["biparietal", "abdominal", "heart", "spine", "femur", "no_plane"]
NO_PLANE, PLANE_IDS = 5, [0, 1, 2, 3, 4]
VIEW_LABELS = {"biparietal": "head", "abdominal": "tummy", "heart": "heart", "spine": "spine", "femur": "thigh bone"}
IMG_SIZE, CONTRAST_PCT = 224, (1, 99.5)
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
MAX_VIDEO_FRAMES = 600
MIN_FRAMES_FOR_BREECH = 100   # the breech model averages features over a whole sweep (~200 frames in training)

# Danger signs and guidance cards: PLACEHOLDER content, to be completed from Kenyan MOH guidelines
# and approved by a clinician (same as the notebook).
DANGER_SIGNS = ["Vaginal bleeding", "Convulsions (fits)", "Severe headache or blurred vision", "Fever",
                "Severe abdominal pain", "Fast or difficult breathing", "Reduced or no baby movement"]
CARDS = {
    "urgent": ("#b3261e", "🚨 Urgent: refer now",
               "A danger sign was reported. Arrange urgent transfer. The scan result does not change this decision.",
               "[First-response steps while arranging transfer: from the MOH guideline, approved by a clinician.]"),
    "rescan": ("#a15c00", "🔁 Please scan again",
               "No key view was captured. Repeat the sweep, using the vertical pattern (it worked best in our tests).",
               "[How many attempts before escalating, and to whom: from facility protocol.]"),
    "breech": ("#a15c00", "⚠️ Possible breech: refer for review",
               "The model suggests the baby may be lying bottom-first. A sonographer or clinician should confirm; "
               "whether referral is needed depends on how far along the pregnancy is.",
               "[Referral timing and steps for suspected breech: from the MOH guideline, approved by a clinician.]"),
    "defect": ("#7a3e9d", "🩺 Possible brain abnormality: specialist review",
               "",  # message comes from the brain check
               "[Referral pathway for a suspected fetal anomaly: from the MOH guideline or facility protocol, approved by a "
               "clinician. Explain to the client that a more detailed scan is recommended; the specialist discusses any findings.]"),
    "routine": ("#1e6b3a", "✅ Good scan, no flags",
                "Key views were captured and nothing was flagged. Continue routine antenatal care and remind the mother "
                "of the danger signs.",
                "[Next-visit schedule and counselling points: from the MOH guideline.]"),
}
FOOTER = "Decision support only. This tool does not diagnose. When in doubt, refer."


# --------------------------------------------------------------------------------------
# Preprocessing (identical to the notebook)
# --------------------------------------------------------------------------------------
def stretch_pad(a: np.ndarray) -> np.ndarray:
    """Contrast-stretch (1st-99.5th percentile), pad to a square, resize to 224x224 (uint8)."""
    lo, hi = np.percentile(a, CONTRAST_PCT)
    a = np.clip((a - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    h, w = a.shape
    s = max(h, w)
    sq = np.zeros((s, s), np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
    return np.asarray(Image.fromarray(sq).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))


def preprocess_fixed(img_gray: Image.Image, crop: tuple) -> np.ndarray:
    """Training-probe frames: fixed ultrasound-sector crop."""
    return stretch_pad(np.asarray(img_gray.crop(tuple(crop)), dtype=np.float32))


def preprocess_detect(img_gray: Image.Image) -> np.ndarray:
    """Other devices: find the bright sector of each image."""
    a = np.asarray(img_gray, dtype=np.float32)
    m = a > 8
    r, c = np.where(m.mean(1) > 0.02)[0], np.where(m.mean(0) > 0.02)[0]
    if len(r) > 10 and len(c) > 10:
        a = a[r[0]:r[-1] + 1, c[0]:c[-1] + 1]
    return stretch_pad(a)


PREP_MODES = {
    "Automatic (recommended)": "auto",
    "Always use the training-probe crop": "fixed",
    "Always detect the ultrasound sector": "detect",
}


def preprocess_frame(img: Image.Image, cfg: dict, mode: str) -> np.ndarray:
    native = tuple(cfg.get("natalia_size") or ())
    if mode == "fixed" or (mode == "auto" and native and img.size == native):
        return preprocess_fixed(img, cfg["crop"])
    return preprocess_detect(img)


# --------------------------------------------------------------------------------------
# Reading uploads
# --------------------------------------------------------------------------------------
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _video_frames(data: bytes, suffix: str, stem: str):
    import cv2  # optional dependency (opencv-python-headless)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as t:
        t.write(data)
        path = t.name
    try:
        cap, i = cv2.VideoCapture(path), 0
        while i < MAX_VIDEO_FRAMES:
            ok, fr = cap.read()
            if not ok:
                break
            g = fr if fr.ndim == 2 else cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            yield f"{stem}_frame_{i:04d}", Image.fromarray(g)
            i += 1
        cap.release()
    finally:
        os.unlink(path)


def iter_frames(files):
    """Yield (name, grayscale PIL image) in sweep order from images, ZIPs of images, or videos."""
    for f in sorted(files, key=lambda f: natural_key(f.name)):
        ext, data = pathlib.Path(f.name).suffix.lower(), f.getvalue()
        if ext in IMG_EXT:
            yield f.name, Image.open(io.BytesIO(data)).convert("L")
        elif ext == ".zip":
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for n in sorted(z.namelist(), key=natural_key):
                    p = pathlib.PurePosixPath(n)
                    if n.endswith("/") or "__MACOSX" in p.parts or p.name.startswith("."):
                        continue
                    if p.suffix.lower() in IMG_EXT:
                        yield n, Image.open(io.BytesIO(z.read(n))).convert("L")
        elif ext in VIDEO_EXT:
            yield from _video_frames(data, ext, pathlib.Path(f.name).stem)


def uploads_digest(files) -> str:
    h = hashlib.md5()
    for f in files:
        h.update(f.name.encode())
        h.update(f.getvalue())
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# Model: TFLite feature extractor + numpy heads
# --------------------------------------------------------------------------------------
def _interpreter_class():
    """Prefer the light LiteRT runtime; fall back to tflite-runtime, then full TensorFlow."""
    for mod in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter"):
        try:
            return __import__(mod, fromlist=["Interpreter"]).Interpreter
        except ImportError:
            pass
    import tensorflow as tf
    return tf.lite.Interpreter


class FeatureExtractor:
    """EfficientNet-B0 + MobileNetV2 features (2,560-d) for 224x224 grayscale frames."""

    def __init__(self, path):
        Interp = _interpreter_class()
        try:
            self.it = Interp(model_path=str(path), num_threads=os.cpu_count() or 2)
        except TypeError:
            self.it = Interp(model_path=str(path))
        self.it.allocate_tensors()
        self.inp, self.out = self.it.get_input_details()[0], self.it.get_output_details()[0]
        self.lock = threading.Lock()   # the interpreter is not thread-safe; Streamlit sessions run in threads

    def __call__(self, imgs: np.ndarray, progress=None) -> np.ndarray:
        feats = []
        with self.lock:
            for i, a in enumerate(imgs):
                x = np.repeat(a[None, :, :, None], 3, -1).astype(np.float32)   # (1,224,224,3), values 0-255
                self.it.set_tensor(self.inp["index"], x)
                self.it.invoke()
                feats.append(self.it.get_tensor(self.out["index"])[0].copy())
                if progress:
                    progress((i + 1) / len(imgs))
        return np.stack(feats).astype(np.float32)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def plane_probs(F: np.ndarray, h: dict) -> np.ndarray:
    """Per-frame probabilities over the 6 classes (= the notebook's standardise + logistic regression)."""
    z = (F - h["plane_mean"]) / h["plane_scale"]
    return softmax(z @ h["plane_W"] + h["plane_b"])


def decide(P: np.ndarray, tau: float) -> np.ndarray:
    """Frame decision: a plane if P(plane) >= tau (type = most likely plane), else no_plane."""
    return np.where(1 - P[:, NO_PLANE] >= tau, np.array(PLANE_IDS)[P[:, PLANE_IDS].argmax(1)], NO_PLANE)


def breech_prob(f_mean: np.ndarray, h: dict) -> float:
    z = (f_mean - h["breech_mean"]) / h["breech_scale"]
    bias = float(np.ravel(h["breech_b"])[0])
    return float(1.0 / (1.0 + np.exp(-(z @ h["breech_coef"] + bias))))


def _nice(c: str, cfg: dict) -> str:
    return cfg["anom"].get("nice", {}).get(c, c.replace("-", " "))


def brain_message(P: np.ndarray, cfg: dict):
    a = cfg["anom"]
    classes, ni = a["classes"], a["normal_index"]
    pa = 1 - float(P[ni])
    top = [classes[i] for i in np.argsort(-P) if i != ni][:2]
    alt = f" (or {_nice(top[1], cfg)})" if len(top) > 1 else ""
    if pa >= a["tau_refer"]:
        return "refer", pa, f"Possible signs of {_nice(top[0], cfg)}{alt}. Refer to a specialist for a detailed scan."
    if pa >= a["tau_anom"]:
        return "consult", pa, (f"Possible brain abnormality (most like {_nice(top[0], cfg)}). "
                               "Book the client for a consultation and detailed scan.")
    return "no_flag", pa, ("The brain check did not flag a concern. This does not rule out a defect; "
                           "routine anomaly scanning still applies.")


def brain_check(xf: np.ndarray, cfg: dict, h: dict) -> dict:
    """Exploratory brain check: unfamiliar-image guard first, then a tiered, non-diagnostic message."""
    if "anom_W" not in h or "anom" not in cfg:
        return {"status": "unavailable", "msg": "Brain check unavailable (anomaly model not exported)."}
    x = np.atleast_2d(xf)
    zu = (x - h["anom_mean"]) / h["anom_scale"]
    zu = zu / np.linalg.norm(zu, axis=1, keepdims=True).clip(min=1e-9)
    if 1 - float((zu @ h["anom_UA"].T).max()) > cfg["anom"]["ood_limit"]:
        return {"status": "not_assessed",
                "msg": "Brain check could not assess this image: it looks unlike the images the check was trained on."}
    P = softmax(((x - h["anom_mean"]) / h["anom_scale"]) @ h["anom_W"] + h["anom_b"])[0]
    status, pa, msg = brain_message(P, cfg)
    return {"status": status, "p_abnormal": pa, "msg": msg}


# --------------------------------------------------------------------------------------
# The visit: care-pathway logic (same order and safety rules as notebook section 9)
# --------------------------------------------------------------------------------------
def assess_visit(P, F, n_total, cfg, heads, tau, tau_sweep, signs):
    best = float((1 - P[:, NO_PLANE]).max())
    good = best >= tau_sweep
    dec = decide(P, tau)
    views = {}
    for c in PLANE_IDS:
        j = int(P[:, c].argmax())
        views[CLASS_NAMES[c]] = {"frame": j, "prob": float(P[j, c]), "found": bool((dec == c).any())}
    head_found = views["biparietal"]["found"]
    long_enough = n_total >= MIN_FRAMES_FOR_BREECH
    p_breech = breech_prob(F.mean(0), heads)
    # Safety rule: judge position only when a head view was captured (a missing head must never mean "breech")
    breech_flag = good and head_found and long_enough and p_breech >= 0.5
    brain = brain_check(F[views["biparietal"]["frame"]], cfg, heads) if (good and head_found) else None
    defect_flag = brain is not None and brain["status"] in ("refer", "consult")

    key = ("urgent" if signs else "rescan" if not good else "defect" if defect_flag
           else "breech" if breech_flag else "routine")
    colour, title, action, placeholder = CARDS[key]
    if key == "defect":
        action = brain["msg"]
    if key in ("defect", "routine") and breech_flag:
        action += " Also flagged: possible breech (see row 4)."

    scan = (f"Good scan (best frame {best:.2f} ≥ {tau_sweep:.2f})" if good
            else f"Not usable (best frame {best:.2f} < {tau_sweep:.2f})")
    if not good:
        breech_txt = "not assessed (scan not usable)"
    elif not head_found:
        breech_txt = "cannot judge: no head view captured (repeat the sweep, or clinician review if late in pregnancy)"
    elif not long_enough:
        breech_txt = f"cannot judge: sweep too short ({n_total} frames; a full sweep has about 200)"
    else:
        breech_txt = f"{'possible breech' if breech_flag else 'head-down likely'} (P(breech) = {p_breech:.2f})"
    brain_txt = ("not assessed (scan not usable)" if not good else
                 "not assessed (no head view captured)" if brain is None else brain["msg"])
    found = [VIEW_LABELS[k] for k, v in views.items() if v["found"]]
    return {"key": key, "colour": colour, "title": title, "action": action, "placeholder": placeholder,
            "good": good, "best": best, "scan": scan, "views": views, "found": found,
            "breech_txt": breech_txt, "brain_txt": brain_txt, "p_breech": p_breech, "breech_flag": breech_flag,
            "signs": ", ".join(signs) if signs else "none reported"}


def referral_note(v: dict, visit_id: str, midwife: str) -> str:
    return "\n".join([
        f"REFERRAL NOTE · {visit_id} · {date.today().isoformat()}",
        f"Reason: {v['title'].split(' ', 1)[1]}",
        f"Danger signs: {v['signs']}",
        f"Scan: {v['scan']}",
        f"Views captured: {', '.join(v['found']) or 'none'}",
        f"Breech flag: {v['breech_txt']}",
        f"Brain check (exploratory): {v['brain_txt']}",
        f"Referring midwife: {midwife or '__________'}",
        "Best frames attached for review.",
        "",
        FOOTER,
    ])


# --------------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    cfg = json.loads((ART_DIR / "config.json").read_text(encoding="utf-8"))
    with np.load(ART_DIR / "heads.npz") as z:
        heads = {k: z[k].astype(np.float32) for k in z.files}
    ext = FeatureExtractor(ART_DIR / cfg.get("feature_model", "taswira_features.tflite"))
    return cfg, heads, ext


def card_html(colour: str, title: str, action: str) -> str:
    return (f"<div style='padding:14px 16px;border-radius:10px;background:{colour};color:#fff'>"
            f"<div style='font-size:20px;font-weight:700'>{html.escape(title)}</div>"
            f"<div style='margin-top:6px;font-size:15px'>{html.escape(action)}</div></div>")


def run_analysis(files, cfg, heads, extractor, mode, stride):
    bar = st.progress(0.0, text="Reading frames…")
    imgs, ids, n_total = [], [], 0
    for k, (_, im) in enumerate(iter_frames(files)):
        n_total += 1
        if k % stride == 0:
            imgs.append(preprocess_frame(im, cfg, mode))
            ids.append(k)
    if not imgs:
        bar.empty()
        return None
    imgs = np.stack(imgs)
    F = extractor(imgs, progress=lambda p: bar.progress(p, text=f"Analysing frames… {int(p * 100)}%"))
    bar.empty()
    if F.shape[1] != len(heads["plane_mean"]):
        raise ValueError(f"Feature size {F.shape[1]} does not match the exported classifier "
                         f"({len(heads['plane_mean'])}). Re-export both files from the same notebook run.")
    return {"imgs": imgs, "F": F, "ids": ids, "n_total": n_total}


def render_visit(sweep, signs, midwife, cfg, heads, tau, tau_sweep, digest):
    P = plane_probs(sweep["F"], heads)
    v = assess_visit(P, sweep["F"], sweep["n_total"], cfg, heads, tau, tau_sweep, signs)
    visit_id = "V-" + digest[:6].upper()

    st.markdown(card_html(v["colour"], v["title"], v["action"]), unsafe_allow_html=True)
    st.caption(v["placeholder"])

    st.table(pd.DataFrame(
        [("1. Danger signs", v["signs"]), ("2. Scan quality", v["scan"]),
         ("3. Views captured", ", ".join(v["found"]) or "none"), ("4. Breech flag", v["breech_txt"]),
         ("5. Brain check (exploratory)", v["brain_txt"]),
         ("6. Spine, abdomen, femur checks", "not available yet (future work: needs labelled data, ideally Kenyan)")],
        columns=["Step", "Result"]).set_index("Step"))

    if v["good"]:
        st.markdown("**Best frame per view** (for remote review)")
        cols = st.columns(5)
        for col, (k, info) in zip(cols, v["views"].items()):
            with col:
                st.image(sweep["imgs"][info["frame"]], clamp=True,
                         caption=f"{VIEW_LABELS[k]}: {'✓' if info['found'] else 'not found'}\n"
                                 f"frame {sweep['ids'][info['frame']]} · p = {info['prob']:.2f}")

    if v["key"] in ("urgent", "breech", "defect"):
        note = referral_note(v, visit_id, midwife)
        st.markdown("**Referral note**")
        st.code(note, language=None)
        st.download_button("Download referral note (.txt)", note, file_name=f"referral_{visit_id}.txt")

    with st.expander("Frame-by-frame scores"):
        p_plane = 1 - P[:, NO_PLANE]
        st.line_chart(pd.DataFrame({"P(plane)": p_plane, "frame threshold": tau, "scan threshold": tau_sweep},
                                   index=sweep["ids"]))
        table = pd.DataFrame(P, columns=CLASS_NAMES)
        table.insert(0, "frame", sweep["ids"])
        table["P(plane)"] = p_plane
        table["decision"] = [CLASS_NAMES[i] for i in decide(P, tau)]
        st.download_button("Download per-frame probabilities (.csv)", table.to_csv(index=False),
                           file_name=f"frames_{visit_id}.csv", mime="text/csv")
    st.caption(f"Visit {visit_id} · {date.today().isoformat()} · {sweep['n_total']} frames uploaded, "
               f"{len(sweep['ids'])} analysed. {FOOTER}")


def about_tab(cfg):
    st.markdown(
        "**What it does.** A lightweight model looks at every frame of a blind sweep and finds the five ISUOG standard "
        "planes (head, abdomen, heart, spine, femur). It then decides whether the sweep should be repeated and "
        "suggests the best frames for remote review.\n\n"
        "**Method.** Frozen ImageNet features (EfficientNet-B0 + MobileNetV2) → logistic regression with hard-negative "
        "mining, thresholds tuned on held-out sweeps. The breech flag is a second logistic regression on the "
        "average features of a whole sweep.\n\n"
        "**Data.** NatalIA PBF-US1 (90 blind sweeps of a 23-week *phantom*), FETAL_PLANES_DB (real images, Spain) and "
        "an African fetal-planes set used for external testing.")
    if cfg.get("summary"):
        st.markdown("**Headline results from the notebook**")
        st.dataframe(pd.DataFrame(cfg["summary"]), hide_index=True)
    st.warning(
        "**Limitations.** Trained on phantom sweeps from one probe and one gestational age; few positive frames, so "
        "uncertainty is wide; the plane finder reached macro-F1 well below the 0.70 target; the rescan decision "
        "flags too few empty sweeps for unsupervised use; real-patient performance has not been validated, and "
        "none of the data is from Kenya.")
    st.markdown(
        "**Responsible use.** Decision support only: the tool never diagnoses or decides treatment; danger signs "
        "always override the scan; a named clinician makes every decision. Real use needs clinical validation in "
        "Kenya, regulatory approval, consent and data protection. Guidance texts are placeholders until taken from "
        "the Kenyan MOH guidelines and approved by a clinician. Uploaded frames are processed in memory and not stored.\n\n"
        "**Data citations.** González et al. (2024), NatalIA PBF-US1, doi:10.5281/zenodo.14193949 · "
        "Burgos-Artizzu et al. (2020), FETAL_PLANES_DB, zenodo.org/record/3904280 · "
        "Sendra-Balcells et al. (2023), African fetal planes, zenodo.org/records/7540448.")


def main():
    st.set_page_config(page_title="Taswira Analytics", page_icon="🩺", layout="wide")
    st.title("🩺 Taswira Analytics")
    st.caption("Finding standard fetal planes in novice blind sweeps · Group 11 capstone · "
               "**study prototype, decision support only**")

    needed = ["config.json", "heads.npz", "taswira_features.tflite"]
    missing = [n for n in needed if not (ART_DIR / n).exists()]
    if missing:
        st.error(f"Missing model files in `artifacts/`: {', '.join(missing)}. Run `export_artifacts_cell.py` as the "
                 "last cell of the notebook and copy the `artifacts/` folder next to `app.py`.")
        st.stop()
    cfg, heads, extractor = load_artifacts()

    with st.sidebar:
        st.header("Visit")
        midwife = st.text_input("Midwife (optional)", placeholder="name")
        st.info("Trained on a phantom. Not validated on real patients.")
        with st.expander("Study settings (advanced)"):
            mode_label = st.selectbox("Frame preprocessing", list(PREP_MODES),
                                      help="The training probe's frames use a fixed crop. Frames from other devices "
                                           "get their ultrasound sector detected automatically.")
            stride = st.number_input("Analyse every N-th frame", 1, 10, 1,
                                     help="2 halves the waiting time; a full sweep (~200 frames) is best.")
            tau = st.slider("Frame threshold τ (plane if P(plane) ≥ τ)", 0.0, 1.0, float(cfg["tau"]), 0.01)
            tau_sweep = st.slider("Scan threshold τ_sweep (good scan if best frame ≥ τ_sweep)", 0.0, 1.0,
                                  float(cfg["tau_sweep"]), 0.005, format="%.3f")
        mode = PREP_MODES[mode_label]

    tab_visit, tab_about = st.tabs(["Midwife visit", "About this prototype"])
    with tab_about:
        about_tab(cfg)

    with tab_visit:
        st.subheader("1 · Danger-sign check (ask the mother first)")
        cols = st.columns(2)
        signs = [s for i, s in enumerate(DANGER_SIGNS) if cols[i % 2].checkbox(s, key=f"danger_{i}")]
        if signs:
            colour, title, action, placeholder = CARDS["urgent"]
            st.markdown(card_html(colour, title, action), unsafe_allow_html=True)
            st.caption(placeholder)

        st.subheader("2 · Upload the sweep")
        files = st.file_uploader(
            "One sweep per visit: image frames, a ZIP of frames, or a video (about 200 frames is ideal)",
            type=[e.lstrip(".") for e in sorted(IMG_EXT | VIDEO_EXT | {".zip"})], accept_multiple_files=True)
        digest = uploads_digest(files) if files else None
        settings = (mode, int(stride))

        if st.button("Analyse sweep", type="primary", disabled=not files):
            try:
                result = run_analysis(files, cfg, heads, extractor, mode, int(stride))
            except Exception as e:  # noqa: BLE001  (show a readable message instead of a stack trace)
                st.error(f"Could not analyse the upload: {e}")
                result = None
            if result is None:
                st.warning("No readable image frames were found in the upload.")
            else:
                st.session_state["sweep"] = {**result, "digest": digest, "settings": settings}

        sweep = st.session_state.get("sweep")
        if sweep and files and sweep["digest"] == digest and sweep["settings"] == settings:
            st.subheader("3 · Result")
            if len(sweep["ids"]) < 30:
                st.warning("Very few frames were analysed; results are unreliable. Upload a full sweep.")
            render_visit(sweep, signs, midwife, cfg, heads, tau, tau_sweep, digest)
        elif files:
            st.info("Press **Analyse sweep** to analyse the current upload.")
        else:
            st.info("Upload a sweep to begin. Danger signs always come first and override the scan.")


if __name__ == "__main__":
    main()
