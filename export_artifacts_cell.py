# =====================================================================================
# EXPORT FOR THE STREAMLIT APP  ->  paste as the LAST cell of the notebook and run it.
# (Needs sections 0-10 to have run: it reuses CV, MODELS, DEPLOY_NAME, XS, yb, SUMMARY, ...)
# Creates ./artifacts/ with:
#   taswira_features.tflite  both backbones -> 2,560-d features
#   heads.npz                plane classifier, breech classifier (+ brain check if section 8 ran)
#   config.json              crop box, thresholds, headline results
# =====================================================================================
QUANTIZE = False        # False = exact features (~25 MB). True = ~7 MB, features differ very slightly.

ART_DIR = NOTEBOOK_DIR / "artifacts"
ART_DIR.mkdir(exist_ok=True)

# 1) Feature extractor (same backbones and weights as the notebook cache) -> TFLite
f_in = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="frame")
f_out = layers.Concatenate()([get_backbone(b)(f_in) for b in FEATURE_BACKBONES])
FEAT = keras.Model(f_in, f_out, name="taswira_features")
conv = tf.lite.TFLiteConverter.from_keras_model(FEAT)
if QUANTIZE:
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
tfl_feat = conv.convert()
(ART_DIR / "taswira_features.tflite").write_bytes(tfl_feat)

# Sanity check: TFLite features must match the cached notebook features
interp_f = tf.lite.Interpreter(model_content=tfl_feat); interp_f.allocate_tensors()
fi, fo = interp_f.get_input_details()[0], interp_f.get_output_details()[0]
chk_i = np.arange(0, len(frames), max(len(frames) // 12, 1))[:12]
cos = []
for i in chk_i:
    x = np.repeat(np.asarray(IMAGES[frames["row"].values[i]])[None, ..., None], 3, -1).astype(np.float32)
    interp_f.set_tensor(fi["index"], x); interp_f.invoke()
    a, b = interp_f.get_tensor(fo["index"])[0], X[i]
    cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
print(f"TFLite feature extractor: {len(tfl_feat) / 1e6:.1f} MB, min cosine vs notebook features = {min(cos):.4f}")

# 2) Plane classifier (refit exactly as for the deployed model) as plain arrays
sc_p, clf_p = train(np.arange(len(frames)), np.random.default_rng(SEED), MODELS[DEPLOY_NAME])
W_plane, b_plane = np.zeros((X.shape[1], 6), np.float32), np.full(6, -1e4, np.float32)
W_plane[:, clf_p.classes_], b_plane[clf_p.classes_] = clf_p.coef_.T, clf_p.intercept_
arrays = {"plane_mean": sc_p.mean_, "plane_scale": sc_p.scale_, "plane_W": W_plane, "plane_b": b_plane}

# 3) Breech classifier trained on ALL sweeps (section 7 used the same settings inside cross-validation)
sc_b = StandardScaler().fit(XS)
clf_b_all = LogisticRegression(C=BREECH_C, max_iter=3000, class_weight="balanced").fit(sc_b.transform(XS), yb)
arrays.update({"breech_mean": sc_b.mean_, "breech_scale": sc_b.scale_,
               "breech_coef": clf_b_all.coef_[0], "breech_b": clf_b_all.intercept_[0:1]})

# 4) Thresholds: median over the cross-validation folds of the deployed model
params = CV[DEPLOY_NAME]["params"].values()
config = {
    "deploy_model": DEPLOY_NAME,
    "feature_model": "taswira_features.tflite",
    "crop": [int(v) for v in CROP],                                     # NatalIA sector crop box
    "natalia_size": list(Image.open(frames["path"].iloc[0]).size),      # (width, height) of NatalIA frames
    "tau": float(np.median([p["tau"] for p in params])),
    "tau_sweep": float(np.median([p["tau_sweep"] for p in params])),
    "summary": [{k: str(v) for k, v in r.items()} for r in SUMMARY.to_dict("records")] if "SUMMARY" in globals() else [],
}

# 5) Optional: exploratory brain check (only if section 8 ran with the Roboflow data)
if globals().get("HAVE_ANOM", False) and "TAU_ANOM" in globals():
    n_a, d = len(A_CLASSES), X.shape[1]
    W_a, b_a = np.zeros((d, n_a), np.float32), np.full(n_a, -1e4, np.float32)
    if len(clf_a.classes_) == 2:               # binary case: softmax([0, z]) = [1 - sigmoid(z), sigmoid(z)]
        c0, c1 = clf_a.classes_
        W_a[:, c0], b_a[c0] = 0, 0
        W_a[:, c1], b_a[c1] = clf_a.coef_[0], clf_a.intercept_[0]
    else:
        W_a[:, clf_a.classes_], b_a[clf_a.classes_] = clf_a.coef_.T, clf_a.intercept_
    arrays.update({"anom_mean": sc_a.mean_, "anom_scale": sc_a.scale_, "anom_W": W_a, "anom_b": b_a,
                   "anom_UA": UA.astype(np.float16)})
    config["anom"] = {"classes": list(A_CLASSES), "normal_index": int(ni), "tau_anom": float(TAU_ANOM),
                      "tau_refer": float(TAU_REFER), "ood_limit": float(OOD_LIMIT), "nice": dict(NICE)}
    print("Brain check exported.")
else:
    print("Brain check NOT exported (section 8 not run): the app will show 'unavailable'.")

np.savez_compressed(ART_DIR / "heads.npz", **{k: np.asarray(v, np.float32) for k, v in arrays.items()})
(ART_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
print("Saved to", ART_DIR, "->", sorted(p.name for p in ART_DIR.iterdir()))
print(f"tau = {config['tau']:.2f}, tau_sweep = {config['tau_sweep']:.3f}, crop = {config['crop']}")
