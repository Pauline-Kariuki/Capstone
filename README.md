# Taswira Analytics: Streamlit app

## Folder layout (what goes on GitHub)
```
app.py
requirements.txt
.streamlit/config.toml
artifacts/                  <- created by the notebook (step 1)
    config.json
    heads.npz
    taswira_features.tflite
```

## 1. Export the trained model from the notebook
1. Run the notebook as usual (at least through section 10; section 8 is optional and enables the brain check).
2. Add `export_artifacts_cell.py` as the **last cell** and run it.
3. Copy the new `artifacts/` folder next to `app.py`.

The export cell prints a check that the TFLite features match the notebook's cached features
(cosine should be ~1.000), plus the thresholds and crop box it saved.

## 2. Test locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 3. Deploy on Streamlit Community Cloud
1. Push the folder to a GitHub repo (the `.tflite` is ~25 MB, within GitHub's limits).
2. share.streamlit.io -> New app -> pick the repo -> main file `app.py`.
3. Advanced settings -> Python 3.11 or 3.12.
4. If `ai-edge-litert` fails to install, replace it with `tensorflow-cpu` in `requirements.txt`
   (the app falls back to it automatically).

## Notes
- Frames are processed in memory and never stored. No personal data is collected.
- A full sweep (~200 frames) takes about 40-100 s on shared cloud CPU. Use "Analyse every N-th frame"
  in the sidebar to speed it up.
- Guidance texts and the danger-sign list are placeholders until approved by a clinician (as in the notebook).
- If the public repo is a concern (patient-adjacent project), deploy privately or add authentication.
