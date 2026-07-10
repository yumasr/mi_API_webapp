# MI Risk Web App — Google Drive model version

Third teaching variant for the AI/ML lecture. The trained model (`MI_model.pkl`)
is stored on **Google Drive**; the Streamlit app downloads it at startup using a
**Google API key kept in Streamlit secrets** — the same secrets pattern as the
official tutorial for Google Cloud Storage
(https://docs.streamlit.io/develop/tutorials/databases/gcs), adapted to the
Google Drive API.

```
[ Streamlit app ]  --API key (from st.secrets)-->  [ Google Drive ]  -->  MI_model.pkl
  holds only the key + file id                       hosts the model file
```

## How this compares to the other two variants

| Variant | Where the model lives | Secret needed |
|---|---|---|
| `../MI_webapp` | Behind a FastAPI server you run | Your own API key (X-API-Key) |
| `../MI_webapp_local` | Inside the app folder | None |
| **This app** | On Google Drive | Google API key in `st.secrets` |

Teaching point: the model file is **not** committed with the app code — you can
update the model on Drive without redeploying the app (the app re-downloads it
after the 1-hour cache expires). However, a plain API key only reads files
shared as *"Anyone with the link"*, so the model is not truly private — anyone
with the file ID can also download it. For a truly private model, use a service
account (as the GCS tutorial does) or the backend approach in `../MI_webapp`.

---

## Setup

### 1. Put the model on Google Drive
1. Upload `MI_model.pkl` to your Google Drive.
2. Right-click → **Share** → General access: **Anyone with the link** (Viewer).
3. Copy the link, e.g. `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`,
   and note the **FILE_ID** part.

### 2. Create a Google API key
1. Go to https://console.cloud.google.com/ and create (or pick) a project.
2. **APIs & Services → Library** → search **Google Drive API** → **Enable**.
3. **APIs & Services → Credentials → Create credentials → API key**. Copy it.
4. (Recommended) Restrict the key to the Google Drive API only.

### 3. Configure secrets
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml: fill in api_key and file_id
```
`.streamlit/secrets.toml` is git-ignored — never commit it. On Streamlit
Community Cloud, paste the same content under **App → Settings → Secrets**.

### 4. Install and run
```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the Streamlit URL, fill the form, press **ทำนายผล / Predict**.

---

## Inputs

`name` (display only), `weight` (kg), `height` (cm), `FBS` (mg/dL),
`HDL` (mg/dL), `LDL` (mg/dL), `age` (years).

Feature order sent to the model — **must not change** — is
`["weight", "height", "FBS", "HDL", "LDL", "AGE"]`, matching `ML_MI_complete.ipynb`.

Label meaning: `DIAG_MI` where **0 = MI**, **1 = Normal**, so
`P(MI) = predict_proba[:, 0]`.

## Notes

- `MI_model.pkl` holds a fitted `Pipeline(StandardScaler, LogisticRegression)`,
  so single-patient inputs are scaled with the training data's mean/std — no
  scaler is refit in the app.
- The pickle must be loadable by the scikit-learn version in
  `requirements.txt`. If the model was exported from Colab, keep the
  scikit-learn versions compatible (a version-mismatch warning or error on
  load means retrain/re-export with a matching version).
- This tool is for teaching only — not a medical diagnosis.
