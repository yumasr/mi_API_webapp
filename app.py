"""
MI risk web app — GOOGLE DRIVE model version (API key in Streamlit secrets).

Difference from the other two variants:
    ../MI_webapp        model hidden behind a FastAPI server, gated by a secret key
    ../MI_webapp_local  model file ships inside the app folder, loaded directly
    THIS app            model file lives on Google Drive; the app downloads it at
                        startup with a Google API key kept in .streamlit/secrets.toml

This follows the same secrets pattern as the official Streamlit tutorial for
Google Cloud Storage (https://docs.streamlit.io/develop/tutorials/databases/gcs):
credentials live in .streamlit/secrets.toml locally (git-ignored) and in
App -> Settings -> Secrets on Streamlit Community Cloud — never in the code.

Setup (see README.md for full steps):
    1. Upload MI_model.pkl to Google Drive, share as "Anyone with the link".
    2. Create a Google API key with the Drive API enabled.
    3. Copy .streamlit/secrets.toml.example to .streamlit/secrets.toml and
       fill in the api_key and file_id.

Run:
    streamlit run app.py
"""

import io
import pickle

import pandas as pd
import requests
import streamlit as st

# --- Feature order MUST match training (ML_MI_complete.ipynb) ---------------
FEATURE_NAMES = ["weight", "height", "FBS", "HDL", "LDL", "AGE"]

# Label meaning from the notebook: DIAG_MI where 0 = MI, 1 = Normal.
# model.classes_ == [0, 1], so predict_proba[:, 0] is P(MI).
MI_CLASS_INDEX = 0

# Google Drive API v3 "download file content" endpoint. Works with a plain
# API key only when the file is shared as "Anyone with the link".
GDRIVE_DOWNLOAD_URL = "https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={api_key}"


@st.cache_resource(ttl=3600)
def load_model_from_gdrive():
    # Secrets come from .streamlit/secrets.toml locally, or from the app's
    # Secrets settings on Streamlit Community Cloud — never hard-coded here.
    api_key = st.secrets["gdrive"]["api_key"]
    file_id = st.secrets["gdrive"]["file_id"]

    resp = requests.get(
        GDRIVE_DOWNLOAD_URL.format(file_id=file_id, api_key=api_key),
        timeout=30,
    )
    resp.raise_for_status()

    # MI_model.pkl is a fitted Pipeline(StandardScaler, LogisticRegression),
    # so the scaler inside was fit on the training data and single patients
    # are scaled correctly without refitting anything here.
    return pickle.load(io.BytesIO(resp.content))


st.set_page_config(page_title="ทำนายความเสี่ยงกล้ามเนื้อหัวใจตาย (MI)", page_icon="❤️")

st.title("❤️ ทำนายความเสี่ยงกล้ามเนื้อหัวใจตาย (Myocardial Infarction)")
st.caption(
    "กรอกข้อมูลผู้ป่วยเพื่อประเมินความเสี่ยง — เวอร์ชันนี้ดาวน์โหลดโมเดลจาก Google Drive "
    "โดยใช้ API key ที่เก็บไว้ใน Streamlit secrets"
)

try:
    gdrive_secrets = st.secrets["gdrive"]
    _ = gdrive_secrets["api_key"], gdrive_secrets["file_id"]
except (KeyError, FileNotFoundError):
    st.error(
        "ยังไม่ได้ตั้งค่า secrets — copy `.streamlit/secrets.toml.example` เป็น "
        "`.streamlit/secrets.toml` แล้วใส่ `api_key` และ `file_id` ของ Google Drive "
        "(ดูขั้นตอนใน README.md)"
    )
    st.stop()

try:
    with st.spinner("กำลังดาวน์โหลดโมเดลจาก Google Drive..."):
        model = load_model_from_gdrive()
except requests.HTTPError as e:
    st.error(
        f"ดาวน์โหลดโมเดลไม่สำเร็จ (HTTP {e.response.status_code}) — "
        "ตรวจสอบว่า file_id ถูกต้อง, ไฟล์แชร์เป็น 'Anyone with the link' "
        "และ API key เปิดใช้ Google Drive API แล้ว"
    )
    st.stop()

with st.form("patient_form"):
    name = st.text_input("ชื่อ (Name)", "")
    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input("น้ำหนัก / Weight (kg)", min_value=0.0, max_value=300.0, value=70.0, step=0.5)
        fbs = st.number_input("น้ำตาลในเลือด / FBS (mg/dL)", min_value=0.0, max_value=600.0, value=100.0, step=1.0)
        ldl = st.number_input("LDL cholesterol (mg/dL)", min_value=0.0, max_value=400.0, value=120.0, step=1.0)
    with col2:
        height = st.number_input("ส่วนสูง / Height (cm)", min_value=0.0, max_value=250.0, value=170.0, step=0.5)
        hdl = st.number_input("HDL cholesterol (mg/dL)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
        age = st.number_input("อายุ / Age (years)", min_value=0.0, max_value=120.0, value=45.0, step=1.0)

    submitted = st.form_submit_button("ทำนายผล / Predict")

if submitted:
    # The model was downloaded once and cached — prediction runs in-process.
    row = pd.DataFrame(
        [[weight, height, fbs, hdl, ldl, age]],
        columns=FEATURE_NAMES,
    )

    proba = model.predict_proba(row)[0]
    p_mi = float(proba[MI_CLASS_INDEX])
    pred_label = int(model.predict(row)[0])  # 0 = MI, 1 = Normal

    who = f"คุณ {name} " if name else ""
    if pred_label == 0:
        st.error(f"⚠️ {who}มีความเสี่ยงต่อภาวะ MI (ความน่าจะเป็น = {p_mi:.1%})")
    else:
        st.success(f"✅ {who}ไม่พบความเสี่ยงต่อภาวะ MI (ความน่าจะเป็น MI = {p_mi:.1%})")

    st.progress(min(max(p_mi, 0.0), 1.0))
    st.caption("หมายเหตุ: ผลลัพธ์นี้ใช้เพื่อการเรียนการสอนเท่านั้น ไม่ใช่คำวินิจฉัยทางการแพทย์")
