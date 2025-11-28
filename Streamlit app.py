# ssism_prm_mvp.py

# Single-file Streamlit MVP for SSISM-PRM intake, RSS, triage, evidence JSON schema, and brief generation.

# Requirements: see requirements list above.

# Run: streamlit run ssism_prm_mvp.py

# Author: Assistant (for your SSISM-PRM project). Edit/extend as needed.



import os

import io

import json

import time

import hashlib

import datetime

import tempfile

from pathlib import Path

from typing import Dict, Any, List, Optional



import streamlit as st

import feedparser

from PIL import Image, ExifTags

import piexif

from langdetect import detect, LangDetectException



# Optional LLM (OpenAI)

try:

    import openai

    OPENAI_AVAILABLE = True

except Exception:

    OPENAI_AVAILABLE = False



# ---------- Config / folders ----------

BASE_DIR = Path.cwd()

EVIDENCE_DIR = BASE_DIR / "evidence_store"

MEDIA_DIR = EVIDENCE_DIR / "media"

EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

MEDIA_DIR.mkdir(parents=True, exist_ok=True)



# Default RSS feeds you can change in UI:

DEFAULT_RSS_FEEDS = [

    # Replace or keep empty; recommended to add official NLD/NUG feeds in the UI.

    "https://www.irrawaddy.com/feed/",   # example public feed

]



# ---------- Utilities ----------

def now_ts():

    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()



def sha256_bytes(data: bytes) -> str:

    return hashlib.sha256(data).hexdigest()



def save_media_file(uploaded_file) -> Dict[str, Any]:

    """Save uploaded file to disk and return metadata dict."""

    raw = uploaded_file.read()

    h = sha256_bytes(raw)

    fname = uploaded_file.name

    ext = Path(fname).suffix or ""

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")

    safe_name = f"{timestamp}_{h[:8]}{ext}"

    path = MEDIA_DIR / safe_name

    with open(path, "wb") as f:

        f.write(raw)

    return {

        "media_id": safe_name,

        "filename": fname,

        "saved_path": str(path),

        "hash_sha256": h,

        "size_bytes": len(raw),

    }



def extract_image_exif(path: str) -> Dict[str, Any]:

    """Extract basic EXIF; return timestamps and GPS if present."""

    try:

        img = Image.open(path)

    except Exception:

        return {}

    exif = {}

    try:

        exif_bytes = img.info.get("exif")

        if exif_bytes:

            ex = piexif.load(exif_bytes)

            # extract DateTimeOriginal if present

            dto = ex["0th"].get(piexif.ImageIFD.DateTime)

            if dto:

                try:

                    exif["datetime"] = dto.decode("utf-8")

                except:

                    exif["datetime"] = str(dto)

            # GPS

            gps_ifd = ex.get("GPS", {})

            if gps_ifd:

                # convert rational to float if present

                def _to_deg(values):

                    # values are tuples of (num,den)

                    def conv(t):

                        return t[0]/t[1] if (isinstance(t, tuple) and t[1]!=0) else float(t)

                    try:

                        d = conv(values[0]); m = conv(values[1]); s = conv(values[2])

                        return d + (m/60.0) + (s/3600.0)

                    except Exception:

                        return None

                lat = gps_ifd.get(piexif.GPSIFD.GPSLatitude)

                lat_ref = gps_ifd.get(piexif.GPSIFD.GPSLatitudeRef)

                lon = gps_ifd.get(piexif.GPSIFD.GPSLongitude)

                lon_ref = gps_ifd.get(piexif.GPSIFD.GPSLongitudeRef)

                if lat and lon:

                    latf = _to_deg(lat)

                    lonf = _to_deg(lon)

                    if lat_ref and lon_ref:

                        if lat_ref.decode("utf-8") in ['S','s']:

                            latf = -latf

                        if lon_ref.decode("utf-8") in ['W','w']:

                            lonf = -lonf

                    exif["gps"] = {"lat": latf, "lon": lonf}

    except Exception:

        pass

    return exif



def detect_language(text: str) -> str:

    try:

        return detect(text)

    except LangDetectException:

        return "unknown"



# ---------- Credibility scoring function (use the formula discussed) ----------

def compute_credibility_score(features: Dict[str, float]) -> float:

    """

    features expected keys:

     - sr: source reputation 0..1

     - cc: corroboration count derived 0..1

     - ma: metadata authenticity 0..1

     - tr: temporal recency 0..1

     - geo: geolocation consistency 0..1

     - fi: file integrity 0..1

    """

    weights = {"sr":0.4, "cc":0.25, "ma":0.15, "tr":0.05, "geo":0.05, "fi":0.05}

    score = 0.0

    for k,w in weights.items():

        score += features.get(k, 0.0) * w

    return round(score, 4)



# ---------- Evidence JSON creation ----------

def create_evidence_json(submission: Dict[str,Any], media_list: List[Dict[str,Any]], auto_features: Dict[str,float]) -> Dict[str,Any]:

    sid = submission.get("submission_id") or f"prm-{int(time.time())}"

    evidence = {

        "submission_id": sid,

        "received_at": now_ts(),

        "submitted_via": submission.get("submitted_via","manual"),

        "submitter_alias": submission.get("submitter_alias","source-unknown"),

        "consent_level": submission.get("consent_level","anonymous_publication"),

        "language": submission.get("language","my"),

        "location": submission.get("location", {}),

        "media": media_list,

        "claim_summary": submission.get("claim_summary",""),

        "claimed_sources": submission.get("claimed_sources", []),

        "initial_credibility_est": auto_features.get("est", 0.0),

        "features_used": auto_features,

        "attached_documents": submission.get("attached_documents", []),

        "evidence_tags": submission.get("evidence_tags", []),

        "analyst_notes": submission.get("analyst_notes",""),

        "ingest_status": "new"

    }

    # save as file

    fp = EVIDENCE_DIR / f"{sid}.json"

    with open(fp, "w", encoding="utf-8") as f:

        json.dump(evidence, f, ensure_ascii=False, indent=2)

    return evidence



# ---------- RSS Fetching ----------

def fetch_rss_items(feed_urls: List[str], max_items=10) -> List[Dict[str,Any]]:

    items = []

    for url in feed_urls:

        try:

            d = feedparser.parse(url)

            for e in d.entries[:max_items]:

                item = {

                    "feed_url": url,

                    "title": e.get("title"),

                    "link": e.get("link"),

                    "published": e.get("published", e.get("updated")),

                    "summary": e.get("summary") or e.get("description",""),

                    "raw": e

                }

                items.append(item)

        except Exception as exc:

            st.warning(f"Failed to parse feed {url}: {exc}")

    return items



# ---------- LLM Brief generation (optional, uses OpenAI) ----------

SYSTEM_PROMPT_BRIEF = (

    "You are PRM Brief Assistant. Use only the evidence chunks provided. "

    "Produce a concise 1-page Situation Brief with: 1) Headline (<=12 words), "

    "2) 5 numbered findings (each 1 sentence) with inline [Source] markers, "

    "3) One paragraph implications, and 4) Three recommended actions. "

    "If information is not in evidence, mark it as [UNVERIFIED]. Do not hallucinate."

)



def generate_brief_with_openai(evidence: Dict[str,Any], api_key: str) -> str:

    if not OPENAI_AVAILABLE:

        return "OpenAI library not available on server."

    openai.api_key = api_key

    # assemble context: short strings from claim_summary + top media filenames + tags

    context_pieces = []

    context_pieces.append(f"Claim summary: {evidence.get('claim_summary','')}")

    for m in evidence.get("media", [])[:4]:

        context_pieces.append(f"Media: {m.get('filename')} (saved:{m.get('saved_path')})")

    context_pieces.append("Tags: " + ", ".join(evidence.get("evidence_tags", [])))

    context = "\n\n".join(context_pieces)

    prompt = f"{SYSTEM_PROMPT_BRIEF}\n\nEVIDENCE:\n{context}\n\nProduce the brief now."

    try:

        resp = openai.ChatCompletion.create(

            model="gpt-4o-mini",  # use the model you have access to; change as needed

            messages=[

                {"role":"system","content":SYSTEM_PROMPT_BRIEF},

                {"role":"user","content":context}

            ],

            temperature=0.0,

            max_tokens=800,

        )

        return resp["choices"][0]["message"]["content"]

    except Exception as exc:

        return f"OpenAI call failed: {exc}"



# ---------- Simple local brief generator (deterministic fallback) ----------

def generate_brief_local(evidence: Dict[str,Any]) -> str:

    headline = evidence.get("claim_summary","").strip()

    if len(headline) > 60:

        headline = headline[:57] + "..."

    if not headline:

        headline = "Situation Brief — New Submission"

    findings = []

    findings.append(f"1) Summary: {evidence.get('claim_summary','[no summary provided]')}")

    if evidence.get("media"):

        findings.append(f"2) Media attached: {len(evidence.get('media'))} files ({', '.join([m['filename'] for m in evidence.get('media')])})")

    else:

        findings.append("2) No media attached.")

    tags = ", ".join(evidence.get("evidence_tags", [])) or "none"

    findings.append(f"3) Tags: {tags}")

    findings.append(f"4) Initial credibility score: {evidence.get('initial_credibility_est',0.0)}")

    findings.append("5) Recommendation: Attempt corroboration via other local sources and satellite imagery.")

    implications = ("Implication: If corroborated, this event suggests a continuing pattern of criminal economies "

                    "operating in border zones that could finance armed activities. Treat as intelligence lead, not confirmed fact.")

    actions = ["1) Contact vetted local stringers for corroboration.",

               "2) Request satellite imagery for the reported date & location.",

               "3) Flag to PRM analysts for triage and legal review before publication."]

    text = f"Headline: {headline}\n\nFindings:\n" + "\n".join(findings) + f"\n\nImplications:\n{implications}\n\nRecommended actions:\n" + "\n".join(actions)

    return text



# ---------- UI ----------

st.set_page_config(page_title="SSISM-PRM MVP — Intake & Triage", layout="wide")

st.title("SSISM-PRM MVP — Intake & Triage (VIP)")



# Left column: submission

col1, col2 = st.columns([1,2])



with col1:

    st.header("Submit Evidence")

    with st.form("submit_form"):

        submitter_alias = st.text_input("Submitter alias / code name", value="source-unknown")

        submitted_via = st.selectbox("Submitted via", ["manual", "signal", "element", "secureform", "pgp_email"])

        consent_level = st.selectbox("Consent level", ["anonymous_publication", "attributed", "private_internal"])

        language = st.selectbox("Language", ["my", "en", "other"])

        location_text = st.text_input("Location (text)", value="")

        coords_lat = st.text_input("Location latitude (optional)", value="")

        coords_lon = st.text_input("Location longitude (optional)", value="")

        claim_summary = st.text_area("Short claim summary / what happened", height=120)

        claimed_sources_raw = st.text_input("Claimed sources (comma-separated pseudonyms)", value="")

        tags_raw = st.text_input("Initial tags (comma-separated)", value="")

        uploaded_files = st.file_uploader("Upload photo/video/document (multiple allowed)", accept_multiple_files=True)

        submitted = st.form_submit_button("Create evidence entry")



    if submitted:

        # Save files

        media_list = []

        for up in uploaded_files:

            meta = save_media_file(up)

            # try EXIF for images

            exif = {}

            try:

                exif = extract_image_exif(meta["saved_path"])

            except Exception as e:

                exif = {}

            meta["exif"] = exif

            media_list.append(meta)



        # auto feature heuristics (very simple)

        sr = 0.5  # default new source reputation

        num_claimed = len([s for s in claimed_sources_raw.split(",") if s.strip()])

        cc = min(1.0, (1.0 if num_claimed>=2 else 0.0))  # naive

        ma = 1.0 if any(m.get("exif") for m in media_list) else 0.5

        tr = 1.0  # assume current

        geo = 0.6 if coords_lat and coords_lon else (0.3 if location_text else 0.0)

        fi = 1.0  # we saved file locally, assume integrity verified via hash

        features = {"sr": sr, "cc": cc, "ma": ma, "tr": tr, "geo": geo, "fi": fi}

        est = compute_credibility_score(features)

        features["est"] = est



        submission = {

            "submission_id": f"prm-{int(time.time())}",

            "submitted_via": submitted_via,

            "submitter_alias": submitter_alias,

            "consent_level": consent_level,

            "language": language,

            "location":{

                "text": location_text,

                "coords": {"lat": float(coords_lat) if coords_lat else None, "lon": float(coords_lon) if coords_lon else None}

            },

            "claim_summary": claim_summary,

            "claimed_sources": [s.strip() for s in claimed_sources_raw.split(",") if s.strip()],

            "evidence_tags": [t.strip() for t in tags_raw.split(",") if t.strip()]

        }

        evidence = create_evidence_json(submission, media_list, features)

        st.success(f"Evidence created: {evidence['submission_id']} (credibility {features['est']})")

        st.json(evidence)



# Right column: RSS and evidence list

with col2:

    st.header("RSS Feeds — add official feeds (NLD / NUG / others)")

    feeds_text = st.text_area("RSS feed URLs (one per line)", value="\n".join(DEFAULT_RSS_FEEDS), height=120)

    max_items = st.number_input("Max items per feed", min_value=1, max_value=50, value=8)

    if st.button("Fetch RSS now"):

        feed_urls = [u.strip() for u in feeds_text.splitlines() if u.strip()]

        rss_items = fetch_rss_items(feed_urls, max_items=max_items)

        st.success(f"Fetched {len(rss_items)} items from {len(feed_urls)} feeds.")

        for it in rss_items:

            st.markdown(f"**{it['title']}** — {it.get('published','')}")

            st.write(it.get("summary",""))

            st.markdown(f"[Source link]({it.get('link')})")

            st.divider()



    st.header("Evidence store — existing entries")

    # list JSON files

    entries = sorted(EVIDENCE_DIR.glob("prm-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not entries:

        st.info("No evidence entries yet.")

    else:

        sel = st.selectbox("Select evidence to inspect", options=[p.name for p in entries])

        if sel:

            p = EVIDENCE_DIR / sel

            with open(p, "r", encoding="utf-8") as f:

                e = json.load(f)

            st.subheader(f"Evidence: {e['submission_id']}")

            st.write("Received at:", e.get("received_at"))

            st.write("Submitter:", e.get("submitter_alias"), "| Consent:", e.get("consent_level"))

            st.write("Credibility (initial):", e.get("initial_credibility_est"))

            st.write("Tags:", ", ".join(e.get("evidence_tags",[])))

            st.write("Claim summary:")

            st.write(e.get("claim_summary",""))



            st.write("Media files:")

            for m in e.get("media", []):

                st.write("-", m.get("filename"), "| hash:", m.get("hash_sha256")[:12], "| saved:", m.get("saved_path"))

                # if image show thumbnail

                try:

                    if Path(m.get("saved_path")).exists():

                        im = Image.open(m.get("saved_path"))

                        st.image(im, width=300)

                except Exception:

                    pass



            st.write("Features used:")

            st.json(e.get("features_used", {}))



            st.write("Analyst actions:")

            analyst_action = st.radio("Action", ["monitor", "corroborate", "corroborated", "reject"], index=0)

            analyst_note = st.text_area("Analyst note", value=e.get("analyst_notes",""))

            if st.button("Save analyst update"):

                e["ingest_status"] = analyst_action

                e["analyst_notes"] = analyst_note

                with open(p, "w", encoding="utf-8") as f:

                    json.dump(e, f, ensure_ascii=False, indent=2)

                st.success("Updated evidence file.")



            st.write("Generate PRM brief:")

            api_key_env = os.environ.get("OPENAI_API_KEY")

            use_llm = st.checkbox("Use OpenAI/ChatGPT to generate brief (requires OPENAI_API_KEY in environment)", value=bool(api_key_env))

            if st.button("Generate brief now"):

                if use_llm and api_key_env:

                    st.info("Calling OpenAI... (ensure API key present)")

                    res = generate_brief_with_openai(e, api_key_env)

                    st.code(res)

                else:

                    res = generate_brief_local(e)

                    st.code(res)



            if st.button("Export evidence JSON"):

                st.download_button("Download JSON", data=json.dumps(e, ensure_ascii=False, indent=2), file_name=f"{e['submission_id']}.json", mime="application/json")



st.markdown("---")

st.caption("SSISM-PRM MVP — built for VIP use. Keep the server secure. This app stores uploaded files locally under `evidence_store/`.")

