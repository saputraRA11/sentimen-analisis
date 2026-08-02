from pathlib import Path
import cgi
import io
import json
import logging
import os
import re
import socket
import tempfile
import time
import uuid
import zipfile
from logging.handlers import RotatingFileHandler
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

try:
    import joblib
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception:
    joblib = None
    load_model = None
    pad_sequences = None

st.set_page_config(page_title="Dashboard ABSA Livin", layout="wide", initial_sidebar_state="collapsed")

DEFAULT_DASHBOARD_FILE = Path("data_test_prediksi.csv")
DEFAULT_LABELED_FILE = Path("V4_LABELED_DATASET_FINAL.csv")
TEMPLATE_COLUMNS = ["Ulasan", "Aspek", "Sentimen"]
MODEL_DIR = Path("models")
MODEL_ASPEK_FILE = MODEL_DIR / "model_aspek_raw.keras"
MODEL_SENTIMEN_FILE = MODEL_DIR / "model_sentimen_cascaded.keras"
TOKENIZER_FILE = MODEL_DIR / "tokenizer_absa.joblib"
ENCODER_ASPEK_FILE = MODEL_DIR / "encoder_aspek.joblib"
ENCODER_SENTIMEN_FILE = MODEL_DIR / "encoder_sentimen.joblib"
MODEL_CONFIG_FILE = MODEL_DIR / "dashboard_model_config.joblib"
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "dashboard_model.log"
ALLOW_MODEL_UPLOADS = os.environ.get("ALLOW_MODEL_UPLOADS", "").lower() in {"1", "true", "yes"}
MODEL_UPLOADS = {
    "model_aspek": MODEL_ASPEK_FILE,
    "model_sentimen": MODEL_SENTIMEN_FILE,
    "tokenizer": TOKENIZER_FILE,
    "encoder_aspek": ENCODER_ASPEK_FILE,
    "encoder_sentimen": ENCODER_SENTIMEN_FILE,
}


def setup_dashboard_logger():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("dashboard_model")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=2_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


logger = setup_dashboard_logger()


def log_event(event, **fields):
    safe_fields = {
        key: value
        for key, value in fields.items()
        if value is not None
    }
    detail = " | ".join(f"{key}={value}" for key, value in safe_fields.items())
    logger.info("%s%s", event, f" | {detail}" if detail else "")


def log_trace(message, **fields):
    log_event(f"TRACE - {message}", **fields)


def elapsed_ms(start_time):
    return int((time.perf_counter() - start_time) * 1000)

ASPECT_MAP = {
    "availability": "reliability",
    "functionality": "functional suitability",
    "functionality suitability": "functional suitability",
    "interactivity capability": "interaction capability",
    "interactive capability": "interaction capability",
    "maintenance": "maintainability",
    "usability": "security",
    "performace efficiency": "performance efficiency",
    "2": "compatibility",
}
SENTIMENT_MAP = {
    -1: 2,
    "-1": 2,
    0: 0,
    1: 1,
    2: 2,
    "0": 0,
    "1": 1,
    "2": 2,
    "negatif": 0,
    "positif": 1,
    "netral": 2,
    "Negatif": 0,
    "Positif": 1,
    "Netral": 2,
}
SENTIMENT_LABEL = {0: "Negatif", 1: "Positif", 2: "Netral"}


def clean_aspect(value):
    text = str(value).strip().lower()
    return ASPECT_MAP.get(text, text)


def clean_sentiment(value):
    mapped = SENTIMENT_MAP.get(value, SENTIMENT_MAP.get(str(value).strip(), 2))
    try:
        mapped = int(mapped)
    except (TypeError, ValueError):
        mapped = 2
    return SENTIMENT_LABEL.get(mapped, "Netral")


def strip_keras_incompatible_config(value):
    if isinstance(value, dict):
        return {
            key: strip_keras_incompatible_config(item)
            for key, item in value.items()
            if key != "quantization_config"
        }
    if isinstance(value, list):
        return [strip_keras_incompatible_config(item) for item in value]
    return value


def load_keras_compatible(path):
    start_time = time.perf_counter()
    log_event("keras_load_start", file=path)
    try:
        model = load_model(path, compile=False)
        log_event("keras_load_done", file=path, duration_ms=elapsed_ms(start_time))
        return model
    except Exception as original_error:
        if not zipfile.is_zipfile(path):
            logger.exception("keras_load_failed | file=%s", path)
            raise original_error

        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            log_event("keras_load_retry_without_quantization_config", file=path)
            with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(tmp_path, "w") as target:
                for name in source.namelist():
                    data = source.read(name)
                    if name == "config.json":
                        config = strip_keras_incompatible_config(json.loads(data.decode("utf-8")))
                        data = json.dumps(config).encode("utf-8")
                    target.writestr(name, data)
            model = load_model(tmp_path, compile=False)
            log_event("keras_load_done_after_retry", file=path, duration_ms=elapsed_ms(start_time))
            return model
        except Exception:
            logger.exception("keras_load_retry_failed | file=%s", path)
            raise original_error
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass


_model_lock = threading.Lock()
_predict_lock = threading.Lock()
_model_artifacts = None
_model_status = {
    "state": "idle",
    "message": "Model belum dimuat.",
    "ready": False,
    "missing": [],
    "missing_runtime": [],
    "loaded_at": None,
}
_prediction_jobs = {}
_prediction_jobs_lock = threading.Lock()


def update_prediction_job(job_id, **updates):
    if not job_id:
        return
    with _prediction_jobs_lock:
        if job_id in _prediction_jobs:
            _prediction_jobs[job_id].update(updates)


def _set_model_status(**updates):
    with _model_lock:
        _model_status.update(updates)


def load_model_artifacts():
    global _model_artifacts
    start_time = time.perf_counter()

    with _model_lock:
        if _model_artifacts is not None:
            log_event("model_artifacts_cache_hit")
            return _model_artifacts
        if _model_status.get("state") == "loading":
            log_event("model_artifacts_already_loading")
            return {
                "ready": False,
                "loading": True,
                "message": "Model sedang disiapkan. Tunggu beberapa saat.",
            }
        _model_status.update({
            "state": "loading",
            "ready": False,
            "message": "Model sedang disiapkan.",
            "missing": [],
            "missing_runtime": [],
        })

    required = [MODEL_ASPEK_FILE, MODEL_SENTIMEN_FILE, TOKENIZER_FILE, ENCODER_ASPEK_FILE, ENCODER_SENTIMEN_FILE]
    missing = [path.name for path in required if not path.exists()]
    missing_runtime = []
    if joblib is None:
        missing_runtime.append("joblib")
    if load_model is None or pad_sequences is None:
        missing_runtime.append("tensorflow")
    if missing or missing_runtime:
        artifacts = {"ready": False, "missing": missing, "missing_runtime": missing_runtime}
        log_event(
            "model_artifacts_missing",
            missing=",".join(missing),
            missing_runtime=",".join(missing_runtime),
        )
        _set_model_status(
            state="error",
            ready=False,
            message="Artefak model atau dependency belum lengkap.",
            missing=missing,
            missing_runtime=missing_runtime,
        )
        return artifacts

    try:
        log_event("model_artifacts_load_start", max_len_config=MODEL_CONFIG_FILE.exists())
        config = joblib.load(MODEL_CONFIG_FILE) if MODEL_CONFIG_FILE.exists() else {"MAX_LEN": 50}
        artifacts = {
            "ready": True,
            "model_aspek": load_keras_compatible(MODEL_ASPEK_FILE),
            "model_sentimen": load_keras_compatible(MODEL_SENTIMEN_FILE),
            "tokenizer": joblib.load(TOKENIZER_FILE),
            "encoder_aspek": joblib.load(ENCODER_ASPEK_FILE),
            "encoder_sentimen": joblib.load(ENCODER_SENTIMEN_FILE),
            "max_len": int(config.get("MAX_LEN", 50)),
        }
        log_event(
            "model_artifacts_load_done",
            duration_ms=elapsed_ms(start_time),
            max_len=artifacts["max_len"],
            aspek_classes=len(getattr(artifacts["encoder_aspek"], "classes_", [])),
            sentimen_classes=len(getattr(artifacts["encoder_sentimen"], "classes_", [])),
            tokenizer_vocab=len(getattr(artifacts["tokenizer"], "word_index", {})),
        )
    except Exception as exc:
        artifacts = {"ready": False, "message": "Model gagal dimuat.", "detail": str(exc)}
        logger.exception("model_artifacts_load_failed")
        _set_model_status(
            state="error",
            ready=False,
            message="Model gagal dimuat.",
            detail=str(exc),
        )
        return artifacts

    with _model_lock:
        _model_artifacts = artifacts
        _model_status.update({
            "state": "ready",
            "ready": True,
            "message": "Model siap digunakan.",
            "missing": [],
            "missing_runtime": [],
            "detail": "",
            "loaded_at": time.time(),
        })
    return artifacts


def get_model_status():
    with _model_lock:
        status = dict(_model_status)
    return {
        "ok": True,
        "ready": bool(status.get("ready")),
        "state": status.get("state", "idle"),
        "message": status.get("message", "Model belum dimuat."),
        "missing": status.get("missing", []),
        "missing_runtime": status.get("missing_runtime", []),
    }


def reset_model_cache():
    global _model_artifacts
    with _model_lock:
        _model_artifacts = None
        _model_status.update({
            "state": "idle",
            "message": "Model belum dimuat.",
            "ready": False,
            "missing": [],
            "missing_runtime": [],
            "detail": "",
            "loaded_at": None,
        })


def preload_model_async(force=False):
    if force:
        reset_model_cache()
    with _model_lock:
        if _model_artifacts is not None or _model_status.get("state") == "loading":
            return
    threading.Thread(target=load_model_artifacts, daemon=True).start()


def predict_with_model(review_text, job_id=None):
    start_time = time.perf_counter()
    text = str(review_text or "").strip()
    log_event(
        "prediction_start",
        job_id=job_id,
        text_chars=len(text),
        text_words=len(text.split()),
    )
    log_trace("Prediksi dimulai", job_id=job_id, tahap="mulai", panjang_teks=len(text))
    update_prediction_job(job_id, stage="mulai", stage_message="Prediksi dimulai.", stage_started_at=time.time())

    artifacts = load_model_artifacts()
    if not artifacts.get("ready"):
        log_event(
            "prediction_blocked_model_not_ready",
            job_id=job_id,
            missing=",".join(artifacts.get("missing", [])),
            missing_runtime=",".join(artifacts.get("missing_runtime", [])),
        )
        return {
            "ok": False,
            "message": "Model belum siap dipakai. Jalankan cell penyimpanan artefak model, atau upload artefak model dari panel Unggah Model di sidebar.",
            "missing": artifacts.get("missing", []),
            "missing_runtime": artifacts.get("missing_runtime", []),
        }

    if not text:
        log_event("prediction_blocked_empty_text", job_id=job_id)
        return {"ok": False, "message": "Teks ulasan masih kosong."}

    try:
        update_prediction_job(job_id, stage="tokenisasi", stage_message="Mengubah teks menjadi sequence token.", stage_started_at=time.time())
        tokenize_start = time.perf_counter()
        log_trace("Tokenisasi teks dimulai", job_id=job_id, tahap="tokenisasi")
        seq = artifacts["tokenizer"].texts_to_sequences([text])
        x = pad_sequences(seq, maxlen=artifacts["max_len"], padding="post", truncating="post")
        non_zero_tokens = int(np.count_nonzero(x))
        log_event(
            "prediction_tokenized",
            job_id=job_id,
            duration_ms=elapsed_ms(tokenize_start),
            sequence_len=len(seq[0]) if seq else 0,
            non_zero_tokens=non_zero_tokens,
            input_shape=tuple(x.shape),
        )
        log_trace(
            "Tokenisasi teks selesai",
            job_id=job_id,
            tahap="tokenisasi",
            durasi_ms=elapsed_ms(tokenize_start),
            token_terisi=non_zero_tokens,
            bentuk_input=tuple(x.shape),
        )

        update_prediction_job(job_id, stage="prediksi_aspek", stage_message="Model aspek sedang menghitung probabilitas.", stage_started_at=time.time())
        aspek_start = time.perf_counter()
        log_event(
            "prediction_aspect_start",
            job_id=job_id,
            input_shape=tuple(x.shape),
            max_len=artifacts["max_len"],
        )
        log_trace("Prediksi aspek dimulai", job_id=job_id, tahap="prediksi_aspek", bentuk_input=tuple(x.shape))
        with _predict_lock:
            log_trace("Lock prediksi didapat untuk model aspek", job_id=job_id, tahap="prediksi_aspek")
            aspek_prob = artifacts["model_aspek"](x, training=False).numpy()
        log_event(
            "prediction_aspect_raw_output",
            job_id=job_id,
            output_shape=tuple(aspek_prob.shape),
            min_prob=round(float(np.min(aspek_prob)), 6),
            max_prob=round(float(np.max(aspek_prob)), 6),
        )
        aspek_idx = np.argmax(aspek_prob, axis=1)
        update_prediction_job(job_id, stage="decode_aspek", stage_message="Mengubah class index aspek menjadi label.", stage_started_at=time.time())
        aspek_label = artifacts["encoder_aspek"].inverse_transform(aspek_idx)[0]
        aspect_confidence = float(np.max(aspek_prob))
        log_event(
            "prediction_aspect_done",
            job_id=job_id,
            duration_ms=elapsed_ms(aspek_start),
            raw_label=aspek_label,
            clean_label=clean_aspect(aspek_label),
            class_index=int(aspek_idx[0]),
            confidence=round(aspect_confidence, 6),
            output_shape=tuple(aspek_prob.shape),
        )
        log_trace(
            "Prediksi aspek selesai",
            job_id=job_id,
            tahap="prediksi_aspek",
            durasi_ms=elapsed_ms(aspek_start),
            label=clean_aspect(aspek_label),
            confidence=round(aspect_confidence, 6),
        )

        update_prediction_job(job_id, stage="prediksi_sentimen", stage_message="Model sentimen sedang menghitung probabilitas.", stage_started_at=time.time())
        sentimen_start = time.perf_counter()
        log_event(
            "prediction_sentiment_start",
            job_id=job_id,
            text_input_shape=tuple(x.shape),
            aspect_input_shape=tuple(aspek_idx.shape),
            aspect_class_index=int(aspek_idx[0]),
        )
        log_trace(
            "Prediksi sentimen dimulai",
            job_id=job_id,
            tahap="prediksi_sentimen",
            aspek_terprediksi=clean_aspect(aspek_label),
            aspek_index=int(aspek_idx[0]),
        )
        with _predict_lock:
            log_trace("Lock prediksi didapat untuk model sentimen", job_id=job_id, tahap="prediksi_sentimen")
            sentimen_prob = artifacts["model_sentimen"]([x, aspek_idx], training=False).numpy()
        log_event(
            "prediction_sentiment_raw_output",
            job_id=job_id,
            output_shape=tuple(sentimen_prob.shape),
            min_prob=round(float(np.min(sentimen_prob)), 6),
            max_prob=round(float(np.max(sentimen_prob)), 6),
        )
        sentimen_idx = np.argmax(sentimen_prob, axis=1)
        update_prediction_job(job_id, stage="decode_sentimen", stage_message="Mengubah class index sentimen menjadi label.", stage_started_at=time.time())
        sentimen_raw = artifacts["encoder_sentimen"].inverse_transform(sentimen_idx)[0]
        sentiment_confidence = float(np.max(sentimen_prob))
        result = {
            "ok": True,
            "aspect": clean_aspect(aspek_label),
            "sentiment": clean_sentiment(sentimen_raw),
            "aspectConfidence": aspect_confidence,
            "sentimentConfidence": sentiment_confidence,
            "confidence": float((aspect_confidence + sentiment_confidence) / 2),
        }
        log_event(
            "prediction_sentiment_done",
            job_id=job_id,
            duration_ms=elapsed_ms(sentimen_start),
            raw_label=sentimen_raw,
            clean_label=result["sentiment"],
            class_index=int(sentimen_idx[0]),
            confidence=round(sentiment_confidence, 6),
            output_shape=tuple(sentimen_prob.shape),
        )
        log_trace(
            "Prediksi sentimen selesai",
            job_id=job_id,
            tahap="prediksi_sentimen",
            durasi_ms=elapsed_ms(sentimen_start),
            label=result["sentiment"],
            confidence=round(sentiment_confidence, 6),
        )
        log_event(
            "prediction_done",
            job_id=job_id,
            duration_ms=elapsed_ms(start_time),
            aspect=result["aspect"],
            sentiment=result["sentiment"],
            confidence=round(result["confidence"], 6),
        )
        update_prediction_job(job_id, stage="selesai", stage_message="Prediksi selesai.", stage_started_at=time.time())
        log_trace(
            "Prediksi selesai",
            job_id=job_id,
            tahap="selesai",
            total_durasi_ms=elapsed_ms(start_time),
            aspek=result["aspect"],
            sentimen=result["sentiment"],
            confidence=round(result["confidence"], 6),
        )
        return result
    except Exception:
        logger.exception("prediction_failed | job_id=%s", job_id)
        raise


def start_prediction_job(review_text, run_async=True):
    text = str(review_text or "").strip()
    if not text:
        log_event("prediction_job_rejected_empty_text")
        return {"ok": False, "message": "Teks ulasan masih kosong."}

    job_id = uuid.uuid4().hex
    log_event("prediction_job_created", job_id=job_id, text_chars=len(text), text_words=len(text.split()))
    with _prediction_jobs_lock:
        _prediction_jobs[job_id] = {
            "ok": True,
            "job_id": job_id,
            "state": "queued",
            "message": "Prediksi masuk antrean.",
            "stage": "queued",
            "stage_message": "Menunggu worker prediksi.",
            "stage_started_at": time.time(),
            "created_at": time.time(),
        }

    def monitor():
        last_stage = None
        last_reported_second = 0
        while True:
            time.sleep(10)
            with _prediction_jobs_lock:
                job = dict(_prediction_jobs.get(job_id, {}))
            if not job or job.get("state") not in {"queued", "running"}:
                return
            stage = job.get("stage", "-")
            elapsed_total = int(time.time() - job.get("created_at", time.time()))
            elapsed_stage = int(time.time() - job.get("stage_started_at", job.get("created_at", time.time())))
            should_report = stage != last_stage or elapsed_total - last_reported_second >= 30
            if should_report:
                log_trace(
                    "Prediksi masih berjalan",
                    job_id=job_id,
                    tahap=stage,
                    durasi_total_detik=elapsed_total,
                    durasi_tahap_detik=elapsed_stage,
                    pesan=job.get("stage_message"),
                )
                last_stage = stage
                last_reported_second = elapsed_total

    def worker():
        with _prediction_jobs_lock:
            _prediction_jobs[job_id].update({
                "state": "running",
                "message": "Model sedang memproses ulasan.",
                "stage": "worker_mulai",
                "stage_message": "Worker prediksi mulai berjalan.",
                "stage_started_at": time.time(),
            })
        log_event("prediction_job_running", job_id=job_id)
        log_trace("Worker prediksi mulai", job_id=job_id, tahap="worker_mulai")
        try:
            result = predict_with_model(text, job_id=job_id)
            with _prediction_jobs_lock:
                _prediction_jobs[job_id].update({
                    "state": "done" if result.get("ok") else "error",
                    "result": result,
                    "message": result.get("message", "Prediksi selesai."),
                    "finished_at": time.time(),
                })
            log_event("prediction_job_finished", job_id=job_id, state="done" if result.get("ok") else "error")
            log_trace("Worker prediksi selesai", job_id=job_id, status="done" if result.get("ok") else "error")
        except Exception as exc:
            logger.exception("prediction_job_failed | job_id=%s", job_id)
            with _prediction_jobs_lock:
                _prediction_jobs[job_id].update({
                    "state": "error",
                    "result": {"ok": False, "message": "Prediksi belum bisa diproses oleh server model."},
                    "message": "Prediksi belum bisa diproses oleh server model.",
                    "detail": str(exc),
                    "stage": "error",
                    "stage_message": "Prediksi gagal. Lihat traceback di log.",
                    "stage_started_at": time.time(),
                    "finished_at": time.time(),
                })

    if run_async:
        threading.Thread(target=monitor, daemon=True).start()
        threading.Thread(target=worker, daemon=True).start()
    return {"ok": True, "job_id": job_id, "state": "queued", "message": "Prediksi sedang diproses."}


def get_prediction_job(job_id):
    with _prediction_jobs_lock:
        job = dict(_prediction_jobs.get(job_id, {}))
    if not job:
        return {"ok": False, "state": "missing", "message": "Job prediksi tidak ditemukan."}
    now = time.time()
    job["running_seconds"] = int(now - job.get("created_at", now))
    job["stage_seconds"] = int(now - job.get("stage_started_at", job.get("created_at", now)))
    return job


@st.cache_data
def load_dataset():
    candidate_files = [DEFAULT_DASHBOARD_FILE, DEFAULT_LABELED_FILE]
    source_file = next((path for path in candidate_files if path.exists()), None)
    if source_file is None:
        return pd.DataFrame(columns=["Ulasan", "Aspek", "Sentimen"])

    raw = pd.read_csv(source_file)

    normalized_cols = {str(col).strip().lower(): col for col in raw.columns}
    if {"ulasan", "aspek", "sentimen"}.issubset(normalized_cols):
        df = raw.copy()
        df = df.rename(columns={
            normalized_cols["ulasan"]: "Ulasan",
            normalized_cols["aspek"]: "Aspek",
            normalized_cols["sentimen"]: "Sentimen",
        })
    elif {"Ulasan", "Aspek", "Sentimen"}.issubset(raw.columns):
        df = raw.copy()
    elif {"final_text", "aspek_llm", "sentimen_llm"}.issubset(raw.columns):
        df = pd.DataFrame({
            "Ulasan": raw["final_text"],
            "Aspek": raw["aspek_llm"],
            "Sentimen": raw["sentimen_llm"],
        })
    else:
        st.error("Format CSV tidak sesuai. Gunakan kolom ulasan, aspek, sentimen.")
        st.stop()

    df["Ulasan"] = df["Ulasan"].astype(str)
    df["Aspek"] = df["Aspek"].map(clean_aspect)
    df["Sentimen"] = df["Sentimen"].map(clean_sentiment)
    for col in ["True_Aspek", "True_Sentimen"]:
        if col in df.columns:
            if col == "True_Aspek":
                df[col] = df[col].map(clean_aspect)
            else:
                df[col] = df[col].map(clean_sentiment)
    for col in [
        "Confidence_Aspek",
        "Confidence_Sentimen",
        "Prediksi_Benar_Aspek",
        "Prediksi_Benar_Sentimen",
        "Prediksi_Benar_End_to_End",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Ulasan"])


def compute_payload(df):
    aspect_counts = df["Aspek"].value_counts().reset_index()
    aspect_counts.columns = ["aspek", "jumlah"]
    sent_counts = df["Sentimen"].value_counts().reindex(["Positif", "Netral", "Negatif"]).fillna(0).astype(int)

    importance = df.groupby("Aspek").size().reset_index(name="importance")
    positive = df[df["Sentimen"] == "Positif"].groupby("Aspek").size().reset_index(name="positive")
    negative = df[df["Sentimen"] == "Negatif"].groupby("Aspek").size().reset_index(name="negative")
    ipa = importance.merge(positive, on="Aspek", how="left").merge(negative, on="Aspek", how="left")
    ipa[["positive", "negative"]] = ipa[["positive", "negative"]].fillna(0)
    top3_mean = float(ipa["importance"].nlargest(min(3, len(ipa))).mean()) if len(ipa) else 0.0
    top3_mean = top3_mean if top3_mean else 1.0
    ipa["negative_rate"] = ipa["negative"] / ipa["importance"]
    ipa["performance"] = ipa["positive"] / ipa["importance"]
    ipa["importance_score_raw"] = ipa["importance"] / top3_mean
    ipa["performance_score_raw"] = ipa["performance"]

    performance_mean = float(ipa["performance_score_raw"].mean()) if len(ipa) else 0.0
    importance_mean = float(ipa["importance_score_raw"].mean()) if len(ipa) else 0.0
    performance_std = float(ipa["performance_score_raw"].std(ddof=0)) if len(ipa) else 0.0
    importance_std = float(ipa["importance_score_raw"].std(ddof=0)) if len(ipa) else 0.0
    ipa["performance_score"] = (ipa["performance_score_raw"] - performance_mean) / performance_std if performance_std else 0.0
    ipa["importance_score"] = (ipa["importance_score_raw"] - importance_mean) / importance_std if importance_std else 0.0
    x_mid = 0.0
    y_mid = 0.0

    def quadrant(row):
        if row["importance_score"] >= 0 and row["performance_score"] < 0:
            return "A"
        if row["importance_score"] >= 0 and row["performance_score"] >= 0:
            return "B"
        if row["importance_score"] < 0 and row["performance_score"] < 0:
            return "C"
        return "D"

    ipa["quadrant"] = ipa.apply(quadrant, axis=1)
    ipa = ipa.sort_values(["quadrant", "importance"], ascending=[True, False]).reset_index(drop=True)
    ipa["rank"] = ipa.index + 1

    cross = pd.crosstab(df["Aspek"], df["Sentimen"]).reset_index()
    for col in ["Positif", "Netral", "Negatif"]:
        if col not in cross.columns:
            cross[col] = 0

    total = int(len(df))
    pos = int(sent_counts.get("Positif", 0))
    neg = int(sent_counts.get("Negatif", 0))
    top_aspect = aspect_counts.iloc[0].to_dict() if len(aspect_counts) else {"aspek": "-", "jumlah": 0}
    priority = ipa[ipa["quadrant"] == "A"]

    reviews = df.to_dict(orient="records")
    template_cols = [
        "Ulasan",
        "Aspek",
        "Sentimen",
        "True_Aspek",
        "True_Sentimen",
        "Confidence_Aspek",
        "Confidence_Sentimen",
        "Prediksi_Benar_Aspek",
        "Prediksi_Benar_Sentimen",
        "Prediksi_Benar_End_to_End",
        "Sumber_Data",
    ]
    template = ",".join(template_cols) + "\n"

    return {
        "total": total,
        "positiveRate": pos / total if total else 0,
        "negativeRate": neg / total if total else 0,
        "topAspect": top_aspect,
        "priorityCount": int(len(priority)),
        "aspectCounts": aspect_counts.to_dict(orient="records"),
        "sentimentCounts": [{"sentimen": k, "jumlah": int(v)} for k, v in sent_counts.items()],
        "cross": cross[["Aspek", "Positif", "Netral", "Negatif"]].to_dict(orient="records"),
        "ipa": ipa.to_dict(orient="records"),
        "xMid": x_mid,
        "yMid": y_mid,
        "reviews": reviews,
        "templateCsv": template,
        "sourceData": str(DEFAULT_DASHBOARD_FILE if DEFAULT_DASHBOARD_FILE.exists() else DEFAULT_LABELED_FILE),
    }


def build_html(payload):
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""
<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600&family=Work+Sans:wght@600;700&display=swap" rel="stylesheet" />
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=block" rel="stylesheet" />
<style>
:root {{
  --inverse-on-surface:#eaf1ff; --on-primary-fixed:#001b3c; --error-container:#ffdad6;
  --surface-dim:#cbdbf5; --secondary-fixed:#ffdea7; --on-background:#0b1c30;
  --surface-tint:#335f9c; --on-primary:#ffffff; --error:#ba1a1a;
  --on-surface-variant:#434750; --primary:#002752; --primary-container:#003d79;
  --background:#f8f9ff; --surface:#f8f9ff; --surface-container:#e5eeff;
  --surface-container-lowest:#ffffff; --surface-container-low:#eff4ff; --surface-container-high:#dce9ff;
  --surface-variant:#d3e4fe; --outline-variant:#c3c6d1; --outline:#737781;
  --on-surface:#0b1c30; --secondary-container:#fcb812; --success:#16a34a;
  --grid-gutter:20px; --container-padding:32px; --sidebar-width:260px;
}}
* {{ box-sizing:border-box; }}
html, body {{ margin:0; min-height:100%; background:var(--background); color:var(--on-surface); font-family:'Public Sans', sans-serif; font-size:14px; line-height:20px; }}
body {{ overflow:hidden; }}
.app {{ min-height:100vh; display:flex; }}
.sidebar {{ width:var(--sidebar-width); height:100vh; position:fixed; left:0; top:0; background:var(--surface); border-right:1px solid var(--outline-variant); display:flex; flex-direction:column; padding:12px 16px; z-index:20; overflow-y:auto; overscroll-behavior:contain; }}
.brand {{ display:flex; align-items:center; gap:12px; padding:8px 8px 30px; }}
.brand h1 {{ margin:0; font-family:'Work Sans'; font-size:20px; line-height:28px; font-weight:700; color:var(--primary); }}
.brand p {{ margin:0; font-size:12px; line-height:16px; letter-spacing:.05em; font-weight:600; color:var(--on-surface-variant); }}
.nav {{ flex:1; display:flex; flex-direction:column; gap:4px; }}
.nav button {{ appearance:none; border:0; background:transparent; width:100%; display:flex; align-items:center; gap:12px; padding:10px 12px; border-radius:8px; color:var(--on-surface-variant); font:600 12px/16px 'Public Sans'; letter-spacing:.05em; cursor:pointer; text-align:left; transition:.16s ease; }}
.nav button:hover {{ background:var(--surface-container); }}
.nav button.active {{ color:var(--primary); background:var(--surface-container-low); border-right:4px solid var(--primary); font-weight:700; }}
.nav .material-symbols-outlined {{ font-size:20px; }}
.side-actions {{ margin-top:28px; display:flex; flex-direction:column; gap:10px; padding-bottom:120px; flex-shrink:0; }}
.primary-btn {{ border:0; border-radius:8px; background:var(--primary); color:var(--on-primary); padding:11px 14px; font:700 12px/16px 'Public Sans'; letter-spacing:.04em; cursor:pointer; display:flex; align-items:center; justify-content:center; gap:8px; }}
.secondary-link {{ border:0; background:transparent; color:var(--on-surface-variant); display:flex; align-items:center; gap:10px; padding:9px 12px; border-radius:8px; font:600 12px/16px 'Public Sans'; cursor:pointer; }}
.secondary-link:hover {{ background:var(--surface-container); }}
.format-note {{ color:var(--on-surface-variant); font-size:10px; line-height:14px; padding:0 4px 10px; }}
.main {{ margin-left:var(--sidebar-width); width:calc(100% - var(--sidebar-width)); height:100vh; overflow:auto; }}
.topbar {{ position:sticky; top:0; z-index:10; height:60px; background:var(--surface); border-bottom:1px solid var(--outline-variant); display:flex; align-items:center; justify-content:space-between; padding:12px var(--container-padding); }}
.topbar h2 {{ margin:0; font-family:'Work Sans'; font-size:20px; line-height:28px; color:var(--primary); font-weight:700; }}
.canvas {{ max-width:1280px; margin:0 auto; padding:var(--container-padding); display:flex; flex-direction:column; gap:24px; }}
.page-head {{ display:flex; justify-content:space-between; align-items:end; gap:20px; }}
.page-head h1 {{ margin:0 0 8px; font-family:'Work Sans'; font-size:36px; line-height:44px; color:var(--primary); font-weight:700; letter-spacing:-.02em; }}
.page-head p {{ margin:0; color:var(--on-surface-variant); font-size:14px; }}
.kpi-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:var(--grid-gutter); }}
.kpi-grid.prediction {{ grid-template-columns:repeat(2,minmax(0,1fr)); max-width:680px; }}
.card {{ background:var(--surface-container-lowest); border:1px solid #e2e8f0; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,.02); }}
.kpi {{ padding:20px; min-height:128px; display:flex; flex-direction:column; justify-content:space-between; }}
.kpi-label {{ color:var(--outline); text-transform:uppercase; letter-spacing:.05em; font:600 12px/16px 'Public Sans'; display:flex; align-items:center; justify-content:space-between; gap:8px; }}
.kpi-value {{ margin-top:14px; font-family:'Work Sans'; font-size:36px; line-height:44px; color:var(--primary); font-weight:700; word-break:break-word; }}
.kpi-note {{ color:var(--on-surface-variant); font-size:12px; margin-top:4px; }}
.grid-12 {{ display:grid; grid-template-columns:repeat(12,minmax(0,1fr)); gap:var(--grid-gutter); }}
.col-4 {{ grid-column:span 4; }} .col-5 {{ grid-column:span 5; }} .col-7 {{ grid-column:span 7; }} .col-8 {{ grid-column:span 8; }} .col-12 {{ grid-column:span 12; }}
.panel {{ padding:28px; }}
.panel-title {{ margin:0 0 20px; color:var(--outline); text-transform:uppercase; letter-spacing:.08em; font:600 12px/16px 'Public Sans'; display:flex; justify-content:space-between; align-items:center; }}
.headline {{ margin:0 0 6px; font-family:'Work Sans'; font-size:28px; line-height:36px; color:var(--on-surface); font-weight:600; }}
.body-muted {{ color:var(--on-surface-variant); }}
textarea {{ width:100%; min-height:260px; border:1px solid var(--outline-variant); border-radius:8px; background:var(--surface); color:var(--on-surface); padding:16px; resize:vertical; font:400 14px/20px 'Public Sans'; outline:none; }}
textarea:focus {{ border-color:var(--primary); box-shadow:0 0 0 3px rgba(51,95,156,.12); }}
.result-card {{ padding:18px; border:1px solid var(--outline-variant); border-radius:8px; background:var(--surface); }}
.prediction-result-card {{ border:1px solid var(--outline-variant); border-radius:8px; background:var(--surface); padding:18px 20px; display:flex; align-items:center; justify-content:space-between; gap:18px; }}
.prediction-result-title {{ color:var(--on-surface); font:700 18px/24px 'Public Sans'; }}
.prediction-result-confidence {{ color:var(--outline); font:700 14px/20px 'Public Sans'; letter-spacing:.04em; }}
.sentiment-pill {{ border-radius:999px; padding:10px 18px; font:700 16px/20px 'Public Sans'; white-space:nowrap; }}
.sentiment-pill.positif {{ background:#e8f6ec; color:#167a38; }}
.sentiment-pill.negatif {{ background:#ffdad6; color:#93000a; }}
.sentiment-pill.netral {{ background:#e5eeff; color:#003d79; }}
.result-label {{ color:var(--on-surface-variant); text-transform:uppercase; letter-spacing:.05em; font:600 12px/16px 'Public Sans'; }}
.result-value {{ margin-top:8px; color:var(--primary); font-family:'Work Sans'; font-size:22px; line-height:30px; font-weight:700; }}
.notice {{ padding:16px; border-radius:8px; background:#fff8e8; border:1px solid #f1d48b; color:#5e4200; }}
.bar-list {{ display:flex; flex-direction:column; gap:12px; }}
.bar-row {{ display:grid; grid-template-columns:190px 1fr 52px; gap:12px; align-items:center; }}
.bar-label {{ color:var(--on-surface-variant); font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.bar-track {{ background:var(--surface-container); border-radius:999px; height:10px; overflow:hidden; }}
.bar-fill {{ background:var(--primary-container); height:100%; border-radius:999px; }}
.donut-wrap {{ display:flex; align-items:center; justify-content:center; min-height:260px; }}
.donut {{ width:210px; height:210px; border-radius:50%; display:grid; place-items:center; background:conic-gradient(#16a34a 0 var(--pos), #335f9c var(--pos) var(--net), #ba1a1a var(--net) 100%); position:relative; }}
.donut::after {{ content:''; width:126px; height:126px; border-radius:50%; background:white; position:absolute; }}
.donut-center {{ position:relative; z-index:1; text-align:center; }}
.donut-center b {{ display:block; font-family:'Work Sans'; font-size:24px; color:var(--primary); }}
.legend {{ display:flex; flex-wrap:wrap; justify-content:center; gap:14px; }}
.legend-item {{ display:flex; gap:8px; align-items:center; color:var(--on-surface-variant); font-size:12px; font-weight:600; }}
.swatch {{ width:12px; height:12px; border-radius:3px; }}
.stack-chart {{ display:flex; flex-direction:column; gap:12px; }}
.stack-row {{ display:grid; grid-template-columns:180px 1fr; gap:12px; align-items:center; }}
.stack-bar {{ height:22px; border-radius:4px; overflow:hidden; display:flex; background:var(--surface-container); }}
.seg-pos {{ background:#16a34a; }} .seg-net {{ background:#335f9c; }} .seg-neg {{ background:#ba1a1a; }}
.ipa-area {{ height:560px; background:var(--surface); border:1px solid var(--outline-variant); border-radius:4px; position:relative; overflow:hidden; }}
.quad {{ position:absolute; width:50%; height:50%; }} .qa {{ left:0; top:0; background:rgba(186,26,26,.07); }} .qb {{ right:0; top:0; background:rgba(0,39,82,.07); }} .qc {{ left:0; bottom:0; background:rgba(115,119,129,.08); }} .qd {{ right:0; bottom:0; background:rgba(252,184,18,.12); }}
.mid-v {{ position:absolute; top:0; bottom:0; left:50%; border-left:1px dashed var(--outline); }} .mid-h {{ position:absolute; left:0; right:0; top:50%; border-top:1px dashed var(--outline); }}
.quad-label {{ position:absolute; z-index:2; font:700 12px/16px 'Public Sans'; letter-spacing:.03em; }}
.point {{ position:absolute; z-index:4; width:16px; height:16px; border-radius:50%; background:var(--primary); border:3px solid white; box-shadow:0 4px 10px rgba(0,0,0,.18); transform:translate(-50%, 50%); cursor:pointer; appearance:none; padding:0; }}
.point.a {{ background:var(--error); }} .point.b {{ background:var(--primary); }} .point.c {{ background:var(--outline); }} .point.d {{ background:var(--secondary-container); }}
.point:hover, .point.selected {{ box-shadow:0 0 0 6px rgba(51,95,156,.16), 0 6px 16px rgba(0,0,0,.22); z-index:6; }}
.point.selected span {{ border-color:var(--primary); color:var(--primary); }}
.point span {{ position:absolute; left:14px; bottom:10px; white-space:nowrap; background:white; border:1px solid var(--outline-variant); border-radius:4px; padding:2px 6px; color:var(--on-surface); font:600 11px/14px 'Public Sans'; cursor:pointer; }}
.priority-card {{ border:1px solid var(--error-container); background:rgba(255,218,214,.25); border-radius:8px; padding:16px; margin-bottom:12px; }}
.priority-card h4 {{ margin:0 0 8px; color:var(--on-surface); font:700 13px/18px 'Public Sans'; }}
.ipa-detail-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; margin:14px 0; }}
.ipa-detail-metric {{ border:1px solid var(--outline-variant); border-radius:8px; background:var(--surface); padding:12px; }}
.ipa-detail-metric span {{ display:block; color:var(--outline); font:700 10px/14px 'Public Sans'; text-transform:uppercase; letter-spacing:.05em; }}
.ipa-detail-metric b {{ display:block; margin-top:4px; color:var(--primary); font:800 18px/24px 'Public Sans'; }}
.ipa-hint {{ color:var(--outline); font:700 11px/16px 'Public Sans'; margin:10px 0 0; }}
.table tr.clickable-row {{ cursor:pointer; }}
.table tr.clickable-row:hover {{ background:var(--surface-container-low); }}
.table {{ width:100%; border-collapse:collapse; font-size:13px; }} .table th {{ text-align:left; color:var(--outline); font:700 12px/16px 'Public Sans'; text-transform:uppercase; letter-spacing:.05em; border-bottom:1px solid var(--outline-variant); padding:10px; }} .table td {{ border-bottom:1px solid #e2e8f0; padding:10px; color:var(--on-surface); vertical-align:top; }}
.filters {{ display:grid; grid-template-columns:1fr 1fr 2fr; gap:16px; }}
select, input[type=text] {{ width:100%; border:1px solid var(--outline-variant); border-radius:8px; background:white; color:var(--on-surface); padding:11px 12px; font:400 14px/20px 'Public Sans'; }}
.hidden {{ display:none !important; }}
.model-upload-box {{ margin-top:0; padding-top:0; padding-bottom:18px; border-top:0; border-bottom:1px solid var(--outline-variant); }}
.model-upload-box h4 {{ margin:0 0 4px; color:var(--primary); font:800 13px/18px 'Public Sans'; }}
.model-upload-box .format-note {{ margin-bottom:10px; }}
.model-file-list {{ display:grid; gap:6px; margin:8px 0 10px; }}
.model-file-row {{ display:grid; grid-template-columns:1fr auto; align-items:center; gap:8px; padding:7px 8px; border:1px solid var(--outline-variant); border-radius:8px; background:var(--surface); }}
.model-file-meta {{ min-width:0; }}
.model-file-label {{ color:var(--on-surface); font:800 11px/15px 'Public Sans'; }}
.model-file-name {{ margin-top:1px; color:var(--outline); font:600 10px/14px 'Public Sans'; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:132px; }}
.model-file-row input {{ display:none; }}
.model-file-pick {{ width:30px; height:30px; border:1px solid var(--outline-variant); border-radius:8px; background:var(--secondary-container); color:var(--primary); display:flex; align-items:center; justify-content:center; cursor:pointer; }}
.model-file-pick .material-symbols-outlined {{ font-size:18px; }}
.upload-status {{ min-height:18px; margin-bottom:8px; color:var(--primary); font:700 11px/16px 'Public Sans'; }}
.sidebar-tail-space {{ height:96px; flex:0 0 96px; }}
.pagination {{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top:16px; padding-top:14px; border-top:1px solid var(--outline-variant); }}
.page-size {{ display:flex; align-items:center; gap:8px; color:var(--outline); font:700 12px/16px 'Public Sans'; }}
.page-size select {{ width:auto; min-width:76px; height:36px; padding:0 8px; }}
.page-controls {{ display:flex; align-items:center; gap:8px; }}
.page-btn {{ width:36px; height:36px; border-radius:8px; border:1px solid var(--outline-variant); background:var(--surface); color:var(--primary); display:flex; align-items:center; justify-content:center; cursor:pointer; }}
.page-btn:disabled {{ opacity:.42; cursor:not-allowed; }}
.page-info {{ min-width:120px; text-align:center; color:var(--on-surface-variant); font:700 12px/16px 'Public Sans'; }}
@media (max-width: 980px) {{ body {{ overflow:auto; }} .sidebar {{ position:relative; width:100%; height:auto; }} .main {{ margin-left:0; width:100%; }} .app {{ display:block; }} .kpi-grid {{ grid-template-columns:repeat(2,1fr); }} .col-4,.col-5,.col-7,.col-8,.col-12 {{ grid-column:span 12; }} }}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="brand"><div><h1>ABSA Livin'</h1><p>Analisis Mandiri</p></div></div>
    <nav class="nav">
      <button class="nav-btn active" data-page="prediksi"><span class="material-symbols-outlined">online_prediction</span>Prediksi Ulasan</button>
      <button class="nav-btn" data-page="overview"><span class="material-symbols-outlined">dashboard</span>Ikhtisar Data</button>
      <button class="nav-btn" data-page="ipa"><span class="material-symbols-outlined">grid_view</span>Matriks IPA</button>
      <button class="nav-btn" data-page="detail"><span class="material-symbols-outlined">table_view</span>Daftar Ulasan</button>
    </nav>
    <div class="side-actions">
      <div class="model-upload-box">
        <h4>Unggah Model</h4>
        <p class="format-note">Opsional. Jika kosong, dashboard memakai artefak lokal bawaan.</p>
        <div class="model-file-list">
          <div class="model-file-row"><div class="model-file-meta"><div class="model-file-label">Model aspek</div><div class="model-file-name" id="modelAspekName">Belum dipilih</div></div><label class="model-file-pick"><span class="material-symbols-outlined">attach_file</span><input id="modelAspekFile" type="file" accept=".keras,.h5"></label></div>
          <div class="model-file-row"><div class="model-file-meta"><div class="model-file-label">Model sentimen</div><div class="model-file-name" id="modelSentimenName">Belum dipilih</div></div><label class="model-file-pick"><span class="material-symbols-outlined">attach_file</span><input id="modelSentimenFile" type="file" accept=".keras,.h5"></label></div>
          <div class="model-file-row"><div class="model-file-meta"><div class="model-file-label">Tokenizer</div><div class="model-file-name" id="tokenizerName">Belum dipilih</div></div><label class="model-file-pick"><span class="material-symbols-outlined">attach_file</span><input id="tokenizerFile" type="file" accept=".joblib,.pkl"></label></div>
          <div class="model-file-row"><div class="model-file-meta"><div class="model-file-label">Encoder aspek</div><div class="model-file-name" id="encoderAspekName">Belum dipilih</div></div><label class="model-file-pick"><span class="material-symbols-outlined">attach_file</span><input id="encoderAspekFile" type="file" accept=".joblib,.pkl"></label></div>
          <div class="model-file-row"><div class="model-file-meta"><div class="model-file-label">Encoder sentimen</div><div class="model-file-name" id="encoderSentimenName">Belum dipilih</div></div><label class="model-file-pick"><span class="material-symbols-outlined">attach_file</span><input id="encoderSentimenFile" type="file" accept=".joblib,.pkl"></label></div>
        </div>
        <button id="uploadModelBtn" class="secondary-link" type="button"><span class="material-symbols-outlined">model_training</span>Upload Model</button>
        <p id="modelUploadStatus" class="upload-status"></p>
      </div>

      <input id="csvInput" type="file" accept=".csv" hidden />
      <button class="primary-btn" onclick="document.getElementById('csvInput').click()"><span class="material-symbols-outlined">upload_file</span>Unggah CSV</button>
      <p class="format-note">Unggah file CSV dengan format: ulasan, aspek, sentimen</p>
      <button class="secondary-link" id="downloadTemplate"><span class="material-symbols-outlined">download</span>Unduh Templat</button>
      <button class="secondary-link" onclick="showPage('detail')"><span class="material-symbols-outlined">info</span>Format Data</button>
      <div class="sidebar-tail-space" aria-hidden="true"></div>
    </div>
  </aside>
  <main class="main">
    <header class="topbar"><h2>Dashboard Evaluasi Kualitas Aplikasi Livin' by Mandiri</h2></header>
    <div class="canvas">
      <section id="page-prediksi" class="page"></section>
      <section id="page-overview" class="page hidden"></section>
      <section id="page-ipa" class="page hidden"></section>
      <section id="page-detail" class="page hidden"></section>
    </div>
  </main>
</div>
<script>
const DATA = {data_json};
let currentData = JSON.parse(JSON.stringify(DATA));
let reviewPage = 1;
let reviewPageSize = 25;
let selectedIpaAspect = null;
const rules = {{
  safety:['saldo','kepotong','uang','hilang','rekening','transaksi gagal','refund'],
  security:['login','otp','pin','password','blokir','verifikasi','aktivasi','akun'],
  reliability:['error','gangguan','down','maintenance','tidak bisa','gagal','server'],
  'performance efficiency':['lambat','lemot','loading','lama','berat','crash'],
  'functional suitability':['transfer','qris','bayar','top up','pembayaran','fitur','transaksi'],
  'interaction capability':['mudah','ribet','tampilan','simple','simpel','menu','user interface'],
  compatibility:['android','ios','hp','device','update','versi'], maintainability:['bug','perbaiki','update','versi baru','maintenance'], flexibility:['atur','custom','fleksibel','pilihan']
}};
const negWords = ['gagal','error','tidak','ga','gak','lemot','lambat','susah','ribet','hilang','kepotong','buruk','kecewa','parah'];
const posWords = ['bagus','mudah','cepat','mantap','baik','simpel','puas','membantu','lancar','keren'];
const fmt = new Intl.NumberFormat('id-ID');
const sentimentMap = {{'-1':'Netral','0':'Negatif','1':'Positif','2':'Netral','negatif':'Negatif','positif':'Positif','netral':'Netral'}};
function displayAspect(value) {{ return value || '-'; }}
function cleanAspect(value) {{ const raw=String(value||'').trim().toLowerCase(); const map={{'availability':'reliability','functionality':'functional suitability','functionality suitability':'functional suitability','interactivity capability':'interaction capability','interactive capability':'interaction capability','maintenance':'maintainability','usability':'security','performace efficiency':'performance efficiency','2':'compatibility'}}; return map[raw] || raw; }}
function cleanSentiment(value) {{ const raw=String(value ?? '').trim(); return sentimentMap[raw] || sentimentMap[raw.toLowerCase()] || 'Netral'; }}
function pct(v) {{ return `${{(v*100).toFixed(1)}}%`; }}
function kpi(label, value, note, icon='') {{ return `<div class="card kpi"><div class="kpi-label"><span>${{label}}</span><span class="material-symbols-outlined">${{icon}}</span></div><div><div class="kpi-value">${{value}}</div><div class="kpi-note">${{note}}</div></div></div>`; }}
function pageHead(title, subtitle) {{ return `<div class="page-head"><div><h1>${{title}}</h1><p>${{subtitle}}</p></div></div>`; }}
function globalKpis() {{ return `<div class="kpi-grid">${{kpi('Total Ulasan', fmt.format(currentData.total), 'Data dianalisis','forum')}}${{kpi('Sentimen Positif', pct(currentData.positiveRate), 'Proporsi ulasan positif','sentiment_satisfied')}}${{kpi('Sentimen Negatif', pct(currentData.negativeRate), 'Proporsi keluhan pengguna','sentiment_dissatisfied')}}${{kpi('Aspek Teratas', displayAspect(currentData.topAspect.aspek), fmt.format(currentData.topAspect.jumlah)+' sebutan','star')}}</div>`; }}
function predictionKpis() {{ return `<div class="kpi-grid prediction">${{kpi('Total Ulasan', fmt.format(currentData.total), 'Data dianalisis','')}}${{kpi('Aspek Dominan', displayAspect(currentData.topAspect.aspek), fmt.format(currentData.topAspect.jumlah)+' sebutan','')}}</div>`; }}
function countBy(rows, key) {{ const out=new Map(); rows.forEach(r=>out.set(r[key], (out.get(r[key])||0)+1)); return [...out.entries()].map(([name,jumlah])=>({{[key==='Aspek'?'aspek':'sentimen']:name, jumlah}})).sort((a,b)=>b.jumlah-a.jumlah); }}
function computeDashboardData(reviews) {{
  const cleanReviews=reviews.map(r=>({{...r, Ulasan:String(r.Ulasan||''), Aspek:cleanAspect(r.Aspek), Sentimen:cleanSentiment(r.Sentimen)}})).filter(r=>r.Ulasan && r.Aspek);
  const total=cleanReviews.length;
  const aspectCounts=countBy(cleanReviews,'Aspek');
  const sentimentRaw=countBy(cleanReviews,'Sentimen');
  const sentValue=name => (sentimentRaw.find(r=>r.sentimen===name)||{{jumlah:0}}).jumlah;
  const sentimentCounts=['Positif','Netral','Negatif'].map(sentimen=>({{sentimen, jumlah:sentValue(sentimen)}}));
  const aspects=[...new Set(cleanReviews.map(r=>r.Aspek))].sort();
  const cross=aspects.map(a=>{{ const rows=cleanReviews.filter(r=>r.Aspek===a); return {{Aspek:a, Positif:rows.filter(r=>r.Sentimen==='Positif').length, Netral:rows.filter(r=>r.Sentimen==='Netral').length, Negatif:rows.filter(r=>r.Sentimen==='Negatif').length}}; }});
  const top3=aspectCounts.slice(0,3).map(r=>r.jumlah);
  const top3Mean=top3.length?top3.reduce((s,v)=>s+v,0)/top3.length:1;
  let ipa=aspects.map(a=>{{ const rows=cleanReviews.filter(r=>r.Aspek===a); const importance=rows.length; const positive=rows.filter(r=>r.Sentimen==='Positif').length; const negative=rows.filter(r=>r.Sentimen==='Negatif').length; const negativeRate=importance?negative/importance:0; const performance=importance?positive/importance:0; return {{Aspek:a, importance, positive, negative, performance, negative_rate:negativeRate, importance_score_raw:importance/(top3Mean||1), performance_score_raw:performance}}; }});
  const mean=key => ipa.length?ipa.reduce((s,r)=>s+r[key],0)/ipa.length:0;
  const std=(key, avg) => ipa.length?Math.sqrt(ipa.reduce((s,r)=>s+Math.pow(r[key]-avg,2),0)/ipa.length):0;
  const performanceMean=mean('performance_score_raw');
  const importanceMean=mean('importance_score_raw');
  const performanceStd=std('performance_score_raw', performanceMean);
  const importanceStd=std('importance_score_raw', importanceMean);
  ipa=ipa.map(r=>{{ const performance_score=performanceStd?(r.performance_score_raw-performanceMean)/performanceStd:0; const importance_score=importanceStd?(r.importance_score_raw-importanceMean)/importanceStd:0; let quadrant='D'; if(importance_score>=0 && performance_score<0) quadrant='A'; else if(importance_score>=0 && performance_score>=0) quadrant='B'; else if(importance_score<0 && performance_score<0) quadrant='C'; return {{...r, performance_score, importance_score, quadrant}}; }}).sort((a,b)=>a.quadrant.localeCompare(b.quadrant)||b.importance-a.importance).map((r,i)=>({{...r, rank:i+1}}));
  const xMid=0;
  const yMid=0;
  const pos=sentValue('Positif'); const neg=sentValue('Negatif');
  return {{total, positiveRate:total?pos/total:0, negativeRate:total?neg/total:0, topAspect:aspectCounts[0]||{{aspek:'-',jumlah:0}}, priorityCount:ipa.filter(r=>r.quadrant==='A').length, aspectCounts, sentimentCounts, cross, ipa, xMid, yMid, reviews:cleanReviews, templateCsv:currentData.templateCsv}};
}}
function resultSummary(aspect, sentiment, confidence, note='') {{ const cls = sentiment.toLowerCase(); return `<div class="prediction-result-card"><div><div class="prediction-result-title">${{displayAspect(aspect)}}</div><div class="prediction-result-confidence">Confidence: ${{Math.round(confidence*100)}}%${{note?'<br>'+note:''}}</div></div><div class="sentiment-pill ${{cls}}">${{sentiment}}</div></div>`; }}
function bars(rows) {{ const max=Math.max(...rows.map(r=>r.jumlah),1); return `<div class="bar-list">${{rows.map(r=>`<div class="bar-row"><div class="bar-label">${{r.aspek}}</div><div class="bar-track"><div class="bar-fill" style="width:${{r.jumlah/max*100}}%"></div></div><div>${{fmt.format(r.jumlah)}}</div></div>`).join('')}}</div>`; }}
function donut() {{ const pos=currentData.positiveRate*100; const neg=currentData.negativeRate*100; const net=100-neg; return `<div class="donut-wrap"><div class="donut" style="--pos:${{pos}}%;--net:${{net}}%"><div class="donut-center"><b>${{fmt.format(currentData.total)}}</b><span>Total</span></div></div></div><div class="legend"><span class="legend-item"><i class="swatch" style="background:#16a34a"></i>Positif</span><span class="legend-item"><i class="swatch" style="background:#335f9c"></i>Netral</span><span class="legend-item"><i class="swatch" style="background:#ba1a1a"></i>Negatif</span></div>`; }}
function stackBars() {{ return `<div class="stack-chart">${{currentData.cross.map(r=>{{ const total=(r.Positif||0)+(r.Netral||0)+(r.Negatif||0)||1; return `<div class="stack-row"><div class="bar-label">${{r.Aspek}}</div><div class="stack-bar"><div class="seg-pos" style="width:${{(r.Positif||0)/total*100}}%"></div><div class="seg-net" style="width:${{(r.Netral||0)/total*100}}%"></div><div class="seg-neg" style="width:${{(r.Negatif||0)/total*100}}%"></div></div></div>` }}).join('')}}</div>`; }}
function renderPrediksi() {{ document.getElementById('page-prediksi').innerHTML = `${{pageHead('Prediksi Ulasan','Analisis sentimen dan aspek secara real-time menggunakan model ABSA.')}}${{predictionKpis()}}<div class="grid-12"><div class="col-7 card panel"><h3 class="headline" style="font-size:18px;line-height:24px;margin-bottom:18px;color:var(--primary)">Masukkan Teks Ulasan</h3><textarea id="reviewText" placeholder="Ketik atau paste ulasan pengguna di sini..."></textarea><p id="modelStatusText" class="body-muted" style="margin:14px 0 0">Mengecek kesiapan model...</p><div style="display:flex;justify-content:flex-end;margin-top:28px"><button id="predictBtn" class="primary-btn" onclick="predictReview()" disabled><span class="material-symbols-outlined">analytics</span><span id="predictBtnText">Menyiapkan Model</span></button></div></div><div class="col-5 card panel"><h3 class="panel-title" style="font-size:18px;line-height:24px;border-bottom:1px solid var(--outline-variant);padding-bottom:16px;margin-bottom:32px">Hasil Analisis Aspek & Sentimen</h3><div id="predictionResult">${{resultSummary('functional suitability','Positif',0.94)}}</div></div></div>`; refreshModelStatus(); }}
function humanError(message, detail='') {{ return `<div class="prediction-result-card"><div><div class="prediction-result-title">Prediksi belum bisa diproses</div><div class="prediction-result-confidence">${{message}}${{detail?'<br><span style="font-weight:600">'+detail+'</span>':''}}</div></div><div class="sentiment-pill netral">Info</div></div>`; }}
function processingSummary(message) {{ return `<div class="prediction-result-card"><div><div class="prediction-result-title">Prediksi sedang diproses</div><div class="prediction-result-confidence">${{message}}</div></div><div class="sentiment-pill netral">Info</div></div>`; }}
async function readJsonResponse(res) {{ const raw=await res.text(); try {{ return JSON.parse(raw); }} catch(err) {{ return {{ok:false, message:'Server belum mengirim respons prediksi yang valid.', detail:'Biasanya Streamlit perlu dijalankan ulang, endpoint /predict belum aktif, atau artefak model belum lengkap.'}}; }} }}
function setPredictButton(enabled, label) {{ const btn=document.getElementById('predictBtn'); const text=document.getElementById('predictBtnText'); if(btn) btn.disabled=!enabled; if(text) text.textContent=label; }}
async function refreshModelStatus() {{ const label=document.getElementById('modelStatusText'); try {{ const res=await fetch('/model-status'); const out=await readJsonResponse(res); if(out.ready) {{ if(label) label.textContent='Model siap digunakan.'; setPredictButton(true,'Prediksi Ulasan'); return; }} if(label) label.textContent=out.message||'Model sedang disiapkan.'; setPredictButton(false, out.state==='error'?'Model Belum Siap':'Menyiapkan Model'); if(out.state==='idle'||out.state==='loading') setTimeout(refreshModelStatus, 1200); }} catch(err) {{ if(label) label.textContent='Status model belum bisa dicek.'; setPredictButton(false,'Model Belum Siap'); }} }}
async function pollPredictionResult(jobId, startedAt) {{ const box=document.getElementById('predictionResult'); try {{ const res=await fetch('/predict-result?id='+encodeURIComponent(jobId)); const out=await readJsonResponse(res); if(out.state==='queued'||out.state==='running') {{ const sec=out.running_seconds || Math.max(1, Math.round((Date.now()-startedAt)/1000)); const stage=out.stage_message || out.message || 'Model sedang memproses ulasan.'; const stageName=out.stage ? `Tahap: ${{out.stage}} · ` : ''; box.innerHTML=processingSummary(`${{stageName}}${{stage}} (${{sec}} dtk)`); setTimeout(()=>pollPredictionResult(jobId, startedAt), 700); return; }} if(out.state==='done' && out.result && out.result.ok) {{ const r=out.result; const note=`Aspek: ${{Math.round((r.aspectConfidence||r.confidence||0)*100)}}% · Sentimen: ${{Math.round((r.sentimentConfidence||r.confidence||0)*100)}}%`; box.innerHTML=resultSummary(r.aspect, r.sentiment, r.confidence||0, note); setPredictButton(true,'Prediksi Ulasan'); return; }} const err=(out.result&&out.result.message)||out.message||'Prediksi belum bisa diproses.'; box.innerHTML=humanError(err); setPredictButton(true,'Prediksi Ulasan'); refreshModelStatus(); }} catch(err) {{ box.innerHTML=humanError('Dashboard belum berhasil mengambil hasil prediksi.'); setPredictButton(true,'Prediksi Ulasan'); }} }}
async function predictReview() {{ const raw=document.getElementById('reviewText').value; const box=document.getElementById('predictionResult'); if(!raw.trim()) {{ box.innerHTML=humanError('Teks ulasan masih kosong.'); return; }} setPredictButton(false,'Memproses'); box.innerHTML=processingSummary('Mengirim ulasan ke server model...'); try {{ const res=await fetch('/predict-job', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify({{text:raw}})}}); const out=await readJsonResponse(res); if(!res.ok || !out.ok) {{ box.innerHTML=humanError(out.message||'Model belum siap dipakai.'); setPredictButton(true,'Prediksi Ulasan'); refreshModelStatus(); return; }} pollPredictionResult(out.job_id, Date.now()); }} catch(err) {{ box.innerHTML=humanError('Dashboard belum berhasil terhubung ke model.'); setPredictButton(true,'Prediksi Ulasan'); }} }}
function renderOverview() {{ document.getElementById('page-overview').innerHTML = `${{pageHead('Ikhtisar Data',`Ringkasan komprehensif sentimen ulasan pengguna Livin' by Mandiri.`)}}${{globalKpis()}}<div class="grid-12"><div class="col-4 card panel"><h3 class="panel-title">Komposisi Sentimen</h3>${{donut()}}</div><div class="col-8 card panel"><h3 class="panel-title">Distribusi Aspek</h3>${{bars(currentData.aspectCounts)}}</div><div class="col-12 card panel"><h3 class="panel-title">Sentimen per Aspek</h3>${{stackBars()}}</div></div>`; }}
function selectIpaAspect(aspect) {{ selectedIpaAspect=aspect; updateIpaSelection(); }}
function clearIpaAspect() {{ selectedIpaAspect=null; updateIpaSelection(); }}
function ipaAspectDetail(row) {{
  const cross=currentData.cross.find(r=>r.Aspek===row.Aspek)||{{Positif:row.positive||0, Netral:0, Negatif:row.negative||0}};
  const q={{A:'Prioritas Utama', B:'Pertahankan', C:'Prioritas Rendah', D:'Berlebihan'}}[row.quadrant]||row.quadrant;
  return `<h3 class="headline">Detail Aspek</h3><p class="body-muted"><b>${{row.Aspek}}</b> berada di Kuadran ${{row.quadrant}}: ${{q}}.</p><div class="ipa-detail-grid"><div class="ipa-detail-metric"><span>Total Ulasan</span><b>${{fmt.format(row.importance)}}</b></div><div class="ipa-detail-metric"><span>Positif</span><b>${{fmt.format(cross.Positif||0)}}</b></div><div class="ipa-detail-metric"><span>Netral</span><b>${{fmt.format(cross.Netral||0)}}</b></div><div class="ipa-detail-metric"><span>Negatif</span><b>${{fmt.format(cross.Negatif||0)}}</b></div><div class="ipa-detail-metric"><span>Importance</span><b>${{row.importance_score_raw.toFixed(2)}}</b></div><div class="ipa-detail-metric"><span>Performance</span><b>${{pct(row.performance)}}</b></div><div class="ipa-detail-metric"><span>Z Performance</span><b>${{row.performance_score.toFixed(2)}}</b></div><div class="ipa-detail-metric"><span>Z Importance</span><b>${{row.importance_score.toFixed(2)}}</b></div></div><button class="secondary-link" type="button" onclick="clearIpaAspect();"><span class="material-symbols-outlined">close</span>Batal pilih</button>`;
}}
function ipaSidePanel() {{
  const selected=currentData.ipa.find(r=>r.Aspek===selectedIpaAspect);
  if(selected) return ipaAspectDetail(selected);
  const pri=currentData.ipa.filter(r=>r.quadrant==='A');
  return `<h3 class="headline">Tindakan Diperlukan</h3><p class="body-muted">Klik titik atau label pada chart untuk melihat informasi detail per aspek.</p><p class="ipa-hint">Ringkasan Kuadran A</p>${{pri.map((r,i)=>`<button class="priority-card" type="button" onclick="selectIpaAspect(${{JSON.stringify(r.Aspek)}})" style="width:100%;text-align:left;cursor:pointer"><h4>${{i+1}}. ${{r.Aspek}}</h4><div>Importance: <b>${{r.importance_score_raw.toFixed(2)}}</b></div><div>Performance: <b>${{pct(r.performance)}}</b></div><div>Z: <b>(${{r.performance_score.toFixed(2)}}, ${{r.importance_score.toFixed(2)}})</b></div></button>`).join('') || '<div class="priority-card"><h4>Tidak ada Kuadran A</h4></div>'}}`;
}}
function updateIpaSelection() {{ const panel=document.getElementById('ipaSidePanel'); if(panel) panel.innerHTML=ipaSidePanel(); document.querySelectorAll('.point').forEach(el=>el.classList.toggle('selected', decodeURIComponent(el.dataset.aspect||'')===selectedIpaAspect)); }}
function bindIpaInteractions() {{ document.querySelectorAll('.point').forEach(el=>{{ el.addEventListener('click', event=>{{ event.preventDefault(); event.stopPropagation(); selectIpaAspect(decodeURIComponent(el.dataset.aspect||'')); }}); }}); document.querySelectorAll('[data-ipa-row]').forEach(el=>{{ el.addEventListener('click', ()=>selectIpaAspect(decodeURIComponent(el.dataset.ipaRow||''))); }}); }}
function renderIpa() {{ const pri=currentData.ipa.filter(r=>r.quadrant==='A'); const maxX=Math.max(1,...currentData.ipa.map(r=>Math.abs(r.performance_score))); const maxY=Math.max(1,...currentData.ipa.map(r=>Math.abs(r.importance_score))); const points=currentData.ipa.map(r=>{{ const x=Math.max(3, Math.min(97, 50+(r.performance_score/maxX)*47)); const y=Math.max(3, Math.min(97, 50+(r.importance_score/maxY)*47)); const selected=r.Aspek===selectedIpaAspect?' selected':''; return `<div class="point ${{r.quadrant.toLowerCase()}}${{selected}}" role="button" tabindex="0" data-aspect="${{encodeURIComponent(r.Aspek)}}" title="${{r.Aspek}}: Z(${{r.performance_score.toFixed(2)}}, ${{r.importance_score.toFixed(2)}})" style="left:${{x}}%;bottom:${{y}}%"><span>${{r.Aspek}}</span></div>`; }}).join(''); document.getElementById('page-ipa').innerHTML = `${{pageHead('Analisis Kepentingan dan Kinerja (IPA)','Evaluasi aspek aplikasi berdasarkan tingkat kepentingan pengguna vs kinerja aktual.')}}<div class="kpi-grid">${{kpi('Total Aspek', currentData.ipa.length, 'Dianalisis')}}${{kpi('Prioritas Utama (A)', pri.length, 'Perlu perbaikan')}}${{kpi('Pertahankan (B)', currentData.ipa.filter(r=>r.quadrant==='B').length, 'Kinerja relatif baik')}}${{kpi('Titik Pusat', '0,00 ; 0,00', 'Z-Score Performance dan Importance')}}</div><div class="grid-12"><div class="col-8 card panel"><h3 class="headline">Plot Sebar Matriks IPA</h3><div class="ipa-area"><div class="quad qa"></div><div class="quad qb"></div><div class="quad qc"></div><div class="quad qd"></div><div class="mid-v"></div><div class="mid-h"></div><div class="quad-label" style="top:16px;left:16px;color:#ba1a1a">A: Prioritas Utama</div><div class="quad-label" style="top:16px;right:16px;color:#002752">B: Pertahankan</div><div class="quad-label" style="bottom:16px;left:16px;color:#737781">C: Prioritas Rendah</div><div class="quad-label" style="bottom:16px;right:16px;color:#7c5800">D: Berlebihan</div>${{points}}</div></div><div id="ipaSidePanel" class="col-4 card panel">${{ipaSidePanel()}}</div><div class="col-12 card panel"><h3 class="panel-title">Tabel Agregasi IPA</h3>${{ipaTable()}}</div></div>`; bindIpaInteractions(); }}
function ipaTable() {{ return `<table class="table"><thead><tr><th>Rank</th><th>Aspek</th><th>Importance</th><th>Performance</th><th>Z Performance</th><th>Z Importance</th><th>Kuadran</th></tr></thead><tbody>${{currentData.ipa.map(r=>`<tr class="clickable-row" onclick="selectIpaAspect(${{JSON.stringify(r.Aspek)}})"><td>${{r.rank}}</td><td>${{r.Aspek}}</td><td>${{r.importance_score_raw.toFixed(2)}}</td><td>${{pct(r.performance)}}</td><td>${{r.performance_score.toFixed(2)}}</td><td>${{r.importance_score.toFixed(2)}}</td><td>${{r.quadrant}}</td></tr>`).join('')}}</tbody></table>`; }}
function renderDetail() {{ const aspects=[...new Set(currentData.reviews.map(r=>r.Aspek))].sort(); reviewPage=1; document.getElementById('page-detail').innerHTML = `${{pageHead('Daftar Ulasan','Tinjau dan telusuri data ulasan berdasarkan aspek, sentimen, dan kata kunci.')}}<div class="card panel"><div class="filters"><select id="fAspect"><option>Semua</option>${{aspects.map(a=>`<option>${{a}}</option>`).join('')}}</select><select id="fSent"><option>Semua</option><option>Negatif</option><option>Positif</option><option>Netral</option></select><input id="fKey" type="text" placeholder="Cari keyword ulasan" /></div><div style="height:18px"></div><div id="reviewTable"></div></div>`; ['fAspect','fSent','fKey'].forEach(id=>document.getElementById(id).addEventListener('input', ()=>{{ reviewPage=1; updateReviews(); }})); updateReviews(); }}
function filteredReviews() {{ const a=document.getElementById('fAspect')?.value||'Semua'; const s=document.getElementById('fSent')?.value||'Semua'; const k=(document.getElementById('fKey')?.value||'').toLowerCase(); return currentData.reviews.filter(r=>(a==='Semua'||r.Aspek===a)&&(s==='Semua'||r.Sentimen===s)&&(!k||String(r.Ulasan).toLowerCase().includes(k))); }}
function setReviewPage(delta) {{ reviewPage += delta; updateReviews(); }}
function setReviewPageSize(value) {{ reviewPageSize=parseInt(value,10)||25; reviewPage=1; updateReviews(); }}
function fmtCell(value) {{ if(value===undefined || value===null || value==='' || String(value)==='NaN') return '-'; if(typeof value==='number') return Number.isInteger(value)?fmt.format(value):value.toFixed(4); if(value===true) return 'Benar'; if(value===false) return 'Salah'; return String(value); }}
function updateReviews() {{ const rows=filteredReviews(); const totalPages=Math.max(1, Math.ceil(rows.length/reviewPageSize)); reviewPage=Math.max(1, Math.min(reviewPage,totalPages)); const start=(reviewPage-1)*reviewPageSize; const pageRows=rows.slice(start,start+reviewPageSize); const cols=['Ulasan','Aspek','Sentimen','True_Aspek','True_Sentimen','Confidence_Aspek','Confidence_Sentimen','Prediksi_Benar_Aspek','Prediksi_Benar_Sentimen','Prediksi_Benar_End_to_End','Sumber_Data']; document.getElementById('reviewTable').innerHTML=`<p class="body-muted"><b>${{fmt.format(rows.length)}}</b> ulasan ditemukan. Menampilkan ${{rows.length?fmt.format(start+1):0}}-${{fmt.format(Math.min(start+pageRows.length, rows.length))}}.</p><div style="overflow-x:auto"><table class="table"><thead><tr>${{cols.map(c=>`<th>${{c}}</th>`).join('')}}</tr></thead><tbody>${{pageRows.map(r=>`<tr>${{cols.map(c=>`<td>${{fmtCell(r[c])}}</td>`).join('')}}</tr>`).join('') || `<tr><td colspan="${{cols.length}}">Tidak ada ulasan yang cocok.</td></tr>`}}</tbody></table></div><div class="pagination"><label class="page-size">Baris per halaman <select onchange="setReviewPageSize(this.value)"><option value="10" ${{reviewPageSize===10?'selected':''}}>10</option><option value="25" ${{reviewPageSize===25?'selected':''}}>25</option><option value="50" ${{reviewPageSize===50?'selected':''}}>50</option><option value="100" ${{reviewPageSize===100?'selected':''}}>100</option></select></label><div class="page-controls"><button class="page-btn" onclick="setReviewPage(-1)" ${{reviewPage<=1?'disabled':''}}><span class="material-symbols-outlined">chevron_left</span></button><div class="page-info">Halaman ${{fmt.format(reviewPage)}} / ${{fmt.format(totalPages)}}</div><button class="page-btn" onclick="setReviewPage(1)" ${{reviewPage>=totalPages?'disabled':''}}><span class="material-symbols-outlined">chevron_right</span></button></div></div>`; }}
function showPage(page) {{ document.querySelectorAll('.page').forEach(p=>p.classList.add('hidden')); document.getElementById('page-'+page).classList.remove('hidden'); document.querySelectorAll('.nav-btn').forEach(b=>b.classList.toggle('active', b.dataset.page===page)); if(page==='prediksi') renderPrediksi(); if(page==='overview') renderOverview(); if(page==='ipa') renderIpa(); if(page==='detail') renderDetail(); }}
document.querySelectorAll('.nav-btn').forEach(b=>b.addEventListener('click',()=>showPage(b.dataset.page)));
document.getElementById('downloadTemplate').addEventListener('click',()=>{{ const blob=new Blob([currentData.templateCsv],{{type:'text/csv'}}); const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='template_upload_ulasan.csv'; a.click(); }});
function parseCsv(text) {{ const lines=text.trim().replaceAll(String.fromCharCode(13),'').split(String.fromCharCode(10)); const headers=lines[0].split(',').map(h=>h.trim()); return lines.slice(1).map(line=>{{ const vals=line.match(/("[^"]*(?:""[^"]*)*"|[^,]*)/g).filter((_,i)=>i%2===0).map(v=>v.replace(/^"|"$/g,'').replace(/""/g,'"')); const obj={{}}; headers.forEach((h,i)=>obj[h]=vals[i]||''); return obj; }}); }}

const modelFileNameBindings = [
  ['modelAspekFile','modelAspekName'], ['modelSentimenFile','modelSentimenName'], ['tokenizerFile','tokenizerName'],
  ['encoderAspekFile','encoderAspekName'], ['encoderSentimenFile','encoderSentimenName']
];
modelFileNameBindings.forEach(([inputId,nameId])=>{{
  const input=document.getElementById(inputId); const label=document.getElementById(nameId);
  input.addEventListener('change',()=>{{ label.textContent=input.files[0]?.name || 'Belum dipilih'; }});
}});
async function uploadModelFiles() {{
  const status=document.getElementById('modelUploadStatus');
  const files={{model_aspek:document.getElementById('modelAspekFile').files[0], model_sentimen:document.getElementById('modelSentimenFile').files[0], tokenizer:document.getElementById('tokenizerFile').files[0], encoder_aspek:document.getElementById('encoderAspekFile').files[0], encoder_sentimen:document.getElementById('encoderSentimenFile').files[0]}};
  const form=new FormData(); let count=0;
  Object.entries(files).forEach(([key,file])=>{{ if(file) {{ form.append(key,file); count++; }} }});
  if(!count) {{ status.textContent='Pilih minimal satu file artefak model.'; return; }}
  status.textContent='Mengunggah artefak model...';
  try {{
    const res=await fetch('/upload-model', {{method:'POST', body:form}});
    const out=await readJsonResponse(res);
    if(!res.ok || !out.ok) {{ status.textContent=out.message||'Upload model belum berhasil.'; return; }}
    status.textContent='Upload berhasil. Tekan Prediksi Ulasan untuk memakai model terbaru.';
  }} catch(err) {{ status.textContent='Upload gagal: '+err.message+'. Jalankan ulang Streamlit lalu coba lagi.'; }}
}}
document.getElementById('uploadModelBtn').addEventListener('click', uploadModelFiles);

document.getElementById('csvInput').addEventListener('change', e=>{{ const file=e.target.files[0]; if(!file) return; const reader=new FileReader(); reader.onload=()=>{{ const rows=parseCsv(reader.result); const reviews=rows.map(r=>{{ const sent=r.sentimen||r.Sentimen||r.sentimen_llm; return {{...r, Ulasan:r.ulasan||r.Ulasan||r.final_text, Aspek:(r.aspek||r.Aspek||r.aspek_llm||'').toLowerCase(), Sentimen:sent==='1'||sent==='Positif'?'Positif':sent==='2'||sent==='Netral'?'Netral':sent==='0'||sent==='Negatif'?'Negatif':sent}}; }}).filter(r=>r.Ulasan); currentData=computeDashboardData(reviews); alert('CSV berhasil dibaca dan kalkulasi dashboard sudah diperbarui.'); showPage('overview'); }}; reader.readAsText(file); }});
showPage('prediksi');
</script>
</body>
</html>
"""


df = load_dataset()
payload = compute_payload(df)

st.markdown(
    """
    <style>
    header[data-testid="stHeader"], div[data-testid="stToolbar"], div[data-testid="stDecoration"], div[data-testid="stStatusWidget"], [data-testid="stSidebar"] { display: none !important; }
    .block-container { padding: 0 !important; margin: 0 !important; max-width: none !important; }
    iframe { display: block; }
    </style>
    """,
    unsafe_allow_html=True,
)
html = build_html(payload)
static_path = Path("dashboard_static.html")
if static_path.exists():
    static_path.unlink()
static_path.write_text(html, encoding="utf-8")

import sys
import traceback as _tb

def _safe_print(msg):
    try:
        print(msg)
        logger.info(msg)
    except Exception:
        pass

def _safe_print_exc():
    try:
        logger.exception("dashboard_server_exception")
        _tb.print_exc()
    except Exception:
        pass

_server_ref = getattr(st, "_dashboard_server", None)

def _start_dashboard_server():
    global _server_ref

    class DashboardHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send_json(self, payload, status=200):
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def end_headers(self):
            if self.path.startswith("/dashboard_static.html"):
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
            super().end_headers()

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            try:
                parsed = urlparse(self.path)
                if parsed.path not in {"/predict-result", "/dashboard_static.html"}:
                    log_event("http_get", path=parsed.path)
                if parsed.path == "/health":
                    self.send_json({
                        "ok": True,
                        "service": "dashboard_static_server",
                        "static_exists": static_path.exists(),
                        "static_path": str(static_path.resolve()),
                        "timestamp": time.time(),
                    })
                    return
                if parsed.path == "/dashboard_static.html":
                    if not static_path.exists():
                        self.send_json({
                            "ok": False,
                            "message": "dashboard_static.html belum dibuat.",
                            "static_path": str(static_path.resolve()),
                        }, status=404)
                        return
                    raw = static_path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(raw)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                if parsed.path == "/model-status":
                    preload_model_async()
                    self.send_json(get_model_status())
                    return
                if parsed.path == "/predict-result":
                    query = parse_qs(parsed.query)
                    job_id = (query.get("id") or [""])[0]
                    self.send_json(get_prediction_job(job_id))
                    return
                if parsed.path == "/developer-log":
                    lines = int((parse_qs(parsed.query).get("lines") or ["80"])[0] or 80)
                    lines = max(10, min(lines, 300))
                    if LOG_FILE.exists():
                        content = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
                        self.send_json({"ok": True, "log_file": str(LOG_FILE.resolve()), "lines": content[-lines:]})
                    else:
                        self.send_json({"ok": False, "message": "File log belum dibuat.", "log_file": str(LOG_FILE.resolve())})
                    return
                return super().do_GET()
            except Exception as exc:
                _safe_print(f"[dashboard] GET ERROR: {exc}")
                _safe_print_exc()
                try:
                    self.send_json({"ok": False, "message": "Permintaan belum bisa diproses oleh server."}, status=500)
                except Exception:
                    pass

        def do_POST(self):
            try:
                log_event("http_post", path=self.path)
                if self.path == "/predict":
                    length = int(self.headers.get("Content-Length", "0") or 0)
                    body = self.rfile.read(length).decode("utf-8") if length else "{}"
                    payload = json.loads(body)
                    request_id = uuid.uuid4().hex
                    self.send_json(predict_with_model(payload.get("text", ""), job_id=request_id))
                    return
                if self.path == "/predict-job":
                    length = int(self.headers.get("Content-Length", "0") or 0)
                    body = self.rfile.read(length).decode("utf-8") if length else "{}"
                    payload = json.loads(body)
                    result = start_prediction_job(payload.get("text", ""), run_async=False)
                    if not result.get("ok") or not result.get("job_id"):
                        self.send_json(result)
                        return
                    job_id = result["job_id"]
                    try:
                        update_prediction_job(
                            job_id,
                            state="running",
                            message="Model sedang memproses ulasan.",
                            stage="request_thread",
                            stage_message="Prediksi berjalan langsung di request thread.",
                            stage_started_at=time.time(),
                        )
                        prediction = predict_with_model(payload.get("text", ""), job_id=job_id)
                        update_prediction_job(
                            job_id,
                            state="done" if prediction.get("ok") else "error",
                            result=prediction,
                            message=prediction.get("message", "Prediksi selesai."),
                            finished_at=time.time(),
                        )
                    except Exception as exc:
                        logger.exception("prediction_request_failed | job_id=%s", job_id)
                        update_prediction_job(
                            job_id,
                            state="error",
                            result={"ok": False, "message": "Prediksi belum bisa diproses oleh server model."},
                            message="Prediksi belum bisa diproses oleh server model.",
                            detail=str(exc),
                            stage="error",
                            stage_message="Prediksi gagal. Lihat traceback di log.",
                            finished_at=time.time(),
                        )
                    self.send_json(get_prediction_job(job_id))
                    return
                if self.path == "/upload-model":
                    if not ALLOW_MODEL_UPLOADS:
                        log_event("upload_model_blocked")
                        self.send_json({"ok": False, "message": "Upload model dinonaktifkan di server publik."}, status=403)
                        return
                    content_type = self.headers.get("Content-Type", "")
                    length = int(self.headers.get("Content-Length", "0") or 0)
                    log_event("upload_model_request", content_type=content_type, content_length=length)
                    if "multipart/form-data" not in content_type or not length:
                        self.send_json({"ok": False, "message": "Request bukan multipart/form-data."}, status=400)
                        return

                    form = cgi.FieldStorage(
                        fp=self.rfile,
                        headers=self.headers,
                        environ={
                            "REQUEST_METHOD": "POST",
                            "CONTENT_TYPE": content_type,
                            "CONTENT_LENGTH": str(length),
                        },
                    )
                    saved = []
                    for field_name, target in MODEL_UPLOADS.items():
                        if field_name not in form:
                            continue
                        item = form[field_name]
                        if isinstance(item, list):
                            item = item[0]
                        if not getattr(item, "filename", ""):
                            continue
                        file_content = item.file.read()
                        target.parent.mkdir(parents=True, exist_ok=True)
                        tmp_target = target.with_suffix(target.suffix + ".uploading")
                        tmp_target.write_bytes(file_content)
                        tmp_target.replace(target)
                        saved.append(target.name)
                        log_event("upload_model_saved", file=target.name, bytes=len(file_content))
                    reset_model_cache()
                    preload_model_async()
                    if not saved:
                        log_event("upload_model_no_valid_files")
                        self.send_json({"ok": False, "message": "Tidak ada artefak model valid yang diterima."}, status=400)
                        return
                    log_event("upload_model_done", saved=",".join(saved))
                    self.send_json({"ok": True, "saved": saved, "message": "Artefak model berhasil diunggah."})
                    return
                self.send_json({"ok": False, "message": "Alamat endpoint tidak dikenal."}, status=404)
            except Exception as exc:
                _safe_print(f"[upload-model] ERROR: {exc}")
                _safe_print_exc()
                try:
                    self.send_json({"ok": False, "message": "Permintaan belum bisa diproses oleh server model.", "detail": str(exc)}, status=500)
                except Exception:
                    pass

    old_server = getattr(st, "_dashboard_server", None)
    if old_server is not None:
        try:
            old_server.shutdown()
            old_server.server_close()
            log_event("dashboard_server_old_closed")
        except Exception:
            logger.exception("dashboard_server_old_close_failed")

    class ReusableServer(ThreadingHTTPServer):
        allow_reuse_address = True
        allow_reuse_port = True

    try:
        server = ReusableServer(("127.0.0.1", 8765), DashboardHandler)
        st._dashboard_server = server
        threading.Thread(target=server.serve_forever, daemon=True).start()
        preload_model_async()
        _safe_print("[dashboard] Server started on port 8765")
    except OSError as e:
        _safe_print(f"[dashboard] Could not start server: {e}")

def _dashboard_server_is_alive():
    try:
        with urlopen("http://127.0.0.1:8765/health", timeout=1.5) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
            return bool(payload.get("ok") and payload.get("static_exists"))
    except Exception as exc:
        log_event("dashboard_server_healthcheck_failed", detail=str(exc))
        return False

if not _dashboard_server_is_alive():
    _start_dashboard_server()

components.html(html, height=920, scrolling=True)
