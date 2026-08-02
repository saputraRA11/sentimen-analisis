from pathlib import Path
import csv
import json
import tempfile
import time
import zipfile

import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


MODEL_DIR = Path("models")
MODEL_ASPEK_FILE = MODEL_DIR / "model_aspek_raw.keras"
MODEL_SENTIMEN_FILE = MODEL_DIR / "model_sentimen_cascaded.keras"
TOKENIZER_FILE = MODEL_DIR / "tokenizer_absa.joblib"
ENCODER_ASPEK_FILE = MODEL_DIR / "encoder_aspek.joblib"
ENCODER_SENTIMEN_FILE = MODEL_DIR / "encoder_sentimen.joblib"
MODEL_CONFIG_FILE = MODEL_DIR / "dashboard_model_config.joblib"
DATASET_FILE = Path("data_test_prediksi.csv")

SAMPLE_TEXTS = [
    ("compatibility", "setelah update aplikasi tidak cocok di hp android saya dan selalu minta versi terbaru"),
    ("flexibility", "pilihan pengaturan notifikasi dan limit transaksi kurang fleksibel tidak bisa disesuaikan"),
    ("functional suitability", "fitur transfer pembayaran qris dan top up tidak berfungsi transaksi selalu gagal"),
    ("interaction capability", "tampilan menu membingungkan tombol sulit ditemukan dan alur penggunaan ribet"),
    ("maintainability", "bug aplikasi belum diperbaiki setelah update terbaru masih banyak masalah"),
    ("performance efficiency", "aplikasi sangat lemot loading lama berat dan sering lag saat dibuka"),
    ("reliability", "aplikasi sering error crash tidak bisa dibuka dan layanan gangguan"),
    ("safety", "saldo terpotong tetapi transaksi gagal uang tidak kembali ke rekening"),
    ("security", "tidak bisa login kode otp gagal verifikasi wajah dan akun terkunci"),
]
DATASET_PROBE_LIMIT_PER_ASPECT = 30


def step(name):
    print(f"\n=== {name} ===", flush=True)
    return time.perf_counter()


def done(start):
    print(f"done in {(time.perf_counter() - start):.3f}s", flush=True)


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
    try:
        return load_model(path, compile=False)
    except Exception as original_error:
        if not zipfile.is_zipfile(path):
            raise

        print(f"normal load failed for {path.name}; retry without quantization_config", flush=True)
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(tmp_path, "w") as target:
                for name in source.namelist():
                    data = source.read(name)
                    if name == "config.json":
                        config = strip_keras_incompatible_config(json.loads(data.decode("utf-8")))
                        data = json.dumps(config).encode("utf-8")
                    target.writestr(name, data)
            return load_model(tmp_path, compile=False)
        except Exception:
            raise original_error
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass


def predict_one(text, tokenizer, max_len, encoder_aspek, encoder_sentimen, model_aspek, model_sentimen):
    seq = tokenizer.texts_to_sequences([text])
    x = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

    aspek_prob = model_aspek.predict(x, verbose=0)
    aspek_idx = np.argmax(aspek_prob, axis=1)
    aspek_label = encoder_aspek.inverse_transform(aspek_idx)[0]

    sentimen_prob = model_sentimen.predict([x, aspek_idx], verbose=0)
    sentimen_idx = np.argmax(sentimen_prob, axis=1)
    sentimen_label = encoder_sentimen.inverse_transform(sentimen_idx)[0]

    top_idx = np.argsort(aspek_prob[0])[::-1][:3]
    top_aspects = [
        (encoder_aspek.inverse_transform([idx])[0], float(aspek_prob[0][idx]))
        for idx in top_idx
    ]

    return {
        "sequence": seq,
        "non_zero": int(np.count_nonzero(x)),
        "aspek_label": aspek_label,
        "aspek_confidence": float(np.max(aspek_prob)),
        "sentimen_label": sentimen_label,
        "sentimen_confidence": float(np.max(sentimen_prob)),
        "top_aspects": top_aspects,
    }


def load_dataset_probe_rows(path, limit_per_aspect):
    if not path.exists():
        return []

    selected = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            aspect = (row.get("Aspek") or row.get("aspek") or row.get("aspek_llm") or "").strip()
            text = (row.get("Ulasan") or row.get("ulasan") or row.get("final_text") or "").strip()
            if not aspect or not text:
                continue
            selected.setdefault(aspect, [])
            if len(selected[aspect]) < limit_per_aspect:
                selected[aspect].append(text)

    rows = []
    for aspect in sorted(selected):
        rows.extend((aspect, text) for text in selected[aspect])
    return rows


def main():
    start = step("check files")
    required = [
        MODEL_ASPEK_FILE,
        MODEL_SENTIMEN_FILE,
        TOKENIZER_FILE,
        ENCODER_ASPEK_FILE,
        ENCODER_SENTIMEN_FILE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing artifacts: {missing}")
    for path in required:
        print(f"{path}: {path.stat().st_size:,} bytes", flush=True)
    done(start)

    start = step("load config")
    config = joblib.load(MODEL_CONFIG_FILE) if MODEL_CONFIG_FILE.exists() else {"MAX_LEN": 50}
    max_len = int(config.get("MAX_LEN", 50))
    print(f"MAX_LEN={max_len}", flush=True)
    done(start)

    start = step("load tokenizer and encoders")
    tokenizer = joblib.load(TOKENIZER_FILE)
    encoder_aspek = joblib.load(ENCODER_ASPEK_FILE)
    encoder_sentimen = joblib.load(ENCODER_SENTIMEN_FILE)
    print(f"tokenizer vocab={len(getattr(tokenizer, 'word_index', {})):,}", flush=True)
    print(f"aspek classes={list(encoder_aspek.classes_)}", flush=True)
    print(f"sentimen classes={list(encoder_sentimen.classes_)}", flush=True)
    done(start)

    start = step("load model aspek")
    model_aspek = load_keras_compatible(MODEL_ASPEK_FILE)
    print("model_aspek input_shape:", getattr(model_aspek, "input_shape", None), flush=True)
    print("model_aspek output_shape:", getattr(model_aspek, "output_shape", None), flush=True)
    done(start)

    start = step("load model sentimen")
    model_sentimen = load_keras_compatible(MODEL_SENTIMEN_FILE)
    print("model_sentimen input_shape:", getattr(model_sentimen, "input_shape", None), flush=True)
    print("model_sentimen output_shape:", getattr(model_sentimen, "output_shape", None), flush=True)
    done(start)

    start = step("predict samples for all aspects")
    predicted_counts = {}
    matched = 0
    for expected_aspect, text in SAMPLE_TEXTS:
        result = predict_one(
            text,
            tokenizer,
            max_len,
            encoder_aspek,
            encoder_sentimen,
            model_aspek,
            model_sentimen,
        )
        predicted_counts[result["aspek_label"]] = predicted_counts.get(result["aspek_label"], 0) + 1
        matched += int(result["aspek_label"] == expected_aspect)
        top_aspects = ", ".join(
            f"{label}:{prob:.4f}" for label, prob in result["top_aspects"]
        )
        print(
            f"expected={expected_aspect} | predicted={result['aspek_label']} "
            f"| aspek_conf={result['aspek_confidence']:.6f} "
            f"| sentimen={result['sentimen_label']} "
            f"| sentimen_conf={result['sentimen_confidence']:.6f}",
            flush=True,
        )
        print(f"text={text!r}", flush=True)
        print(f"non_zero_tokens={result['non_zero']} | top3_aspek={top_aspects}", flush=True)
        print("-", flush=True)
    print(f"matched_expected={matched}/{len(SAMPLE_TEXTS)}", flush=True)
    print(f"predicted_aspect_counts={predicted_counts}", flush=True)
    done(start)

    start = step("dataset probe by expected aspect")
    probe_rows = load_dataset_probe_rows(DATASET_FILE, DATASET_PROBE_LIMIT_PER_ASPECT)
    if not probe_rows:
        print(f"skip dataset probe; {DATASET_FILE} not found or empty", flush=True)
    else:
        matrix = {}
        predicted_totals = {}
        for expected_aspect, text in probe_rows:
            result = predict_one(
                text,
                tokenizer,
                max_len,
                encoder_aspek,
                encoder_sentimen,
                model_aspek,
                model_sentimen,
            )
            predicted = result["aspek_label"]
            matrix.setdefault(expected_aspect, {})
            matrix[expected_aspect][predicted] = matrix[expected_aspect].get(predicted, 0) + 1
            predicted_totals[predicted] = predicted_totals.get(predicted, 0) + 1

        print(f"dataset_file={DATASET_FILE}", flush=True)
        print(f"rows_tested={len(probe_rows)} ({DATASET_PROBE_LIMIT_PER_ASPECT} per expected aspect max)", flush=True)
        print("predicted_totals_by_model:", flush=True)
        for aspect in encoder_aspek.classes_:
            print(f"  {aspect}: {predicted_totals.get(aspect, 0)}", flush=True)
        print("matrix expected_aspect -> predicted counts:", flush=True)
        for expected_aspect in sorted(matrix):
            counts = ", ".join(
                f"{aspect}:{matrix[expected_aspect].get(aspect, 0)}"
                for aspect in encoder_aspek.classes_
                if matrix[expected_aspect].get(aspect, 0)
            )
            print(f"  {expected_aspect} -> {counts}", flush=True)
    done(start)

    print("\nRESULT OK", flush=True)


if __name__ == "__main__":
    main()
