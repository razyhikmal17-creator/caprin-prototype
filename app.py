import json
import os
import threading
import time
from datetime import datetime
from typing import Literal, Optional

import pandas as pd
import streamlit as st
import telebot
from google import genai
from google.genai import types
from pydantic import BaseModel

st.set_page_config(page_title="CAPRIN AI — Smart Livestock Log", page_icon="🐐", layout="wide")

# Ambil kunci rahasia dari Streamlit Secrets
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", os.getenv("TELEGRAM_BOT_TOKEN", ""))
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
DATA_FILE = "caprin_records.json"
file_lock = threading.Lock()


# 1. Skema JSON Terstruktur untuk Gemini AI
class TernakRecord(BaseModel):
    nama_ternak: str
    kategori: Literal["penimbangan", "pakan", "kesehatan", "reproduksi", "lainnya"]
    bobot_kg: Optional[float] = None
    detail_aktivitas: str


def load_records() -> list[dict]:
    if not os.path.exists(DATA_FILE):
        return []
    with file_lock:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)


def save_record(entry: dict) -> None:
    with file_lock:
        records = []
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
        records.append(entry)
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


# 2. Bot Telegram + Ekstraktor Gemini di Background
@st.cache_resource
def start_telegram_worker(tg_token: str, gemini_api_key: str):
    if not tg_token or not gemini_api_key:
        return "MISSING_KEYS"

    ai_client = genai.Client(api_key=gemini_api_key)
    bot = telebot.TeleBot(tg_token, parse_mode="Markdown")

    system_prompt = (
        "Kamu adalah asisten pencatat ternak CAPRIN untuk peternak kambing dan domba. "
        "Ekstrak informasi dari pesan bahasa Indonesia sehari-hari ke format JSON. "
        "Jika nama atau nomor ternak tidak disebut, isi dengan 'Umum'."
    )

    @bot.message_handler(commands=["start", "help"])
    def send_welcome(message):
        bot.reply_to(
            message,
            "Halo! Kirim catatan ternak pakai bahasa sehari-hari aja.\n"
            "Contoh: _Si Cemong hari ini ditimbang 24.5 kg, dikasih konsentrat 500g._",
        )

    @bot.message_handler(func=lambda m: True)
    def handle_log(message):
        models_to_try = ["gemini-3.8-flash", "gemini-3.8-flash", "gemini-3.8-flash-lite"]
        last_error = None

        for model_name in models_to_try:
            try:
                response = ai_client.models.generate_content(
                    model=model_name,
                    contents=message.text,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_schema=TernakRecord,
                    ),
                )
                parsed: TernakRecord = response.parsed
                record_dict = {
                    "waktu": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "peternak": message.from_user.first_name or "Peternak",
                    "nama_ternak": parsed.nama_ternak,
                    "kategori": parsed.kategori,
                    "bobot_kg": parsed.bobot_kg,
                    "detail_aktivitas": parsed.detail_aktivitas,
                    "pesan_asli": message.text,
                }
                save_record(record_dict)

                bobot_str = f"\n• Bobot: *{parsed.bobot_kg} kg*" if parsed.bobot_kg else ""
                bot.reply_to(
                    message,
                    f"✅ *Tercatat di CAPRIN!*\n"
                    f"• Ternak: *{parsed.nama_ternak}*\n"
                    f"• Kategori: `{parsed.kategori}`{bobot_str}\n"
                    f"• Catatan: {parsed.detail_aktivitas}",
                )
                return
            except Exception as e:
                last_error = e
                time.sleep(2)

        bot.reply_to(
            message,
            "⏳ Server AI sedang sibuk sebentar. Coba kirim ulang catatanmu 10 detik lagi ya!",
        )

    thread = threading.Thread(target=bot.infinity_polling, kwargs={"skip_pending": True}, daemon=True)
    thread.start()
    return "RUNNING"


status = start_telegram_worker(TELEGRAM_TOKEN, GEMINI_KEY)

# 3. Tampilan Live Dashboard Streamlit
st.title("🐐 CAPRIN — AI Livestock Recording")
st.caption("Kirim chat ke bot Telegram kamu, dan data otomatis muncul di dasbor ini.")

if status == "MISSING_KEYS":
    st.warning("Isi `TELEGRAM_BOT_TOKEN` dan `GEMINI_API_KEY` di Secrets dulu ya!")
    st.stop()


@st.fragment(run_every=5)
def render_live_dashboard():
    records = load_records()
    if not records:
        st.info("Belum ada data masuk. Coba kirim pesan ke bot Telegram kamu sekarang!")
        return

    df = pd.DataFrame(records)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Catatan", len(df))
    col2.metric("Ternak Terpantau", df["nama_ternak"].nunique())
    valid_weights = df["bobot_kg"].dropna()
    avg_weight = f"{valid_weights.mean():.1f} kg" if not valid_weights.empty else "-"
    col3.metric("Rata-rata Bobot", avg_weight)

    weight_df = df.dropna(subset=["bobot_kg"])
    if not weight_df.empty:
        st.subheader("📈 Perkembangan Bobot Badan (kg)")
        st.line_chart(weight_df, x="waktu", y="bobot_kg", color="nama_ternak")

    st.subheader("📋 Log Aktivitas Terbaru")
    st.dataframe(df.iloc[::-1], use_container_width=True, hide_index=True)


render_live_dashboard()
