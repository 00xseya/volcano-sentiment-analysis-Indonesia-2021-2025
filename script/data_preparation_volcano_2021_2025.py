"""
Data Preparation Pipeline: Volcano Sentiment Analysis Indonesia (2021 - 2025)
=============================================================================
Script ini untuk alur lengkap DATA PREPARATION:
1. DATA SOURCE: Scraping daftar link video YouTube (2021-2025) menggunakan yt-dlp
   -> Disimpan ke: data/source/youtube_video_links_2021_2025.csv
2. EXTRACT DATA INTO CSV FILE: Mengambil seluruh komentar dari daftar link video menggunakan yt-dlp
   -> Disimpan ke: data/extracted/youtube_comments_extracted.csv (dan backup di data/raw/)
3. CLEANING_TEXT: Pembersihan dasar teks komentar dari emoji, URL, mention, tanda baca/simbol,
   case folding, duplikasi, dan missing values.
   -> Disimpan ke: data/cleaning/youtube_comments_cleaned.csv
4. CLEANSING_DATA: Standardisasi kualitas data lanjutan, meliputi:
   - Deteksi & eliminasi komentar spam judi online (judol: slot, maxwin, pragmatic, dsb.)
   - Reduksi karakter berulang berlebihan (elongated characters)
   - Normalisasi reduplikasi numerik (misal: orang2 -> orang orang)
   - Koreksi typo & normalisasi kata slang/singkatan media sosial bahasa Indonesia
   - Penyelarasan ungkapan religi, empati, dan negasi majemuk
   - Penyimpanan log audit komentar judol yang dibersihkan
   -> Disimpan ke: data/cleansing/youtube_comments_cleansed.csv

Penggunaan Fleksibel:
- Mode Interaktif : python3 script/data_preparation_volcano_2021_2025.py
- Mode Per Bagian : python3 script/data_preparation_volcano_2021_2025.py --step 4
                   (Pilihan: 1, 2, 3, 4, 3,4, all, scrape, extract, clean, cleanse)
- Bantuan / Help  : python3 script/data_preparation_volcano_2021_2025.py --help
"""

import os
import sys
import re
import time
import random
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pandas as pd

# Coba import library eksternal dengan pesan bantuan jika belum terinstal
try:
    import yt_dlp
except ImportError:
    yt_dlp = None
    print("[WARNING] Library 'yt-dlp' belum terinstal. Jalankan: pip install yt-dlp")


# =====================================================================
# KONFIGURASI DIREKTORI (Sesuai Struktur Folder Proyek)
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(BASE_DIR) == "script":
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
else:
    PROJECT_ROOT = BASE_DIR

DIR_SOURCE = os.path.join(PROJECT_ROOT, "data", "source")
DIR_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DIR_EXTRACTED = os.path.join(PROJECT_ROOT, "data", "extracted")
DIR_CLEANING = os.path.join(PROJECT_ROOT, "data", "cleaning")
DIR_CLEANSING = os.path.join(PROJECT_ROOT, "data", "cleansing")

for folder in [DIR_SOURCE, DIR_RAW, DIR_EXTRACTED, DIR_CLEANING, DIR_CLEANSING]:
    os.makedirs(folder, exist_ok=True)

LINKS_CSV_PATH = os.path.join(DIR_SOURCE, "youtube_video_links_2021_2025.csv")
RAW_COMMENTS_CSV_PATH = os.path.join(DIR_RAW, "youtube_comments_raw.csv")
EXTRACTED_COMMENTS_CSV_PATH = os.path.join(DIR_EXTRACTED, "youtube_comments_extracted.csv")
CLEANED_COMMENTS_CSV_PATH = os.path.join(DIR_CLEANING, "youtube_comments_cleaned.csv")
CLEANSED_COMMENTS_CSV_PATH = os.path.join(DIR_CLEANSING, "youtube_comments_cleansed.csv")
JUDOL_DETECTED_CSV_PATH = os.path.join(DIR_CLEANSING, "judol_comments_detected.csv")


# =====================================================================
# KONFIGURASI PENCARIAN VIDEO (PERIODE 2021 - 2025)
# =====================================================================
YEARS = [2021, 2022, 2023, 2024, 2025]

BASE_KEYWORDS = [
    "erupsi gunung berapi indonesia",
    "gunung meletus indonesia",
    "erupsi gunung semeru",
    "erupsi gunung merapi",
    "erupsi gunung marapi",
    "erupsi gunung lewotobi",
    "erupsi gunung ruang",
    "erupsi gunung ibu",
    "erupsi gunung anak krakatau",
    "erupsi gunung sinabung",
    "erupsi gunung raung"
]


# =====================================================================
# TAHAP 1: SCRAPING LINK VIDEO YOUTUBE (2021-2025)
# =====================================================================
class _SilentLogger:
    """Logger senyap agar pesan ERROR/WARNING dari yt-dlp tidak membanjiri terminal."""
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


MAX_WORKERS_DETAIL = 8     # pengambilan metadata video (Tahap 1)
MAX_WORKERS_COMMENTS = 2   # pengambilan komentar (Tahap 2); kecil agar tidak kena HTTP 403
RETRY_ROUNDS = 3           # berapa kali video yang gagal dicoba ulang di akhir
ROUND_COOLDOWN = 120       # jeda (detik) sebelum setiap ronde ulang


def is_video_accessible(entry):
    """
    Cek apakah video benar-benar dapat diakses berdasarkan metadata yt-dlp:
    - Video publik (bukan private/needs_auth/premium_only/subscriber_only)
    - Bukan siaran langsung/akan tayang (live/upcoming)
    - Memiliki komentar: comment_count > 0.
    """
    availability = entry.get("availability")
    if availability in ("private", "premium_only", "subscriber_only", "needs_auth"):
        return False
    if entry.get("live_status") in ("is_live", "is_upcoming", "post_live"):
        return False
    if (entry.get("age_limit") or 0) >= 18:
        return False
    if entry.get("comment_count") == 0:
        return False
    return True


def video_has_comments(video_url):
    """Verifikasi cadangan jika comment_count tidak tersedia: cek 1 komentar via yt-dlp."""
    opts = {
        "quiet": True, "no_warnings": True, "skip_download": True,
        "getcomments": True, "logger": _SilentLogger(),
        "extractor_args": {"youtube": {"max_comments": ["1", "1", "0", "0"]}},
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
        return bool(info and info.get("comments"))
    except Exception:
        return False


def _fetch_video_detail(video_id):
    """Ambil metadata lengkap satu video (aman dipanggil dari thread)."""
    opts = {
        "quiet": True, "no_warnings": True, "ignoreerrors": True,
        "skip_download": True, "logger": _SilentLogger(),
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    except Exception:
        return None


def scrape_youtube_links(keywords=BASE_KEYWORDS, years=YEARS, max_results_per_query=20, output_csv=LINKS_CSV_PATH):
    """
    Mencari dan mengumpulkan link video YouTube berdasarkan kata kunci dan rentang tahun
    2021 hingga 2025 menggunakan yt-dlp.
    """
    if yt_dlp is None:
        raise ImportError("Library 'yt-dlp' diperlukan untuk menjalankan fungsi ini. Install dengan: pip install yt-dlp")

    print("\n" + "=" * 70)
    print("TAHAP 1: SCRAPING LINK VIDEO YOUTUBE (2021 - 2025)")
    print("=" * 70)

    search_opts = {
        "quiet": True, "no_warnings": True, "ignoreerrors": True,
        "extract_flat": True, "skip_download": True, "logger": _SilentLogger(),
    }
    candidates = {}

    with yt_dlp.YoutubeDL(search_opts) as ydl_search:
        for year in years:
            for kw in keywords:
                print(f"[*] Mencari: '{kw} {year}' (Maks: {max_results_per_query} video)...")
                try:
                    result = ydl_search.extract_info(f"ytsearch{max_results_per_query}:{kw} {year}", download=False)
                except Exception as e:
                    print(f"[!] Error saat mencari '{kw} {year}': {e}")
                    continue
                for e in (result or {}).get("entries") or []:
                    if e and e.get("id") and e["id"] not in candidates:
                        candidates[e["id"]] = (year, f"{kw} {year}")

    print(f"\n[*] {len(candidates)} kandidat video unik. Memeriksa aksesibilitas & komentar "
          f"({MAX_WORKERS_DETAIL} thread paralel)...")

    collected_videos = {}
    unavailable_count = 0
    no_comment_count = 0
    out_of_range_count = 0
    ids = list(candidates.keys())

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_DETAIL) as pool:
        futures = {pool.submit(_fetch_video_detail, vid): vid for vid in ids}
        for n, fut in enumerate(as_completed(futures), start=1):
            video_id = futures[fut]
            year, search_kw = candidates[video_id]
            entry = fut.result()

            if not entry:
                unavailable_count += 1
            elif not is_video_accessible(entry):
                no_comment_count += 1
            else:
                upload_date_str = entry.get("upload_date")
                video_year = None
                if upload_date_str and len(upload_date_str) >= 4:
                    video_year = int(upload_date_str[:4])
                elif entry.get("release_timestamp"):
                    video_year = datetime.fromtimestamp(entry["release_timestamp"]).year
                elif entry.get("timestamp"):
                    video_year = datetime.fromtimestamp(entry["timestamp"]).year
                target_year = video_year if video_year is not None else year

                video_url = f"https://www.youtube.com/watch?v={video_id}"
                if not (2021 <= target_year <= 2025):
                    out_of_range_count += 1
                elif entry.get("comment_count") is None and not video_has_comments(video_url):
                    no_comment_count += 1
                else:
                    collected_videos[video_id] = {
                        "video_id": video_id,
                        "video_url": video_url,
                        "title": entry.get("title", ""),
                        "channel": entry.get("uploader", ""),
                        "upload_date": upload_date_str if upload_date_str else f"{year}0101",
                        "year": target_year,
                        "view_count": entry.get("view_count", 0),
                        "comment_count_est": entry.get("comment_count", 0),
                        "search_keyword": search_kw,
                        "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }

            if n % 25 == 0 or n == len(ids):
                print(f"    [{n}/{len(ids)}] diperiksa | lolos: {len(collected_videos)}")

    df_links = pd.DataFrame(list(collected_videos.values()))
    if not df_links.empty:
        df_links.sort_values(by=["year", "upload_date"], ascending=[True, True], inplace=True)
        df_links.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"\n[+] Berhasil mengumpulkan {len(df_links)} link video unik (2021-2025).")
        print(f"[i] Ditolak -> tidak tersedia: {unavailable_count} | tanpa komentar/tidak publik: "
              f"{no_comment_count} | di luar 2021-2025: {out_of_range_count}")
        print(f"[+] File daftar link disimpan di: {output_csv}")
    else:
        print("\n[!] Tidak ada link video yang berhasil dikumpulkan.")

    return df_links


# =====================================================================
# TAHAP 2: EKSTRAKSI SELURUH KOMENTAR VIDEO YOUTUBE
# =====================================================================
def _fetch_video_comments(video_url, limit_per_video=None, max_retries=3):
    """Ambil komentar satu video via yt-dlp dengan retry berganda."""
    yt_args = {}
    if limit_per_video is not None:
        yt_args["max_comments"] = [str(limit_per_video), "all", "all", "all"]
    opts = {
        "quiet": True, "no_warnings": True, "skip_download": True,
        "getcomments": True, "logger": _SilentLogger(),
        "retries": 3, "extractor_retries": 3,
        "extractor_args": {"youtube": yt_args} if yt_args else {},
    }
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
            comments = (info or {}).get("comments") or []
            if limit_per_video is not None:
                comments = comments[:limit_per_video]
            return comments
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(20 * (2 ** attempt) + random.uniform(0, 5))
    raise last_error


def extract_all_comments(links_csv=LINKS_CSV_PATH, output_extracted_csv=EXTRACTED_COMMENTS_CSV_PATH,
                         output_raw_csv=RAW_COMMENTS_CSV_PATH, limit_per_video=None, resume=True):
    """Mengambil seluruh komentar dari setiap link video YouTube yang ada di CSV."""
    if yt_dlp is None:
        raise ImportError("Library 'yt-dlp' diperlukan. Install dengan: pip install yt-dlp")

    print("\n" + "=" * 70)
    print("TAHAP 2: EKSTRAKSI SELURUH KOMENTAR DARI VIDEO YOUTUBE")
    print("=" * 70)

    if not os.path.exists(links_csv):
        raise FileNotFoundError(f"File daftar link tidak ditemukan di: {links_csv}. Jalankan Tahap 1 terlebih dahulu.")

    df_links = pd.read_csv(links_csv)
    if df_links.empty:
        print("[!] File daftar link video kosong.")
        return pd.DataFrame()

    df_previous = pd.DataFrame()
    if resume and os.path.exists(output_raw_csv):
        try:
            df_previous = pd.read_csv(output_raw_csv)
            df_previous = df_previous[df_previous["video_id"].isin(df_links["video_id"])]
            done_ids = set(df_previous["video_id"].astype(str))
            if done_ids:
                print(f"[*] Resume: {len(done_ids)} video sudah ada komentarnya, dilewati.")
        except Exception:
            df_previous = pd.DataFrame()
    done_ids = set(df_previous["video_id"].astype(str)) if not df_previous.empty else set()

    all_comments = []
    rows = [row for _, row in df_links.iterrows() if str(row.get("video_id", "")).strip() not in done_ids]
    total_videos = len(rows)
    if total_videos == 0:
        print("[*] Semua video sudah terekstrak sebelumnya.")

    def worker(row):
        video_id = str(row.get("video_id", "")).strip()
        video_url = str(row.get("video_url", f"https://www.youtube.com/watch?v={video_id}")).strip()
        return row, video_id, video_url, _fetch_video_comments(video_url, limit_per_video)

    pending = rows
    processed = 0
    for round_no in range(1, RETRY_ROUNDS + 2):
        if not pending:
            break
        if round_no > 1:
            print(f"\n[*] Ronde ulang {round_no - 1}/{RETRY_ROUNDS}: {len(pending)} video gagal. "
                  f"Menunggu {ROUND_COOLDOWN}s agar rate-limit reda...")
            time.sleep(ROUND_COOLDOWN)
        workers = MAX_WORKERS_COMMENTS if round_no == 1 else 1
        failed = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(worker, r): r for r in pending}
            for fut in as_completed(futures):
                src_row = futures[fut]
                processed += 1
                try:
                    row, video_id, video_url, comments = fut.result()
                except Exception as e:
                    failed.append(src_row)
                    print(f"[{processed}] [!] Gagal ({str(e)[:70]}) video {src_row.get('video_id')} -> akan dicoba ulang")
                    continue

                video_year = row.get("year", "")
                video_title = row.get("title", "")
                for c in comments:
                    all_comments.append({
                        "comment_id": c.get("id", ""),
                        "video_id": video_id,
                        "video_url": video_url,
                        "video_title": video_title,
                        "video_year": video_year,
                        "author": c.get("author", ""),
                        "text_original": c.get("text", ""),
                        "published_time": c.get("timestamp", ""),
                        "votes": c.get("like_count", 0),
                        "heart": c.get("is_favorited", False),
                        "reply": c.get("parent", "root") != "root",
                        "extracted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
                print(f"[{processed}] {len(comments)} komentar <- {video_url} ({video_year}) - {str(video_title)[:40]}")
        pending = failed

    if pending:
        print(f"\n[!] {len(pending)} video tetap gagal (rate-limit). Jalankan ulang script nanti (resume aktif).")

    df_comments = pd.concat([df_previous, pd.DataFrame(all_comments)], ignore_index=True)

    if not df_comments.empty:
        df_comments.to_csv(output_raw_csv, index=False, encoding="utf-8-sig")
        shutil.copy(output_raw_csv, output_extracted_csv)
        print(f"\n[+] Total seluruh komentar yang berhasil diekstrak: {len(df_comments)} komentar.")
        print(f"[+] Disimpan di (Raw)      : {output_raw_csv}")
        print(f"[+] Disimpan di (Extracted): {output_extracted_csv}")
    else:
        print("\n[!] Tidak ada komentar yang berhasil diekstrak.")

    return df_comments


# =====================================================================
# TAHAP 3: CLEANING TEXT
# =====================================================================
SLANG_DICT = {
    "yg": "yang", "dgn": "dengan", "dg": "dengan", "dlm": "dalam", "utk": "untuk",
    "krn": "karena", "karna": "karena", "krna": "karena", "kalo": "kalau", "kl": "kalau",
    "klw": "kalau", "klo": "kalau", "klu": "kalau", "gk": "tidak", "ga": "tidak",
    "gak": "tidak", "nggak": "tidak", "ngga": "tidak", "tdk": "tidak", "tak": "tidak",
    "bgt": "banget", "bngt": "banget", "dr": "dari", "dri": "dari", "pd": "pada",
    "udh": "sudah", "udah": "sudah", "sdh": "sudah", "blm": "belum", "blom": "belum",
    "belom": "belum", "bisa": "bisa", "bs": "bisa", "jd": "jadi", "jdi": "jadi",
    "smg": "semoga", "smoga": "semoga", "moga": "semoga", "org": "orang", "orng": "orang",
    "gn": "gunung", "gng": "gunung", "jml": "jumlah", "thn": "tahun", "th": "tahun",
    "tgl": "tanggal", "skrg": "sekarang", "skg": "sekarang", "byk": "banyak",
    "bnyk": "banyak", "spt": "seperti", "kayak": "seperti", "kyk": "seperti",
    "tp": "tapi", "tpi": "tapi", "tetapi": "tapi", "cm": "cuma", "cma": "cuma",
    "cuman": "cuma", "aja": "saja", "aj": "saja", "sj": "saja", "sy": "saya",
    "aku": "saya", "gw": "saya", "gue": "saya", "km": "kamu", "lu": "kamu",
    "loe": "kamu", "kpd": "kepada", "trs": "terus", "trus": "terus", "emg": "memang",
    "emang": "memang", "mmg": "memang", "gimana": "bagaimana", "gmn": "bagaimana",
    "knp": "kenapa", "knapa": "kenapa", "bkn": "bukan", "sbg": "sebagai",
    "dll": "dan lain lain", "dsb": "dan sebagainya", "jg": "juga", "jga": "juga",
    "lg": "lagi", "lgi": "lagi"
}


# Pola regex emoji dikompilasi di level modul untuk efisiensi
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\u2700-\u27BF"          # dingbats
    "\u2600-\u26FF"          # misc symbols
    "\U00010000-\U0010FFFF"  # supplementary characters & modern emoji ranges
    "]+",
    flags=re.UNICODE,
)


def remove_emojis(text):
    """Menghapus seluruh karakter emoji dan simbol piktograf dari teks."""
    return EMOJI_PATTERN.sub(r" ", text)


def clean_single_text(text, normalize_slang=True):
    """
    Membersihkan teks komentar pada tahap CLEANING:
    - Menghapus URL, Mention, Hashtag, Emoji
    - Case folding ke lowercase
    - Menghapus tanda baca, simbol, timestamp
    - Mengurangi huruf berulang berlebih
    - Normalisasi kata slang dasar
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # 1. Hapus newline, tab, dan karakter kontrol
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")

    # 2. Hapus URL / Link
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Hapus Mention (@username) dan Hashtag (#topik)
    text = re.sub(r"@[A-Za-z0-9_.-]+", " ", text)
    text = re.sub(r"#[A-Za-z0-9_]+", " ", text)

    # 4. Hapus Emoji
    text = remove_emojis(text)

    # 5. Case Folding
    text = text.lower()

    # 6. Hapus timestamp (01:23) dan karakter tanda baca/simbol selain alfabet dan angka
    text = re.sub(r"(?<!\S)\d{1,2}:\d{2}(?::\d{2})?(?!\S)", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 7. Kurangi huruf berulang lebih dari 2 kali berurutan (contoh: "seremmm" -> "seremm")
    text = re.sub(r"([a-z])\1{2,}", r"\1\1", text)

    # 8. Normalisasi kata slang dasar
    if normalize_slang:
        words = text.split()
        normalized_words = [SLANG_DICT.get(w, w) for w in words]
        text = " ".join(normalized_words)

    # 9. Hapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_comments_dataset(input_csv=EXTRACTED_COMMENTS_CSV_PATH, output_csv=CLEANED_COMMENTS_CSV_PATH):
    """Memproses dataset komentar hasil ekstraksi untuk tahap CLEANING."""
    print("\n" + "=" * 70)
    print("TAHAP 3: CLEANING TEXT")
    print("=" * 70)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"File komentar tidak ditemukan di: {input_csv}. Jalankan Tahap 2 terlebih dahulu.")

    df = pd.read_csv(input_csv)
    initial_count = len(df)
    print(f"[*] Jumlah total komentar awal: {initial_count} baris.")

    if df.empty:
        print("[!] Dataset komentar kosong.")
        return df

    # 1. Hapus missing values
    df.dropna(subset=["text_original"], inplace=True)
    after_na_count = len(df)
    print(f"[*] Setelah hapus missing values (NaN): {after_na_count} baris (Dihapus: {initial_count - after_na_count}).")

    # 2. Hapus duplikat
    if "comment_id" in df.columns and df["comment_id"].notna().any():
        df.drop_duplicates(subset=["comment_id"], keep="first", inplace=True)
    df.drop_duplicates(subset=["video_id", "text_original"], keep="first", inplace=True)
    after_dup_count = len(df)
    print(f"[*] Setelah hapus duplikat komentar: {after_dup_count} baris (Dihapus: {after_na_count - after_dup_count}).")

    # 3. Proses Cleaning Text
    print("[*] Sedang membersihkan teks komentar (emoji, URL, simbol, case folding)...")
    df["text_cleaned"] = df["text_original"].astype(str).apply(clean_single_text)

    # 4. Hapus baris yang kosong pasca-cleaning
    df = df[df["text_cleaned"].str.strip() != ""]
    df.drop_duplicates(subset=["video_id", "text_cleaned"], keep="first", inplace=True)
    final_count = len(df)
    print(f"[*] Setelah hapus komentar kosong & redundan: {final_count} baris.")

    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n[+] Tahap Cleaning Text selesai! File disimpan di: {output_csv}")

    return df


# =====================================================================
# TAHAP 4: CLEANSING DATA (TYPO CORRECTION & ELIMINASI JUDOL)
# =====================================================================

# Regex Pola Komentar Judi Online (Judol / Slot / Spam Gambling)
JUDOL_STRICT_PATTERNS = [
    # Brand situs & provider judi online
    r'\b(slot88|slot777?|988slot|squad777|yuk69|sawer4d|berkah99|joker123)\b',
    r'\b(pragmatic\s*play|habanero|spadegaming|pg\s*soft|microgaming)\b',
    r'\b(kakek\s*zeus|gates\s*of\s*olympus|sweet\s*bonanza|mahjong\s*ways)\b',
    # Istilah spesifik slot/judi online
    r'\b(maxwin|sensational|scatter\s*hitam|scatter\s*naga|anti\s*rungkad|anti\s*rungkat)\b',
    r'\b(pola\s*gacor|bocoran\s*admin|bocoran\s*rtp|rtp\s*live|rtp\s*slot)\b',
    r'\b(bo\s*terpercaya|bo\s*gacor|akun\s*pro|freebet|freechip)\b',
    r'\b(jackpot\s*paus|jp\s*paus)\b',
    r'\b(toto\s*macau|togel\s*online|situs\s*toto|live\s*casino)\b',
    # Frasa transaksi dan tautan judol
    r'\b(slot|judi|judol|togel)\s*(online|gacor|resmi|terpercaya|hari\s*ini)\b',
    r'\b(situs|link|daftar|agen|bandar|main|bermain|game)\s*(slot|judi|judol|togel)\b',
    r'\b(slot|judi|judol)\s*(receh|pulsa|dana|gampang|mudah)\b',
    r'\b(depo|deposit)\s*(pulsa|receh|dana|wd|min|10k|20k|25k|50k)\b',
    r'\b(depo\s*wd|wd\s*lancar|wd\s*cepat|proses\s*wd)\b',
    r'\b(judi\s*online|judi\s*slot)\b',
    r'\b\d{2,4}\s*slot\b',
    r'\bslot\s*\d{2,4}\b',
    # Promosi & keluhan spam judol
    r'\b(bagi2?\s*(maxwin|freebet|saldo)|promo\s*judol|spam\s*judol|admin\s*judol|komentar\s*judol|akun\s*judol)\b',
    r'\bjudol\b',
    r'\b(situs\s*legend\s*tapi|web\s*legend\s*tapi).*squad777\b',
]

RE_JUDOL_STRICT = re.compile('|'.join(JUDOL_STRICT_PATTERNS), re.IGNORECASE)


def detect_judol(row):
    """
    Mendeteksi apakah sebuah komentar merupakan promosi judi online (judol/slot)
    atau noise spam judol dengan penanganan false positive:
    - Mengecualikan 'depo pertamina'
    - Mengecualikan kendaraan 'mobil 4 wd' (Four Wheel Drive untuk evakuasi gunung)
    - Mengecualikan nama 'dewa zeus' dalam konteks mitologi Yunani
    """
    t_clean = str(row.get("text_cleaned", ""))
    t_orig = str(row.get("text_original", ""))
    author = str(row.get("author", ""))

    is_matched = bool(RE_JUDOL_STRICT.search(t_clean) or RE_JUDOL_STRICT.search(t_orig))

    if is_matched:
        # Pengecekan pengecualian konteks aman (hanya jika kecocokan disebabkan oleh istilah non-judi):
        stripped = t_clean
        # Jika mengandung depo pertamina, hilangkan frasa tersebut dan uji apakah masih ada pola judol lain
        if re.search(r"\bdepo\s+pertamina\b", stripped):
            stripped = re.sub(r"\bdepo\s+pertamina\b", " ", stripped)
        # Jika mengandung 4 wd / 4wd, hilangkan frasa tersebut dan uji apakah masih ada pola judol lain
        if re.search(r"\b(4\s*wd|4wd)\b", stripped):
            stripped = re.sub(r"\b(4\s*wd|4wd)\b", " ", stripped)
        # Jika mitologi yunani dewa zeus, hilangkan frasa tersebut
        if "mitologi yunani" in stripped and re.search(r"\bdewa\s+zeus\b", stripped):
            stripped = re.sub(r"\bdewa\s+zeus\b", " ", stripped)

        # Jika setelah frasa aman dihilangkan tidak ada lagi pola judol, maka ini BUKAN judol
        if not RE_JUDOL_STRICT.search(stripped):
            return False
        return True

    # Deteksi akun author bot judol spesifik
    if re.search(r"\b(slot|gacor|maxwin|judol)\b", author, re.IGNORECASE):
        if re.search(r"\b(slot|gacor|depo|wd|sensational|zeus|olympus)\b", t_clean):
            return True

    return False


# Kamus normalisasi Typo & Slang Lanjutan untuk Tahap Cleansing
CLEANSING_TYPO_DICT = {
    # Singkatan kata & bahasa percakapan
    "klo": "kalau", "klu": "kalau", "kalo": "kalau", "kl": "kalau", "klw": "kalau", "klow": "kalau",
    "yg": "yang", "dgn": "dengan", "dg": "dengan", "dlm": "dalam", "utk": "untuk", "untk": "untuk",
    "krn": "karena", "karna": "karena", "krna": "karena", "bgt": "banget", "bngt": "banget",
    "bgtt": "banget", "bangett": "banget", "udh": "sudah", "udah": "sudah", "sdh": "sudah",
    "blm": "belum", "blom": "belum", "belom": "belum", "lom": "belum",
    "skrg": "sekarang", "skg": "sekarang", "skrng": "sekarang", "bnyk": "banyak", "byk": "banyak",
    "tp": "tapi", "tpi": "tapi", "sy": "saya", "org": "orang", "orng": "orang",
    "smg": "semoga", "smoga": "semoga", "moga": "semoga", "dapet": "dapat", "dpt": "dapat",
    "nemu": "menemukan", "nemuin": "menemukan", "tau": "tahu", "tw": "tahu", "taw": "tahu",
    "taunya": "tahunya", "liat": "lihat", "liat2": "lihat lihat", "ngeliat": "melihat",
    "ngeliatnya": "melihatnya", "sampe": "sampai", "ampe": "sampai", "sampek": "sampai",
    "smpai": "sampai", "sampe2": "sampai sampai", "smpe": "sampai", "bener": "benar",
    "beneran": "benar", "benernya": "sebenarnya", "bnr": "benar", "tetep": "tetap", "ttep": "tetap",
    "bikin": "membuat", "inget": "ingat", "ingetin": "mengingatkan", "ngingetin": "mengingatkan",
    "ngomong": "bicara", "ngomongin": "membicarakan", "ngerasain": "merasakan", "ngerasa": "merasa",
    "ngasih": "memberi", "makasih": "terima kasih", "makasi": "terima kasih", "trims": "terima kasih",
    "trimakasih": "terima kasih", "tengkyu": "terima kasih", "thanks": "terima kasih", "thx": "terima kasih",
    "kasian": "kasihan", "kesian": "kasihan", "slamat": "selamat", "selamet": "selamat",
    "jg": "juga", "jga": "juga", "lg": "lagi", "lgi": "lagi",
    "gtu": "begitu", "gitu": "begitu", "bgitu": "begitu", "bgtu": "begitu",
    "gini": "begini", "bgini": "begini", "bgni": "begini",
    "kek": "seperti", "kyk": "seperti", "kayak": "seperti", "spt": "seperti",
    "pdhal": "padahal", "pdhl": "padahal", "cuma": "cuma", "cuman": "cuma", "cma": "cuma", "cm": "cuma",
    "aja": "saja", "aj": "saja", "sj": "saja", "gw": "saya", "gue": "saya", "gua": "saya", "aq": "saya",
    "lu": "kamu", "loe": "kamu", "elo": "kamu", "kmu": "kamu", "klian": "kalian", "mrk": "mereka",
    "trs": "terus", "trus": "terus", "emg": "memang", "emang": "memang", "mmg": "memang",
    "gmn": "bagaimana", "gimana": "bagaimana", "knp": "kenapa", "knapa": "kenapa", "kenap": "kenapa",
    "sbg": "sebagai", "dll": "dan lain lain", "dsb": "dan sebagainya", "dr": "dari", "dri": "dari",
    "pd": "pada", "bs": "bisa", "jd": "jadi", "jdi": "jadi", "thn": "tahun", "th": "tahun",
    "tgl": "tanggal", "jml": "jumlah", "gn": "gunung", "gng": "gunung", "cepet": "cepat", "cpet": "cepat",
    "cepetan": "cepat", "denger": "dengar", "dengerin": "mendengarkan", "nonton": "menonton",
    "ngungsi": "mengungsi", "nunggu": "menunggu", "ngerti": "mengerti", "mikir": "berpikir",
    "bayangin": "membayangkan", "nyari": "mencari", "nyampe": "sampai", "nyelamatin": "menyelamatkan",
    "bantuin": "membantu", "ngalamin": "mengalami", "mngalami": "mengalami", "ngajak": "mengajak",
    "ngambil": "mengambil", "ngejelasin": "menjelaskan", "ngeluarin": "mengeluarkan",
    "temen": "teman", "tmn": "teman", "mimin": "admin", "brp": "berapa", "brapa": "berapa",
    "dmn": "di mana", "dmana": "di mana", "kmn": "ke mana", "kmana": "ke mana", "biasaya": "biasanya",
    "suwatu": "suatu", "bsok": "besok", "bsoq": "besok", "sndr": "sendiri", "mletus": "meletus",
    "bhkan": "bahkan", "introfeksi": "introspeksi", "msh": "masih", "msih": "masih",
    "lbh": "lebih", "lbih": "lebih",

    # Negasi & Frasa Majemuk
    "gak": "tidak", "ga": "tidak", "gk": "tidak", "nggak": "tidak", "ngga": "tidak", "tdk": "tidak",
    "tak": "tidak", "ndak": "tidak", "engga": "tidak", "enggak": "tidak", "enggk": "tidak",
    "gatau": "tidak tahu", "gataw": "tidak tahu", "gabisa": "tidak bisa", "gamau": "tidak mau",
    "gapapa": "tidak apa apa", "gapap": "tidak apa apa", "gapopo": "tidak apa apa", "rapopo": "tidak apa apa",
    "gpp": "tidak apa apa", "gamungkin": "tidak mungkin", "gada": "tidak ada", "gaada": "tidak ada",
    "gakada": "tidak ada", "gadak": "tidak ada", "gasampe": "tidak sampai", "bisa2nya": "bisa bisanya",

    # Doa, Religi, & Bencana
    "amiiin": "amin", "amiin": "amin", "aamiin": "amin", "aminn": "amin", "aamiinn": "amin",
    "alhamdulilah": "alhamdulillah", "alhamdullilah": "alhamdulillah",
    "subhanalloh": "subhanallah", "astaghfirullah": "astagfirullah",
    "astagfirullahaladzim": "astagfirullah", "astaghfirullahalazim": "astagfirullah",
    "inalillahi": "innalillahi", "innalilahi": "innalillahi", "innalillahiwainnailaihirojiun": "innalillahi",
    "yaallah": "ya allah", "yaalloh": "ya allah", "yarabb": "ya rabb", "yarobb": "ya rabb",
    "bismilah": "bismillah", "insyaallah": "insya allah", "inshaallah": "insya allah",
}

# Kamus Reduplikasi Numerik (Kata Ulang dengan angka 2)
NUMERIC_REDUPLICATIONS = {
    "orang2": "orang orang", "org2": "orang orang", "teman2": "teman teman", "temen2": "teman teman",
    "tmn2": "teman teman", "saudara2": "saudara saudara", "bener2": "benar benar", "benar2": "benar benar",
    "hati2": "hati hati", "tiba2": "tiba tiba", "tanda2": "tanda tanda", "sehat2": "sehat sehat",
    "baik2": "baik baik", "gunung2": "gunung gunung", "korban2": "korban korban", "anak2": "anak anak",
    "kira2": "kira kira", "masing2": "masing masing", "gara2": "gara gara", "gr2": "gara gara",
    "apa2": "apa apa", "sama2": "sama sama", "kata2": "kata kata", "moga2": "moga moga", "hal2": "hal hal",
    "kawan2": "kawan kawan", "abang2": "abang abang", "laki2": "laki laki", "dosa2": "dosa dosa",
    "cerita2": "cerita cerita", "bapak2": "bapak bapak", "ibu2": "ibu ibu", "rumah2": "rumah rumah",
    "warga2": "warga warga", "desa2": "desa desa", "daerah2": "daerah daerah", "batu2": "batu batu",
    "debu2": "debu debu", "lahar2": "lahar lahar", "asap2": "asap asap", "abu2": "abu abu",
    "was2": "was was", "mati2an": "mati matian", "pura2": "pura pura", "takut2": "takut takut",
    "mudah2an": "mudah mudahan", "lagi2": "lagi lagi", "malam2": "malam malam", "pagi2": "pagi pagi",
    "siang2": "siang siang", "sore2": "sore sore", "pelan2": "pelan pelan", "rame2": "ramai ramai",
    "ramai2": "ramai ramai", "diam2": "diam diam", "pelan2": "pelan pelan", "habis2an": "habis habisan",
    "gedung2": "gedung gedung", "pohon2": "pohon pohon", "tanah2": "tanah tanah"
}

RE_ELONGATED_CLEANSE = re.compile(r"([a-z])\1{2,}")
RE_DOUBLE_CONS_END = re.compile(r"([b-df-hj-np-tv-z])\1$")


def cleanse_single_text(text):
    """
    Menormalkan satu teks komentar pada tahap CLEANSING:
    - Reduksi huruf berulang berlebihan (contoh: 'waaaaspada' -> 'waspada')
    - Ekspansi kata ulang numerik (contoh: 'orang2' -> 'orang orang')
    - Normalisasi konsonan ganda di akhir (contoh: 'bangett' -> 'banget')
    - Koreksi typo dan singkatan slang ke kata baku / makna aslinya
    - Normalisasi ungkapan doa / empati bencana
    """
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return ""

    # 1. Reduksi huruf berulang lebih dari 2 kali (misal: "amiiinnn" -> "amin", "parahhhh" -> "parah")
    text = RE_ELONGATED_CLEANSE.sub(r"\1", text)

    words = text.split()
    cleansed_words = []
    valid_double_cons = {"all", "well", "mall", "full", "call", "still", "will"}

    for word in words:
        # Cek kata ulang angka 2 spesifik
        if word in NUMERIC_REDUPLICATIONS:
            cleansed_words.append(NUMERIC_REDUPLICATIONS[word])
            continue

        # Cek kamus typo & slang dengan bentuk asli kata terlebih dahulu
        if word in CLEANSING_TYPO_DICT:
            cleansed_words.append(CLEANSING_TYPO_DICT[word])
            continue

        # Cek konsonan ganda di akhir kata jika bukan kata serapan/baku
        if len(word) > 3 and word not in valid_double_cons and RE_DOUBLE_CONS_END.search(word):
            trimmed = word[:-1]
            cleansed_words.append(CLEANSING_TYPO_DICT.get(trimmed, trimmed))
            continue

        cleansed_words.append(word)

    result = " ".join(cleansed_words)
    result = re.sub(r"\s+", " ", result).strip()
    return result


def cleanse_comments_dataset(input_csv=CLEANED_COMMENTS_CSV_PATH, output_csv=CLEANSED_COMMENTS_CSV_PATH,
                             save_judol_log=True, judol_log_csv=JUDOL_DETECTED_CSV_PATH):
    """
    Memproses dataset dari hasil CLEANING menuju tahap CLEANSING:
    1. Deteksi komentar promosi & spam judi online (judol)
    2. Menghapus komentar judol dari dataset (opsional: simpan log ke judol_comments_detected.csv)
    3. Normalisasi typo, singkatan gaul/slang, reduplikasi numerik, dan ekspresi bencana
    4. Menyaring komentar yang terlalu pendek (< 2 karakter) atau kosong pasca-cleansing
    5. Deduplikasi teks bersih akhir per video
    6. Menyimpan dataset bersih final ke data/cleansing/
    """
    print("\n" + "=" * 70)
    print("TAHAP 4: CLEANSING DATA (TYPO CORRECTION & ELIMINASI JUDOL)")
    print("=" * 70)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(
            f"File input cleaning tidak ditemukan di: {input_csv}.\n"
            f"Silakan jalankan Tahap 3 (Cleaning Text) terlebih dahulu."
        )

    print(f"[*] Membaca data hasil cleaning dari: {input_csv}")
    df = pd.read_csv(input_csv)
    initial_count = len(df)
    print(f"[*] Jumlah total komentar awal: {initial_count} baris.")

    if df.empty:
        print("[!] Dataset komentar kosong.")
        return df

    # Langkah 1: Deteksi Komentar Judol
    print("[*] Sedang mendeteksi komentar spam / promosi judi online (judol)...")
    judol_mask = df.apply(detect_judol, axis=1)
    judol_count = int(judol_mask.sum())
    print(f"[*] Terdeteksi {judol_count} komentar judol ({judol_count / initial_count * 100:.2f}% dari dataset).")

    if judol_count > 0 and save_judol_log:
        df_judol = df[judol_mask].copy()
        df_judol["flagged_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df_judol.to_csv(judol_log_csv, index=False, encoding="utf-8-sig")
        print(f"[+] Daftar komentar judol yang terdeteksi disimpan di: {judol_log_csv}")

    # Langkah 2: Hapus Komentar Judol
    df_clean = df[~judol_mask].copy()
    print(f"[*] Jumlah komentar setelah menghapus judol: {len(df_clean)} baris.")

    if df_clean.empty:
        print("[!] Tidak ada komentar tersisa setelah pembersihan judol.")
        df_clean["text_cleansed"] = pd.Series(dtype=str)
        df_clean.to_csv(output_csv, index=False, encoding="utf-8-sig")
        return df_clean

    # Langkah 3: Perbaikan Typo & Slang (Cleansing)
    print("[*] Sedang melakukan perbaikan typo, slang, reduplikasi kata, dan normalisasi teks...")
    col_input = "text_cleaned" if "text_cleaned" in df_clean.columns else "text_original"
    df_clean["text_cleansed"] = df_clean[col_input].astype(str).apply(cleanse_single_text).astype(str)

    # Hitung jumlah baris yang mengalami perbaikan typo/slang
    diff_mask = df_clean[col_input] != df_clean["text_cleansed"]
    modified_count = int(diff_mask.sum())
    pct = (modified_count / len(df_clean) * 100) if len(df_clean) else 0.0
    print(f"[*] Berhasil memperbaiki typo/slang pada {modified_count} komentar ({pct:.1f}%).")

    # Langkah 4: Filter Komentar Kosong / Terlalu Pendek (< 2 karakter)
    df_clean = df_clean[df_clean["text_cleansed"].str.strip().str.len() > 1]

    # Langkah 5: Hapus duplikat pasca-cleansing pada video yang sama
    df_clean.drop_duplicates(subset=["video_id", "text_cleansed"], keep="first", inplace=True)
    final_count = len(df_clean)

    # Langkah 6: Simpan Hasil Akhir ke Folder data/cleansing
    df_clean.reset_index(drop=True, inplace=True)
    df_clean.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("\n" + "-" * 70)
    print("RINGKASAN HASIL TAHAP 4 (CLEANSING DATA):")
    print(f"1. Total Komentar Masuk (dari Cleaning) : {initial_count:,} baris")
    print(f"2. Komentar Judol / Spam Tereliminasi   : {judol_count:,} baris")
    print(f"3. Komentar yang Dinormalisasi (Typo)   : {modified_count:,} baris")
    print(f"4. Total Komentar Bersih Akhir          : {final_count:,} baris")
    print(f"[+] File Hasil Cleansing disimpan di    : {output_csv}")
    print("-" * 70)

    return df_clean


# =====================================================================
# ORKESTRASI PIPELINE (PILIHAN MENJALANKAN BAGIAN MANA)
# =====================================================================
def run_pipeline(steps=None, max_videos_per_query=15, comment_limit_per_video=None,
                 save_judol_log=True, skip_links=False):
    """
    Menjalankan alur Data Preparation sesuai tahapan yang dipilih:
    - steps = '1' atau 'scrape'        : Scraping link video
    - steps = '2' atau 'extract'       : Ekstraksi komentar
    - steps = '3' atau 'clean'         : Cleaning text dasar
    - steps = '4' atau 'cleanse'       : Cleansing data (typo & judol)
    - steps = '3,4'                    : Cleaning + Cleansing
    - steps = 'all'                    : Seluruh alur (1 -> 2 -> 3 -> 4)
    """
    if not steps:
        steps = "all"

    steps = str(steps).strip().lower()
    steps = ",".join(p.strip() for p in steps.split(","))
    valid_steps = {
        "1", "scrape", "links",
        "2", "extract", "comments",
        "3", "clean", "cleaning",
        "4", "cleanse", "cleansing",
        "3,4", "clean_and_cleanse",
        "all"
    }
    if steps not in valid_steps:
        raise ValueError(
            f"Tahapan tidak dikenal: '{steps}'. Pilihan valid: {', '.join(sorted(valid_steps))}"
        )

    print("\n" + "=" * 75)
    print("  DATA PREPARATION PIPELINE: VOLCANO SENTIMENT ANALYSIS INDONESIA (2021-2025)")
    print(f"  Mode Operasi: {steps.upper()}")
    print("=" * 75)

    # TAHAP 1
    if steps in ["1", "scrape", "links", "all"]:
        if skip_links and os.path.exists(LINKS_CSV_PATH):
            print("[*] Skip links aktif: menggunakan file daftar link yang sudah ada.")
        else:
            scrape_youtube_links(
                keywords=BASE_KEYWORDS,
                years=YEARS,
                max_results_per_query=max_videos_per_query,
                output_csv=LINKS_CSV_PATH
            )

    # TAHAP 2
    if steps in ["2", "extract", "comments", "all"]:
        extract_all_comments(
            links_csv=LINKS_CSV_PATH,
            output_extracted_csv=EXTRACTED_COMMENTS_CSV_PATH,
            output_raw_csv=RAW_COMMENTS_CSV_PATH,
            limit_per_video=comment_limit_per_video
        )

    # TAHAP 3
    if steps in ["3", "clean", "cleaning", "3,4", "clean_and_cleanse", "all"]:
        clean_comments_dataset(
            input_csv=EXTRACTED_COMMENTS_CSV_PATH,
            output_csv=CLEANED_COMMENTS_CSV_PATH
        )

    # TAHAP 4
    if steps in ["4", "cleanse", "cleansing", "3,4", "clean_and_cleanse", "all"]:
        cleanse_comments_dataset(
            input_csv=CLEANED_COMMENTS_CSV_PATH,
            output_csv=CLEANSED_COMMENTS_CSV_PATH,
            save_judol_log=save_judol_log,
            judol_log_csv=JUDOL_DETECTED_CSV_PATH
        )

    print("\n" + "=" * 75)
    print("  EKSEKUSI TAHAPAN DATA PREPARATION BERHASIL SELESAI!")
    print("=" * 75)
    print(f"  [1] Source Links       : {LINKS_CSV_PATH}")
    print(f"  [2] Extracted Comments : {EXTRACTED_COMMENTS_CSV_PATH}")
    print(f"  [3] Cleaned Comments   : {CLEANED_COMMENTS_CSV_PATH}")
    print(f"  [4] Cleansed Comments  : {CLEANSED_COMMENTS_CSV_PATH}")
    print("=" * 75 + "\n")


def interactive_menu():
    """Menampilkan menu konsol interaktif untuk memilih tahapan yang ingin dijalankan."""
    while True:
        print("\n" + "=" * 75)
        print("  PILIH TAHAPAN DATA PREPARATION SENTIMEN ERUPSI GUNUNG (2021-2025)")
        print("=" * 75)
        print("  [1] Tahap 1: Scraping Daftar Link Video YouTube 2021-2025 (yt-dlp)")
        print("      -> Output: data/source/youtube_video_links_2021_2025.csv")
        print("  [2] Tahap 2: Ekstraksi Seluruh Komentar dari Daftar Link Video")
        print("      -> Output: data/extracted/youtube_comments_extracted.csv")
        print("  [3] Tahap 3: Cleaning Text Komentar (Emoji, URL, Tanda Baca, dsb.)")
        print("      -> Output: data/cleaning/youtube_comments_cleaned.csv")
        print("  [4] Tahap 4: Cleansing Data (Perbaikan Typo, Slang, & Hapus Komentar Judol)")
        print("      -> Input : data/cleaning/youtube_comments_cleaned.csv")
        print("      -> Output: data/cleansing/youtube_comments_cleansed.csv")
        print("  [5] Jalankan Tahap 3 & 4 (Cleaning + Cleansing Berurutan)")
        print("  [6] Jalankan Seluruh Pipeline (Tahap 1 -> 2 -> 3 -> 4)")
        print("  [0] Keluar")
        print("-" * 75)

        try:
            choice = input("Masukkan nomor pilihan Anda [0-6]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKeluar dari program.")
            break

        if choice == "1":
            run_pipeline(steps="1")
            break
        elif choice == "2":
            run_pipeline(steps="2")
            break
        elif choice == "3":
            run_pipeline(steps="3")
            break
        elif choice == "4":
            run_pipeline(steps="4")
            break
        elif choice == "5":
            run_pipeline(steps="3,4")
            break
        elif choice == "6":
            run_pipeline(steps="all")
            break
        elif choice == "0":
            print("[*] Selesai. Keluar.")
            break
        else:
            print("[!] Pilihan tidak valid. Silakan masukkan angka antara 0 hingga 6.")


# =====================================================================
# ENTRY POINT DENGAN DUKUNGAN CLI ARGUMENTS
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Data Preparation: Volcano Sentiment Analysis Indonesia (2021 - 2025)"
    )
    parser.add_argument(
        "--step", "-s",
        type=str,
        default=None,
        help="Pilih tahapan: '1' (scrape), '2' (extract), '3' (clean), '4' (cleanse), '3,4', atau 'all'"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=15,
        help="Maksimal video yang dicari per kata kunci per tahun pada Tahap 1 (default: 15)"
    )
    parser.add_argument(
        "--comment-limit",
        type=int,
        default=None,
        help="Batas jumlah komentar yang diambil per video pada Tahap 2 (default: None = seluruh komentar)"
    )
    parser.add_argument(
        "--skip-links",
        action="store_true",
        help="Gunakan daftar link video yang sudah ada (tidak melakukan scraping link ulang)"
    )
    parser.add_argument(
        "--no-judol-log",
        action="store_true",
        help="Jangan simpan file log terpisah untuk komentar judol yang terdeteksi"
    )

    args = parser.parse_args()

    # Jika ada argumen CLI --step, langsung jalankan tanpa prompt interaktif
    if args.step:
        run_pipeline(
            steps=args.step,
            max_videos_per_query=args.max_videos,
            comment_limit_per_video=args.comment_limit,
            save_judol_log=not args.no_judol_log,
            skip_links=args.skip_links
        )
    # Jika dijalankan interaktif di terminal tanpa argumen:
    elif sys.stdin.isatty():
        interactive_menu()
    # Jika non-interaktif (piped/cron) tanpa argumen:
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
