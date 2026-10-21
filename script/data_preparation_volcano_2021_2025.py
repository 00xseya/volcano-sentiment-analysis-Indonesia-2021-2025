"""
Data Preparation Pipeline: Volcano Sentiment Analysis Indonesia (2021 - 2025)
=============================================================================
Script ini untuk tahap DATA PREPARATION:
1. DATA SOURCE: Scraping daftar link video YouTube (2021-2025) menggunakan yt-dlp
   -> Disimpan ke: data/source/youtube_video_links_2021_2025.csv
2. EXTRACT DATA INTO CSV FILE: Mengambil seluruh komentar dari daftar link video menggunakan youtube-comment-downloader
   -> Disimpan ke: data/extracted/youtube_comments_extracted.csv (dan backup di data/raw/)
3. CLEANING_TEXT: Membersihkan teks komentar dari error langsung, emoji, URL, tanda baca,
   slang/typo umum, missing values, dan duplikasi data.
   -> Disimpan ke: data/cleaning/youtube_comments_cleaned.csv

Persyaratan Library (jalankan di terminal):
pip install -U yt-dlp pandas emoji
"""

import os
import sys
import re
import time
import random
import shutil
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
# Jika script ditaruh di dalam folder 'script/', naik satu level ke root proyek
if os.path.basename(BASE_DIR) == "script":
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
else:
    PROJECT_ROOT = BASE_DIR

DIR_SOURCE = os.path.join(PROJECT_ROOT, "data", "source")
DIR_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DIR_EXTRACTED = os.path.join(PROJECT_ROOT, "data", "extracted")
DIR_CLEANING = os.path.join(PROJECT_ROOT, "data", "cleaning")

for folder in [DIR_SOURCE, DIR_RAW, DIR_EXTRACTED, DIR_CLEANING]:
    os.makedirs(folder, exist_ok=True)

LINKS_CSV_PATH = os.path.join(DIR_SOURCE, "youtube_video_links_2021_2025.csv")
RAW_COMMENTS_CSV_PATH = os.path.join(DIR_RAW, "youtube_comments_raw.csv")
EXTRACTED_COMMENTS_CSV_PATH = os.path.join(DIR_EXTRACTED, "youtube_comments_extracted.csv")
CLEANED_COMMENTS_CSV_PATH = os.path.join(DIR_CLEANING, "youtube_comments_cleaned.csv")


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


# Jumlah thread paralel (naikkan jika koneksi stabil, turunkan jika kena rate-limit)
MAX_WORKERS_DETAIL = 8     # pengambilan metadata video (Tahap 1)
MAX_WORKERS_COMMENTS = 2   # pengambilan komentar (Tahap 2); kecil agar tidak kena HTTP 403
RETRY_ROUNDS = 3           # berapa kali video yang gagal dicoba ulang di akhir
ROUND_COOLDOWN = 120       # jeda (detik) sebelum setiap ronde ulang


def is_video_accessible(entry):
    """
    Cek apakah video benar-benar dapat diakses berdasarkan metadata yt-dlp:
    - Video publik (bukan private/needs_auth/premium_only/subscriber_only)
    - Bukan siaran langsung/akan tayang (live/upcoming)
    - Memiliki komentar: comment_count > 0 (komentar nonaktif/kosong ditolak).
      Jika comment_count tidak tersedia (None), dianggap belum diketahui dan
      diverifikasi terpisah lewat yt-dlp.
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
    """
    Verifikasi cadangan (dipakai hanya jika comment_count tidak tersedia):
    ambil 1 komentar via yt-dlp. Mengembalikan True jika ada minimal 1 komentar.
    """
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
    """Ambil metadata lengkap satu video (aman dipanggil dari thread). None jika tidak tersedia."""
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
    2021 hingga 2025 menggunakan yt-dlp. Hanya video yang dapat diakses dan memiliki
    komentar yang disimpan. Metadata diambil secara paralel agar lebih cepat.
    """
    if yt_dlp is None:
        raise ImportError("Library 'yt-dlp' diperlukan untuk menjalankan fungsi ini. Install dengan: pip install yt-dlp")

    print("=" * 70)
    print("TAHAP 1: SCRAPING LINK VIDEO YOUTUBE (2021 - 2025)")
    print("=" * 70)

    # --- Langkah A: kumpulkan kandidat ID dari hasil pencarian (mode flat, cepat) ---
    search_opts = {
        "quiet": True, "no_warnings": True, "ignoreerrors": True,
        "extract_flat": True, "skip_download": True, "logger": _SilentLogger(),
    }
    candidates = {}  # video_id -> (tahun query, keyword)

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

    # --- Langkah B: ambil metadata per video secara paralel & saring ---
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
                upload_date_str = entry.get("upload_date")  # Format: YYYYMMDD
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
    """
    Ambil komentar satu video via yt-dlp (aman dipanggil dari thread).
    Jika terkena rate-limit YouTube (HTTP 403/429), coba ulang dengan jeda yang makin lama.
    """
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
                time.sleep(20 * (2 ** attempt) + random.uniform(0, 5))  # 20s, 40s, 80s
    raise last_error


def extract_all_comments(links_csv=LINKS_CSV_PATH, output_extracted_csv=EXTRACTED_COMMENTS_CSV_PATH, output_raw_csv=RAW_COMMENTS_CSV_PATH, limit_per_video=None, resume=True):
    """
    Mengambil seluruh komentar dari setiap link video YouTube yang ada di file CSV
    menggunakan yt-dlp (paralel). Library youtube-comment-downloader tidak dipakai lagi
    karena sering mengembalikan 0 komentar akibat perubahan YouTube.
    Set limit_per_video=None untuk mengambil SELURUH komentar di video tersebut.
    """
    if yt_dlp is None:
        raise ImportError("Library 'yt-dlp' diperlukan. Install dengan: pip install yt-dlp")

    print("\n" + "=" * 70)
    print("TAHAP 2: EKSTRAKSI SELURUH KOMENTAR DARI VIDEO YOUTUBE")
    print("=" * 70)

    if not os.path.exists(links_csv):
        raise FileNotFoundError(f"File daftar link tidak ditemukan di: {links_csv}")

    df_links = pd.read_csv(links_csv)
    if df_links.empty:
        print("[!] File daftar link video kosong.")
        return pd.DataFrame()

    # Mode resume: lewati video yang komentarnya sudah tersimpan dari proses sebelumnya
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
        print("[*] Semua video sudah terekstrak.")

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
        print(f"\n[!] {len(pending)} video tetap gagal (rate-limit). Jalankan ulang script nanti; "
              f"video yang sudah berhasil akan dilewati (resume).")

    df_comments = pd.concat([df_previous, pd.DataFrame(all_comments)], ignore_index=True)

    if not df_comments.empty:
        # Simpan ke folder data/raw dan data/extracted sesuai struktur repositori
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
# Kamus normalisasi slang / singkatan umum Bahasa Indonesia di komentar YouTube
SLANG_DICT = {
    "yg": "yang", "dgn": "dengan", "dlm": "dalam", "utk": "untuk", "krn": "karena",
    "karna": "karena", "kalo": "kalau", "kl": "kalau", "klw": "kalau", "gk": "tidak",
    "ga": "tidak", "gak": "tidak", "nggak": "tidak", "ngga": "tidak", "tdk": "tidak",
    "tak": "tidak", "bgt": "banget", "bngt": "banget", "dr": "dari", "dri": "dari",
    "pd": "pada", "udh": "sudah", "udah": "sudah", "sdh": "sudah", "blm": "belum",
    "blom": "belum", "bisa": "bisa", "bs": "bisa", "jd": "jadi", "jdi": "jadi",
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
    "dll": "dan lain lain", "dsb": "dan sebagainya", "미노": ""
}


def remove_emojis(text):
    """Menghapus seluruh karakter emoji dan simbol piktograf dari teks."""
    emoji_pattern = re.compile(
        "["
        "😀-🙏"  # emoticons
        "🌀-🗿"  # symbols & pictographs
        "🚀-🛿"  # transport & map symbols
        "🇠-🇿"  # flags (iOS)
        "─-⯯"  # chinese char
        "✂-➰"
        "Ⓜ-🉑"
        "🤦-🤷"
        "𐀀-􏿿"
        "♀-♂"
        "☀-⭕"
        "‍"
        "⏏"
        "⏩"
        "⌚"
        "️"  # dingbats
        "〰"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r" ", text)


def clean_single_text(text, normalize_slang=True):
    """
    Membersihkan satu teks komentar sesuai spesifikasi diagram CLEANING_TEXT:
    - Memperbaiki error karakter / encoding / HTML entities
    - Menghapus emoji
    - Menghapus URL, mention (@user), dan hashtag (#tag)
    - Mengubah ke huruf kecil (case folding)
    - Mengurangi huruf berulang berlebih (contoh: 'parahhhh' -> 'parah')
    - Menormalisasi kata tidak baku / typo umum (opsional)
    - Merapikan spasi berlebih
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # 1. Hapus newline, tab, dan karakter kontrol
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")

    # 2. Hapus URL / Link (http, https, www)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Hapus Mention (@username) dan Hashtag (#topik)
    text = re.sub(r"@[A-Za-z0-9_.-]+", " ", text)
    text = re.sub(r"#[A-Za-z0-9_]+", " ", text)

    # 4. Hapus Emoji & karakter non-ASCII simbolik
    text = remove_emojis(text)

    # 5. Case Folding (ubah ke huruf kecil semua)
    text = text.lower()

    # 6. Hapus angka timestamp di komentar (misal 01:23) dan karakter tanda baca/simbol
    text = re.sub(r"\d{1,2}:\d{2}(?::\d{2})?", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 7. Kurangi huruf berulang lebih dari 2 kali berurutan (contoh: "seremmm" -> "seremm")
    text = re.sub(r"(.){2,}", r"", text)

    # 8. Normalisasi kata slang / typo singkat
    if normalize_slang:
        words = text.split()
        normalized_words = [SLANG_DICT.get(w, w) for w in words]
        text = " ".join(normalized_words)

    # 9. Hapus spasi ganda / berlebih
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_comments_dataset(input_csv=EXTRACTED_COMMENTS_CSV_PATH, output_csv=CLEANED_COMMENTS_CSV_PATH):
    """
    Memproses seluruh dataset komentar hasil ekstraksi untuk tahap CLEANING_TEXT:
    - Menghapus baris dengan komentar kosong (missing values)
    - Menghapus komentar duplikat (berdasarkan comment_id atau kombinasi video_id + text_original)
    - Menerapkan fungsi clean_single_text pada setiap komentar
    - Menghapus baris yang menjadi kosong setelah dibersihkan (misal komentar yang hanya berisi emoji)
    """
    print("\n" + "=" * 70)
    print("TAHAP 3: CLEANING TEXT")
    print("=" * 70)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"File komentar tidak ditemukan di: {input_csv}")

    df = pd.read_csv(input_csv)
    initial_count = len(df)
    print(f"[*] Jumlah total komentar awal: {initial_count} baris.")

    if df.empty:
        print("[!] Dataset komentar kosong.")
        return df

    # 1. Hapus Missing Values pada kolom teks asli
    df.dropna(subset=["text_original"], inplace=True)
    after_na_count = len(df)
    print(f"[*] Setelah hapus missing values (NaN): {after_na_count} baris (Dihapus: {initial_count - after_na_count}).")

    # 2. Hapus Duplikat Data
    if "comment_id" in df.columns and df["comment_id"].notna().any():
        df.drop_duplicates(subset=["comment_id"], keep="first", inplace=True)
    df.drop_duplicates(subset=["video_id", "text_original"], keep="first", inplace=True)
    after_dup_count = len(df)
    print(f"[*] Setelah hapus komentar duplikat: {after_dup_count} baris (Dihapus: {after_na_count - after_dup_count}).")

    # 3. Proses Cleaning Text
    print("[*] Sedang membersihkan teks komentar (menghapus emoji, simbol, URL, normalisasi typo/slang)...")
    df["text_cleaned"] = df["text_original"].astype(str).apply(clean_single_text)

    # 4. Hapus baris yang teks bersihnya menjadi string kosong (misal tadinya hanya berisi emoji/tanda baca)
    df = df[df["text_cleaned"].str.strip() != ""]
    # Hapus juga duplikat teks bersih pada video yang sama jika diperlukan agar tidak redundan
    df.drop_duplicates(subset=["video_id", "text_cleaned"], keep="first", inplace=True)
    final_count = len(df)
    print(f"[*] Setelah hapus komentar kosong pasca-cleaning: {final_count} baris bersih.")

    # Simpan hasil cleaning ke CSV
    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n[+] Tahap Cleaning Text selesai! File disimpan di: {output_csv}")

    return df


# =====================================================================
# FUNGSI UTAMA (MENJALANKAN PIPELINE DATA PREPARATION)
# =====================================================================
def run_data_preparation_pipeline(max_videos_per_query=15, comment_limit_per_video=None):
    """
    Menjalankan seluruh alur Data Preparation:
    1. Scraping Link Video YouTube (2021-2025) -> CSV
    2. Ekstraksi Seluruh Komentar Video -> CSV
    3. Cleaning Text Komentar -> CSV
    """
    print("Memulai Pipeline Data Preparation Volcano Sentiment Analysis Indonesia (2021-2025)...\n")

    # Tahap 1: Ambil daftar link video YouTube 2021-2025
    # Gunakan `python3 script/data_preparation_volcano_2021_2025.py --skip-links` untuk
    # memakai daftar link yang sudah ada (misal hanya melengkapi komentar yang gagal).
    if "--skip-links" in sys.argv and os.path.exists(LINKS_CSV_PATH):
        print("[*] --skip-links: memakai daftar link yang sudah ada.")
        df_links = pd.read_csv(LINKS_CSV_PATH)
    else:
        df_links = scrape_youtube_links(
            keywords=BASE_KEYWORDS,
            years=YEARS,
            max_results_per_query=max_videos_per_query,
            output_csv=LINKS_CSV_PATH
        )

    # Tahap 2: Ekstrak seluruh komentar dari daftar link video yang didapat
    if not df_links.empty:
        df_comments = extract_all_comments(
            links_csv=LINKS_CSV_PATH,
            output_extracted_csv=EXTRACTED_COMMENTS_CSV_PATH,
            output_raw_csv=RAW_COMMENTS_CSV_PATH,
            limit_per_video=comment_limit_per_video  # None = Ambil seluruh komentar
        )

        # Tahap 3: Cleaning Text
        if not df_comments.empty:
            clean_comments_dataset(
                input_csv=EXTRACTED_COMMENTS_CSV_PATH,
                output_csv=CLEANED_COMMENTS_CSV_PATH
            )

    print("\n" + "=" * 70)
    print("SELURUH PROSES DATA PREPARATION SELESAI!")
    print("=" * 70)
    print(f"1. File Link Video (2021-2025) : {LINKS_CSV_PATH}")
    print(f"2. File Seluruh Komentar (Raw) : {EXTRACTED_COMMENTS_CSV_PATH}")
    print(f"3. File Komentar Bersih        : {CLEANED_COMMENTS_CSV_PATH}")


if __name__ == "__main__":
    # Jalankan seluruh pipeline:
    # - max_videos_per_query: jumlah pencarian maksimal per kata kunci per tahun
    # - comment_limit_per_video=None: mengambil SELURUH komentar pada setiap video tanpa batasan jumlah
    run_data_preparation_pipeline(
        max_videos_per_query=15,
        comment_limit_per_video=None
    )
