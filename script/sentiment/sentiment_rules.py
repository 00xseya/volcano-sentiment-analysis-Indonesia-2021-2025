"""
Leksikon dan aturan domain untuk labeling sentimen komentar gunung Indonesia.

Label kanonik:
    0 = negatif
    1 = netral
    2 = positif

Modul ini hanya menghasilkan skor tambahan (rule prior) yang nanti digabung
dengan probabilitas model transformer. Tujuannya menangkap pola yang sering
salah pada model sentimen umum, misalnya doa, duka, humor, dan hinaan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

LABELS = ("negative", "neutral", "positive")
LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
ID_TO_NAME_ID = {0: "negatif", 1: "netral", 2: "positif"}

# Frasa positif kuat: pujian, terima kasih, doa, rasa syukur.
POSITIVE_STRONG = [
    "terima kasih", "terimakasih", "makasih", "thanks", "thank you",
    "mantap", "mantaf", "mantul", "keren", "bagus", "hebat", "luar biasa",
    "salut", "respect", "respek", "apresiasi", "pahlawan", "heroik",
    "sehat selalu", "sehat sehat", "panjang umur", "semoga sehat",
    "alhamdulillah", "masyaallah", "masya allah", "subhanallah",
    "mukjizat", "berkah", "barokah", "pahala", "ganjaran",
    "semoga", "mudah mudahan", "mudah-mudahan", "moga", "amin", "aamiin",
    "stay safe", "hati hati", "hati-hati", "sukses",
    "bangga", "senang", "suka", "cinta", "rindu", "indah", "memukau",
    "mengagumkan", "menginspirasi", "edukatif", "informatif", "bermanfaat",
    "stunning", "amazing", "awesome", "beautiful", "great", "good", "nice",
    "love", "best", "perfect", "wow", "jos", "joss", "gokil", "asik",
    "seru", "menarik", "ramah", "peduli", "tabah", "sabar", "syukur",
    "dilindungi", "aman", "selamatkan", "doa", "berdoa", "mendoakan",
    "angkat topi", "top", "kereen", "kerenn", "baguss", "hebattt",
    "pinter", "pintar", "baik sekali", "baik banget", "sangat baik",
    "lucu", "kocak", "ngakak", "wkwk", "wkwkwk", "haha", "hehe",
    "awestruck", "holy cow", "allahu akbar",
    "melindungi", "dinantikan",
    "salam", "hormat", "bravo", "pibesdey", "happy birthday",
    "selamat ulang tahun", "lope", "love", "mvp", "epic", "hypnotic",
    "soothing", "cool", "what a show", "show of nature", "terinspirasi",
    "bersyukur", "cocok", "ditunggu", "heroik", "legend", "legenda",
    "enjoy", "santai", "bermakna", "penuh makna", "pembelajaran",
    "alhmdulillah", "lindungi", "lindungilah", "can do it",
    "mksh", "makasi", "makasih", "trims", "thx", "tq",
    "epik", "luarbiasa", "seruu", "kerennn", "bagusss", "mantappp",
    "menikmati", "takjub", "wkwkwkwk", "wkwkwkwkwk", "hahaha", "hehehe",
    "sehat terus", "lanjutkan", "next episode", "next season", "gasin",
    "dicintai", "dasyat", "dahsyat", "koplak",
]

# Frasa positif yang biasanya cukup jelas tanpa konteks tambahan.
POSITIVE_HIGH_PRECISION = [
    "terima kasih", "terimakasih", "makasih", "thanks", "thank you",
    "mantap", "mantaf", "mantul", "keren", "bagus", "hebat", "luar biasa",
    "salut", "respect", "respek", "apresiasi", "pahlawan",
    "sehat selalu", "panjang umur", "alhamdulillah", "masyaallah",
    "masya allah", "subhanallah", "mukjizat", "berkah", "barokah",
    "pinter", "pintar", "baik sekali", "baik banget", "angkat topi",
    "jos", "joss", "gokil", "edukatif", "informatif", "bermanfaat",
    "menginspirasi", "memukau", "mengagumkan",
    "lucu", "kocak", "ngakak", "wkwk", "wkwkwk", "haha", "hehe",
    "awestruck", "holy cow", "allahu akbar",
    "salam hormat", "bravo", "pibesdey", "lope", "mvp", "epic",
    "hypnotic", "soothing", "terinspirasi", "bersyukur", "heroik",
    "legend", "legenda", "can do it", "what a show",
    "mksh", "makasi", "makasih", "trims", "thx", "tq",
    "epik", "luarbiasa", "seruu", "kerennn", "bagusss", "mantappp",
    "menikmati", "takjub", "wkwkwkwk", "wkwkwkwkwk", "hahaha", "hehehe",
    "sehat terus", "lanjutkan", "next episode", "next season", "gasin",
    "dicintai", "dasyat", "dahsyat", "koplak",
]

# Frasa negatif yang biasanya cukup jelas tanpa konteks tambahan.
NEGATIVE_HIGH_PRECISION = [
    "tolol", "goblok", "bodoh", "bangsat", "anjing", "babi", "kontol",
    "memek", "pelacur", "lonte", "bajingan", "kampret", "brengsek",
    "jancok", "jancuk", "pukimak", "tidak peduli", "gak peduli",
    "nggak peduli", "tak peduli", "kecewa", "bohong", "hoax",
    "provokator", "korupsi", "dzalim", "zalim", "sombong",
    "tidak becus", "gak becus", "nggak becus", "penipu", "penipuan",
    "fitnah", "nyinyir", "provokasi",
    "kata siapa", "pecah perawan", "tidak beralur", "seadanya",
    "dzolim", "dholim", "shombong", "astaghfirullah", "astaghfirullahal",
    "dibodohi", "bodohi", "pesek", "rip the", "tidak tenang",
    "tidak kebayang", "mninggoy", "meninggoy",
    "poor", "too many ads", "menyedihkan", "tidak safety",
]

NEGATIVE_ENGLISH = [
    "terrifying", "scary", "sad", "pity", "destroyed", "helpless",
    "horror", "nightmare", "devastating", "tragic", "victim",
]

# Kalimat pendek yang maknanya sudah jelas.
SHORT_TEXT_OVERRIDES = {
    "jos": 2, "joss": 2, "mantap": 2, "mantaf": 2, "mantul": 2,
    "keren": 2, "bagus": 2, "hebat": 2, "sip": 2, "top": 2, "asik": 2,
    "seru": 2, "lucu": 2, "wkwk": 2, "wkwkwk": 2, "haha": 2, "hehe": 2,
    "alhamdulillah": 2, "masyaallah": 2, "subhanallah": 2, "amin": 2,
    "aamiin": 2, "mantab": 2, "gokil": 2,
    "tolol": 0, "goblok": 0, "bodoh": 0, "anjing": 0, "bangsat": 0,
    "kampret": 0, "brengsek": 0, "jancok": 0, "jancuk": 0,
    "ha": 1, "hah": 1, "iyq": 1, "iya": 1, "ya": 1, "sagh": 1,
    "oh yeah": 1, "oh": 1, "thunderstorm": 1, "sama": 1, "sama sama": 1,
}

# Kata yang menandakan duka/kesedihan kuat (bukan kata topik).
SADNESS_WORDS = [
    "sedih", "nangis", "menangis", "miris", "duka", "berduka",
    "prihatin", "kasihan", "pilu", "haru", "harunya", "kehilangan",
    "takut", "takutny", "khawatir", "cemas", "waswas",
]

CONDOLENCE_PHRASES = [
    "turut berduka", "berduka cita", "innalillahi", "al fatihah",
    "alfatihah", "rest in peace", "meninggal dunia", "telah berpulang",
    "inalilahi", "innalilahi", "innallilahi", "inalillah",
]

# Frasa positif lembut: humor dan afeksi.
POSITIVE_SOFT = [
    "lucu", "kocak", "ngakak", "wkwk", "wkwkwk", "wkwkwkwk", "haha",
    "hehe", "hihi", "hahaha", "hehehe", "sayang", "gemas", "gemoy",
    "cute", "lol", "enak", "enak banget", "sehat", "nikmat",
    "unbelievable", "incredible",
]

# Hinaan / ujaran kebencian: sinyal negatif kuat.
NEGATIVE_STRONG = [
    "tolol", "goblok", "bodoh", "bangsat", "anjing", "babi",
    "kontol", "memek", "pelacur", "lonte", "bajingan", "kampret", "tai",
    "sampah", "brengsek", "asu", "jancok", "jancuk", "kimak", "pukimak",
    "tidak peduli", "gak peduli", "nggak peduli", "tak peduli",
    "kecewa", "marah", "parah", "bohong", "hoax", "provokator",
    "korupsi", "dzalim", "zalim", "sombong", "gagal", "tidak becus",
    "gak becus", "nggak becus", "penipu", "penipuan", "fitnah",
    "mengganggu", "nyinyir", "provokasi", "kacau", "salah",
    "sadis", "gak punya empati", "ga punya empati", "nggak punya empati",
    "tidak punya empati", "nonsense", "disgrace",
]

# Emosi negatif: takut, sedih, duka, dan kata peristiwa buruk.
NEGATIVE_MEDIUM = [
    "serem", "seram", "ngeri", "takut", "panik", "cemas", "khawatir",
    "waswas", "mengerikan", "horor", "mimpi buruk", "trauma",
    "sedih", "nangis", "menangis", "miris", "duka", "prihatin",
    "kasihan", "pilu", "haru", "kehilangan", "innalillahi", "astaghfirullah",
    "tragedi", "hancur", "rusak", "kelaparan", "menderita", "sengsara",
]

# Kata topik / peringatan: bobot kecil supaya laporan faktual tetap netral.
NEGATIVE_SOFT = [
    "bahaya", "rawan", "waspada", "korban", "pengungsi", "mengungsi",
    "dampak", "kerusakan", "panas", "gelap", "sesak", "sulit",
    "meninggal", "tewas", "korban jiwa", "korban meninggal",
    "terdampak", "rezim", "stop", "deg2an", "deg-degan",
]

# Kata topik murni: tidak menambah polaritas, hanya penanda konteks.
NEGATIVE_TOPIC = [
    "bencana", "erupsi", "meletus", "letusan", "gempa", "tsunami",
    "lahar", "awan panas", "abu vulkanik",
]

NEGATORS = {
    "tidak", "tak", "bukan", "jangan", "gak", "nggak", "ga", "enggak",
    "belum", "tanpa", "kurang", "bukanlah", "tiada",
}

# Frasa yang maknanya positif walau mengandung kata negatif.
POSITIVE_EXCEPTIONS = [
    "tidak ada korban", "tak ada korban", "tanpa korban", "nol korban",
    "tidak menimbulkan korban", "aman semua", "selamat semua",
    "tidak apa apa", "tidak apa-apa", "baik baik saja", "baik-baik saja",
    "tidak separah", "tidak terlalu parah", "tidak sabar",
]

# Frasa negatif yang tidak boleh dibalik oleh negasi.
NEGATIVE_EXCEPTIONS = [
    "tidak peduli", "gak peduli", "nggak peduli", "tak peduli",
]


_WORD_RE = re.compile(r"[a-z]+(?:-[a-z]+)?")


@dataclass
class RuleResult:
    """Skor aturan dan bukti frasa yang cocok."""

    negative: float = 0.0
    neutral: float = 0.0
    positive: float = 0.0
    hits: list[str] | None = None

    def as_vector(self) -> list[float]:
        return [self.negative, self.neutral, self.positive]


def normalize_text(text: str) -> str:
    """Normalisasi ringan tanpa menghapus informasi sentimen."""
    text = str(text).lower()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _find_phrase(text: str, phrase: str) -> list[int]:
    """Cari frasa dengan batas kata sederhana, kembalikan posisi awal."""
    pattern = r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])"
    return [m.start() for m in re.finditer(pattern, text)]


def _is_negated(text: str, start: int, window: int = 4) -> bool:
    prefix = text[:start]
    tokens = _WORD_RE.findall(prefix)[-window:]
    return any(tok in NEGATORS for tok in tokens)


def _add_hit(hits: list[str], phrase: str, weight: float, polarity: str) -> None:
    hits.append(f"{polarity}:{phrase}({weight:+.2f})")


def strong_label(text: str) -> tuple[int | None, str]:
    """Aturan presisi tinggi yang boleh mengalahkan prediksi model.

    Returns:
        (label_id atau None, alasan)
    """
    text = normalize_text(text)
    tokens = _WORD_RE.findall(text)

    # 1. Kalimat pendek yang sudah jelas.
    if text in SHORT_TEXT_OVERRIDES:
        return SHORT_TEXT_OVERRIDES[text], "short-text"

    # 2. Duka/belasungkawa selalu negatif.
    if any(_find_phrase(text, p) for p in CONDOLENCE_PHRASES):
        return 0, "condolence"

    has_sadness = any(_find_phrase(text, p) for p in SADNESS_WORDS)
    has_positive_praise = any(
        _find_phrase(text, p) for p in (
            "hebat", "semangat", "salut", "pahlawan", "bangga",
            "luar biasa", "inspirasi", "menginspirasi", "tabah",
        )
    )
    pos_hits = [p for p in POSITIVE_HIGH_PRECISION if _find_phrase(text, p)]
    neg_hits = [p for p in NEGATIVE_HIGH_PRECISION if _find_phrase(text, p)]

    # 3. Kesedihan dominan (kata emosi eksplisit).
    if has_sadness and not has_positive_praise:
        return 0, "sadness"

    # 4. Hinaan/kritik eksplisit mengalahkan doa atau pujian.
    if neg_hits:
        return 0, f"negative:{neg_hits[0]}"

    # 5. Doa/harapan baik.
    if any(_find_phrase(text, p) for p in ("semoga", "mudah mudahan",
                                           "mudah-mudahan", "moga", "amin",
                                           "aamiin", "smgh", "smga", "smoga")):
        if not any(_find_phrase(text, p) for p in ("mati", "matian",
                                                   "celaka", "binasa")):
            return 2, "prayer"

    # 6. Pujian/terima kasih yang eksplisit.
    if pos_hits:
        # Jangan override jika ada negasi langsung pada frasa positif.
        if not any(_is_negated(text, start) for p in pos_hits
                   for start in _find_phrase(text, p)):
            return 2, f"positive:{pos_hits[0]}"

    # 7. Pertanyaan / permintaan informasi.
    if any(_find_phrase(text, p) for p in (
        "ingin tahu", "mau nanya", "mau tanya", "bertanya", "nanya",
    )):
        return 1, "question-intent"

    # 8. Pertanyaan / kalimat tanpa sinyal.
    question_words = ("apa", "apakah", "bagaimana", "kenapa", "mengapa",
                      "siapa", "kapan", "dimana", "berapa", "kah")
    is_question = "?" in text or any(_find_phrase(text, w) for w in question_words)
    if is_question and len(tokens) <= 12:
        return 1, "question"

    return None, ""


def score_rules(text: str) -> RuleResult:
    """Hitung rule prior untuk satu kalimat.

    Skor positif/negatif bersifat aditif. Frasa yang dinegasi dibalik
    polaritasnya, kecuali frasa yang sudah eksplisit mengandung negasi.
    """
    text = normalize_text(text)
    result = RuleResult(hits=[])

    # 1. Exception positif (mis. "tidak ada korban") diproses lebih dulu.
    for phrase in POSITIVE_EXCEPTIONS:
        for _ in _find_phrase(text, phrase):
            result.positive += 2.0
            _add_hit(result.hits, phrase, 2.0, "pos")

    # 2. Frasa negatif eksplisit yang tidak boleh dibalik.
    for phrase in NEGATIVE_EXCEPTIONS:
        for _ in _find_phrase(text, phrase):
            result.negative += 2.0
            _add_hit(result.hits, phrase, 2.0, "neg")

    # 3. Leksikon utama. Frasa exception sudah dihitung di atas.
    exceptions = set(POSITIVE_EXCEPTIONS) | set(NEGATIVE_EXCEPTIONS)
    for phrase, weight, polarity in _iter_lexicon():
        if phrase in exceptions:
            continue
        for start in _find_phrase(text, phrase):
            # Lewati jika sudah tercakup exception (hindari double count).
            if _covered_by_exception(text, start, phrase):
                continue

            negated = _is_negated(text, start)
            if polarity == "pos":
                if negated:
                    result.negative += weight * 0.7
                    _add_hit(result.hits, phrase, -weight * 0.7, "neg(negated)")
                else:
                    result.positive += weight
                    _add_hit(result.hits, phrase, weight, "pos")
            else:
                if negated:
                    result.positive += weight * 0.5
                    _add_hit(result.hits, phrase, -weight * 0.5, "pos(negated)")
                else:
                    result.negative += weight
                    _add_hit(result.hits, phrase, weight, "neg")

    # 4. Aturan khusus yang lebih presisi.
    result = _apply_special_rules(text, result)
    return result


def _iter_lexicon():
    for phrase in POSITIVE_STRONG:
        yield phrase, 1.6, "pos"
    for phrase in POSITIVE_SOFT:
        yield phrase, 0.7, "pos"
    for phrase in NEGATIVE_STRONG:
        yield phrase, 2.0, "neg"
    for phrase in NEGATIVE_MEDIUM:
        yield phrase, 1.0, "neg"
    for phrase in NEGATIVE_SOFT:
        yield phrase, 0.3, "neg"
    for phrase in NEGATIVE_ENGLISH:
        yield phrase, 2.0, "neg"


def _covered_by_exception(text: str, start: int, phrase: str) -> bool:
    """Cegah double counting untuk frasa di dalam exception."""
    for exc in POSITIVE_EXCEPTIONS + NEGATIVE_EXCEPTIONS:
        for exc_start in _find_phrase(text, exc):
            if exc_start <= start < exc_start + len(exc):
                return exc != phrase
    return False


def _apply_special_rules(text: str, result: RuleResult) -> RuleResult:
    """Aturan yang mengalahkan leksikon umum."""
    # Duka / belasungkawa -> negatif, walaupun ada kata doa.
    condolence = [
        "turut berduka", "berduka cita", "innalillahi", "al fatihah",
        "rest in peace", "rip ", "meninggal dunia", "telah berpulang",
        "husnul khotimah", "khusnul khotimah",
    ]
    if any(_find_phrase(text, p) for p in condolence):
        result.negative += 2.0
        result.positive = max(0.0, result.positive - 1.0)
        _add_hit(result.hits, "condolence-rule", 2.0, "neg")

    # "astaghfirullah" biasanya ekspresi syok/penyesalan, bukan doa bahagia.
    if _find_phrase(text, "astaghfirullah"):
        result.negative += 0.8
        _add_hit(result.hits, "astaghfirullah-shock", 0.8, "neg")

    # Doa/harapan baik: pastikan tetap positif meski ada kata topik negatif.
    if _find_phrase(text, "semoga") or _find_phrase(text, "mudah mudahan") or _find_phrase(text, "moga"):
        if not any(_find_phrase(text, p) for p in ("mati", "matian", "celaka", "binasa")):
            result.positive += 1.0
            _add_hit(result.hits, "prayer-rule", 1.0, "pos")

    # Kalimat tanya / kalimat tanpa sinyal polaritas cenderung netral.
    question_words = ("apa", "apakah", "bagaimana", "kenapa", "mengapa",
                      "siapa", "kapan", "dimana", "berapa", "kah")
    is_question = "?" in text or any(_find_phrase(text, w) for w in question_words)
    if is_question and result.negative == 0 and result.positive == 0:
        result.neutral += 1.5
        _add_hit(result.hits, "question-rule", 1.5, "neu")

    if result.negative == 0 and result.positive == 0:
        # Tidak ada sinyal sama sekali: beri sedikit dorongan netral.
        result.neutral += 0.5
        _add_hit(result.hits, "no-signal-rule", 0.5, "neu")

    return result


def rule_label(text: str) -> tuple[int, float, list[str]]:
    """Label berbasis aturan saja (untuk audit / baseline).

    Returns:
        (label_id, margin, hits)
    """
    result = score_rules(text)
    strong_id, reason = strong_label(text)
    if strong_id is not None:
        return strong_id, 5.0, [f"strong:{reason}"] + (result.hits or [])
    scores = result.as_vector()
    if max(scores) < 0.6:
        return 1, 0.0, result.hits or []
    label = max(range(3), key=lambda i: scores[i])
    ordered = sorted(scores, reverse=True)
    margin = ordered[0] - ordered[1]
    return label, margin, result.hits or []


if __name__ == "__main__":
    examples = [
        "semoga tidak ada korban jiwa",
        "turut berduka cita untuk korban",
        "mantap sekali penjelasannya",
        "pemerintah tidak peduli rakyat kecil",
        "apa itu gunung berapi",
        "wkwkwk lucu banget",
        "tolol",
        "alhamdulillah aman semua",
    ]
    for example in examples:
        label, margin, hits = rule_label(example)
        print(f"{label} ({ID_TO_NAME_ID[label]}, margin={margin:.2f}) | {example}")
        print(f"   hits: {hits}")
