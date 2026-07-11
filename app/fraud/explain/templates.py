"""Indonesian-language templates for model (SHAP)-derived risk factors.

Used only for features whose SHAP contribution pushes a transaction toward "fraud"
but which no transparent rule already covers. Rules take precedence (they carry the
value-specific wording); these fill in model-only signals. Each feature maps to a
semantic topic so factors from different sources can be de-duplicated per topic.
"""

from collections.abc import Mapping
from typing import Callable

# Feature -> semantic topic (shared vocabulary with rules_engine topics).
FEATURE_TOPIC: dict[str, str] = {
    "amount_idr": "amount",
    "amount_ratio_user": "amount",
    "amount_zscore_user": "amount",
    "hour_sin": "odd_hour",
    "hour_cos": "odd_hour",
    "duplicate_similarity": "duplicate",
    "payer_velocity_10m": "velocity",
    "payer_velocity_24h": "velocity",
    "user_velocity_24h": "velocity",
    "is_new_payer": "new_payer",
    "payer_seen_before": "new_payer",
    "payer_age_days": "new_payer",
    "payer_name_quality": "identity_name",
    "country_seen_before": "country",
    "currency_seen_before": "currency",
}

# Feature -> Indonesian message (may use the transaction row for context).
FEATURE_TEMPLATE: dict[str, Callable[[Mapping], str]] = {
    "amount_idr": lambda r: "Nominal transaksi tergolong besar.",
    "amount_ratio_user": lambda r: "Nominal transaksi lebih besar dari pola normal pengguna.",
    "amount_zscore_user": lambda r: "Nominal transaksi menyimpang dari kebiasaan pengguna.",
    "hour_sin": lambda r: f"Waktu transaksi tidak biasa (sekitar pukul {int(r.get('hour', 0)):02d}:00).",
    "hour_cos": lambda r: f"Waktu transaksi tidak biasa (sekitar pukul {int(r.get('hour', 0)):02d}:00).",
    "duplicate_similarity": lambda r: "Transaksi menyerupai pembayaran sebelumnya.",
    "payer_velocity_10m": lambda r: "Frekuensi transaksi pembayar dalam 10 menit terakhir tinggi.",
    "payer_velocity_24h": lambda r: "Frekuensi transaksi pembayar dalam 24 jam terakhir tinggi.",
    "user_velocity_24h": lambda r: "Aktivitas transaksi pengguna dalam 24 jam terakhir meningkat.",
    "is_new_payer": lambda r: "Pembayar baru pertama kali bertransaksi dengan pengguna.",
    "payer_seen_before": lambda r: "Pembayar baru pertama kali bertransaksi dengan pengguna.",
    "payer_age_days": lambda r: "Identitas pembayar masih tergolong baru.",
    "payer_name_quality": lambda r: "Kualitas nama pembayar rendah.",
    "country_seen_before": lambda r: "Negara asal tidak biasa bagi pengguna.",
    "currency_seen_before": lambda r: "Mata uang tidak biasa bagi pengguna.",
}

# A high-risk transaction must always have a reason, even if nothing specific triggers.
FALLBACK_FACTOR = "Pola transaksi terdeteksi tidak biasa oleh model."
