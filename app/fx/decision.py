"""Current interpretable FX action policy."""

from dataclasses import dataclass

from app.fx.features import FxStatistics


@dataclass(frozen=True)
class Decision:
    action: str
    rationale: str


def decide(base: str, quote: str, stats: FxStatistics) -> Decision:
    if stats.current >= stats.ma_7d and stats.z_score > 0.5:
        return Decision(
            "CONVERT_NOW",
            f"Kurs {base}/{quote} saat ini ({stats.current:,.2f}) berada di atas "
            f"rata-rata pergerakan 7 hari ({stats.ma_7d:,.2f}). Momentum menguntungkan untuk konversi.",
        )
    if stats.current < stats.ma_7d and stats.z_score < -0.5:
        return Decision(
            "WAIT",
            f"Kurs saat ini ({stats.current:,.2f}) di bawah rata-rata 7 hari "
            f"({stats.ma_7d:,.2f}). Disarankan menunggu rebound sebelum konversi.",
        )
    return Decision(
        "HOLD",
        f"Kurs bergerak netral di sekitar rata-rata 7 hari ({stats.ma_7d:,.2f}). "
        "Tidak ada sinyal kuat; konversi sesuai kebutuhan likuiditas.",
    )
