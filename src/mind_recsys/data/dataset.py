from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

BEHAVIORS_COLUMNS = [
    "impression_id",
    "user_id",
    "time",
    "history",
    "impressions",
]


class MINDNews(BaseModel):
    news_id: str
    category: str
    subcategory: str
    title: str
    abstract: str
    text_repr: str
    url: str | None = None


class MINDImpression(BaseModel):
    impression_id: str
    user_id: str
    time: str
    history: list[str] = Field(default_factory=list)
    candidates: list[tuple[str, int]] = Field(default_factory=list)

    @property
    def clicked_news_ids(self) -> list[str]:
        return [nid for nid, clicked in self.candidates if clicked == 1]

    @property
    def non_clicked_news_ids(self) -> list[str]:
        return [nid for nid, clicked in self.candidates if clicked == 0]


def format_news_text(
    category: str,
    subcategory: str,
    title: str,
    abstract: str,
) -> str:
    cat = (category or "General").strip()
    subcat = (subcategory or "").strip()
    tit = (title or "").strip()
    abstr = (abstract or "").strip()

    prefix = f"[{cat}] {subcat}: " if subcat else f"[{cat}]: "
    suffix = f"{tit}. {abstr}" if abstr else tit
    return f"{prefix}{suffix}".strip()


def parse_history(history_val: str | float | None) -> list[str]:
    if not history_val or pd.isna(history_val):
        return []
    return str(history_val).strip().split()


def parse_impressions(impressions_val: str | float | None) -> list[tuple[str, int]]:
    if not impressions_val or pd.isna(impressions_val):
        return []
    result: list[tuple[str, int]] = []
    for item in str(impressions_val).strip().split():
        parts = item.split("-")
        if len(parts) == 2:
            try:
                result.append((parts[0], int(parts[1])))
            except ValueError:
                continue
    return result


def load_news(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        names=NEWS_COLUMNS,
        header=None,
        dtype=str,
        quoting=3,  # QUOTE_NONE
        on_bad_lines="skip",
    )
    df["category"] = df["category"].fillna("General")
    df["subcategory"] = df["subcategory"].fillna("")
    df["title"] = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")

    df["text_repr"] = df.apply(
        lambda r: format_news_text(
            r["category"],
            r["subcategory"],
            r["title"],
            r["abstract"],
        ),
        axis=1,
    )
    return df


def load_behaviors(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        names=BEHAVIORS_COLUMNS,
        header=None,
        dtype=str,
        quoting=3,
        on_bad_lines="skip",
    )
    df["history"] = df["history"].fillna("")
    df["impressions"] = df["impressions"].fillna("")
    df["parsed_history"] = df["history"].apply(parse_history)
    df["parsed_impressions"] = df["impressions"].apply(parse_impressions)
    return df


@dataclass
class MINDDataset:
    data_dir: Path

    def load_train_news(self) -> pd.DataFrame:
        return load_news(self.data_dir / "train" / "news.tsv")

    def load_train_behaviors(self) -> pd.DataFrame:
        return load_behaviors(self.data_dir / "train" / "behaviors.tsv")

    def load_dev_news(self) -> pd.DataFrame:
        return load_news(self.data_dir / "dev" / "news.tsv")

    def load_dev_behaviors(self) -> pd.DataFrame:
        return load_behaviors(self.data_dir / "dev" / "behaviors.tsv")
