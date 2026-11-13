from pathlib import Path

from mind_recsys.data.dataset import (
    MINDImpression,
    format_news_text,
    load_behaviors,
    load_news,
    parse_history,
    parse_impressions,
)


def test_format_news_text():
    formatted = format_news_text(
        category="sports",
        subcategory="football",
        title="Super Bowl Preview",
        abstract="Exciting match coming up.",
    )
    assert formatted == "[sports] football: Super Bowl Preview. Exciting match coming up."


def test_format_news_text_empty_abstract():
    formatted = format_news_text(
        category="news",
        subcategory="",
        title="Breaking News",
        abstract="",
    )
    assert formatted == "[news]: Breaking News"


def test_parse_history():
    assert parse_history("N123 N456 N789") == ["N123", "N456", "N789"]
    assert parse_history("") == []
    assert parse_history(None) == []


def test_parse_impressions():
    parsed = parse_impressions("N100-1 N200-0 N300-1")
    assert parsed == [("N100", 1), ("N200", 0), ("N300", 1)]
    assert parse_impressions("") == []
    assert parse_impressions("bad_token") == []


def test_mind_impression_model():
    imp = MINDImpression(
        impression_id="1",
        user_id="U1001",
        time="11/11/2019 9:05:58 AM",
        history=["N10", "N20"],
        candidates=[("N30", 1), ("N40", 0), ("N50", 1)],
    )
    assert imp.clicked_news_ids == ["N30", "N50"]
    assert imp.non_clicked_news_ids == ["N40"]


def test_load_news_and_behaviors(tmp_path: Path):
    news_file = tmp_path / "news.tsv"
    news_file.write_text(
        "N100\tsports\tfootball\tMatch Preview\tTeam A vs Team B\thttp://news/100\t[]\t[]\n"
        "N200\tfinance\t\tMarket Rally\tStocks rise\thttp://news/200\t[]\t[]\n",
        encoding="utf-8",
    )

    behaviors_file = tmp_path / "behaviors.tsv"
    behaviors_file.write_text(
        "1\tU101\t11/11/2019 9:00:00 AM\tN100 N200\tN300-1 N400-0\n",
        encoding="utf-8",
    )

    df_news = load_news(news_file)
    assert len(df_news) == 2
    assert "text_repr" in df_news.columns
    assert df_news.iloc[0]["text_repr"] == "[sports] football: Match Preview. Team A vs Team B"

    df_beh = load_behaviors(behaviors_file)
    assert len(df_beh) == 1
    assert df_beh.iloc[0]["parsed_history"] == ["N100", "N200"]
    assert df_beh.iloc[0]["parsed_impressions"] == [("N300", 1), ("N400", 0)]
