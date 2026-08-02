"""Unit tests for AlphaSense-Style Free Intelligence Engine ($0 Cost)."""

from __future__ import annotations

import pytest

from finb.data.sources.alphasense_free import AlphaSenseFreeEngine


def test_analyze_filing_text_sentiment():
    engine = AlphaSenseFreeEngine()
    
    bullish_text = "The company reported record growth and profit expansion with positive dividend outlook."
    score = engine.analyze_filing_text(bullish_text)
    assert score > 0.5

    bearish_text = "The company faces decline loss investigation lawsuit and risk of default."
    score_bearish = engine.analyze_filing_text(bearish_text)
    assert score_bearish < -0.5


def test_generate_intelligence_report():
    engine = AlphaSenseFreeEngine()
    filing = "Strong quarterly profit and expansion beat expectations."
    report = engine.generate_intelligence_report("AAPL", filing_text=filing, news_count=15, avg_news_count=5.0, vix_level=15.0)

    assert report.symbol == "AAPL"
    assert report.filing_sentiment_score > 0.0
    assert report.news_volume_surge == pytest.approx(3.0)
    assert report.macro_regime_flag == "LOW_VOL"
