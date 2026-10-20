"""Page expression parsing. No third-party dependency."""

from rag.utils import format_pages, parse_pages


def test_parse_single_and_range():
    assert parse_pages("1-5,8") == [1, 2, 3, 4, 5, 8]


def test_parse_is_sorted_and_deduplicated():
    assert parse_pages("8,1-3,2") == [1, 2, 3, 8]


def test_parse_open_range():
    assert parse_pages("1-3,") == [1, 2, 3]


def test_parse_empty_returns_empty_list():
    assert parse_pages("") == []
    assert parse_pages("   ") == []


def test_parse_ignores_invalid_tokens():
    assert parse_pages("abc, 4, ,;5") == [4, 5]


def test_parse_reversed_range_is_normalized():
    assert parse_pages("5-3") == [3, 4, 5]


def test_format_pages_compresses_ranges():
    assert format_pages([1, 2, 3, 7]) == "1-3, 7"
    assert format_pages([]) == "-"
