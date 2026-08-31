import pytest

from holdem.cards import (
    Deck, card_str, hole_label, make_card, parse_card, parse_cards, rank_of, suit_of,
)


def test_card_encoding_roundtrip():
    for card in range(52):
        assert parse_card(card_str(card)) == card
        assert make_card(rank_of(card), suit_of(card)) == card


def test_parse_forms():
    assert parse_cards("AsKh") == parse_cards("As Kh") == parse_cards(["As", "Kh"])
    with pytest.raises(ValueError):
        parse_card("Xx")
    with pytest.raises(ValueError):
        parse_card("A")


def test_hole_labels():
    assert hole_label(parse_cards("AsKs")) == "AKs"
    assert hole_label(parse_cards("KhAd")) == "AKo"
    assert hole_label(parse_cards("7c7d")) == "77"


def test_deck_deals_without_repeats(rng):
    deck = Deck(rng)
    drawn = deck.deal(52)
    assert sorted(drawn) == list(range(52))
    assert deck.remaining == 0
    with pytest.raises(ValueError):
        deck.deal(1)


def test_deck_stack_next_controls_the_next_cards(rng):
    deck = Deck(rng)
    wanted = parse_cards("AsKdQh")
    deck.stack_next(wanted)
    assert deck.deal(3) == wanted
    rest = deck.deal(deck.remaining)
    assert not set(rest) & set(wanted)
