from welt_io_langgraph import decode_interrupt_responses


def test_answers_become_the_resume_mapping() -> None:
    answers = {"i-1": "approve", "i-2": "n"}

    assert decode_interrupt_responses(answers) == {"i-1": "approve", "i-2": "n"}


def test_answer_order_is_preserved() -> None:
    answers = {"i-2": "n", "i-1": "y"}

    assert list(decode_interrupt_responses(answers)) == ["i-2", "i-1"]


def test_the_input_is_left_untouched() -> None:
    answers = {"i-1": "y"}

    decoded = decode_interrupt_responses(answers)
    decoded["i-1"] = "changed"

    assert answers == {"i-1": "y"}


def test_non_string_answers_are_skipped() -> None:
    answers = {"i-1": "y", "i-2": 42, "i-3": None}

    assert decode_interrupt_responses(answers) == {"i-1": "y"}


def test_no_answers_decode_to_an_empty_mapping() -> None:
    assert decode_interrupt_responses({}) == {}
