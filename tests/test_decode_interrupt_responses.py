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


def test_no_answers_decode_to_no_resume_input() -> None:
    assert decode_interrupt_responses({}) == {}


def test_hitl_answers_are_rejoined_into_decisions() -> None:
    answers = {
        "i-1#0": "welt-io:hitl:approve",
        "i-1#1": "welt-io:hitl:reject",
        "i-1#2": "ask ops first",
    }

    assert decode_interrupt_responses(answers) == {
        "i-1": {
            "decisions": [
                {"type": "approve"},
                {"type": "reject"},
                {"type": "respond", "message": "ask ops first"},
            ]
        }
    }


def test_hitl_decisions_follow_action_order() -> None:
    answers = {"i-1#1": "welt-io:hitl:reject", "i-1#0": "welt-io:hitl:approve"}

    assert decode_interrupt_responses(answers) == {
        "i-1": {"decisions": [{"type": "approve"}, {"type": "reject"}]}
    }


def test_an_answer_carrying_no_button_value_becomes_a_respond() -> None:
    answers = {"i-1#0": "approve"}

    assert decode_interrupt_responses(answers) == {
        "i-1": {"decisions": [{"type": "respond", "message": "approve"}]}
    }


def test_hitl_answers_leaving_a_gap_travel_on_for_the_middleware_to_refuse() -> None:
    answers = {"i-1#0": "welt-io:hitl:approve", "i-1#2": "welt-io:hitl:approve"}

    assert decode_interrupt_responses(answers) == {
        "i-1": {"decisions": [{"type": "approve"}, {"type": "approve"}]}
    }


def test_hitl_answers_keep_the_place_of_their_first_answer() -> None:
    answers = {
        "i-1": "y",
        "i-2#0": "welt-io:hitl:approve",
        "i-3": "n",
        "i-2#1": "welt-io:hitl:reject",
    }

    assert list(decode_interrupt_responses(answers)) == ["i-1", "i-2", "i-3"]


def test_hitl_requests_are_rejoined_one_by_one() -> None:
    answers = {"i-1#0": "welt-io:hitl:approve", "i-2#0": "welt-io:hitl:reject"}

    assert decode_interrupt_responses(answers) == {
        "i-1": {"decisions": [{"type": "approve"}]},
        "i-2": {"decisions": [{"type": "reject"}]},
    }


def test_an_id_without_an_index_answers_a_plain_interrupt() -> None:
    answers = {"i-1#x": "y", "#0": "n", "i-2#": "y"}

    assert decode_interrupt_responses(answers) == {"i-1#x": "y", "#0": "n", "i-2#": "y"}
