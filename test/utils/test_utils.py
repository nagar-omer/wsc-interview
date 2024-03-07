from wsc_interview.utils.utils import find_params, find_tokens


# sanity check find_params
def test_sanity_find_params():
    text = "here comes called her on the high pick and roll rolling inside lebron james year behind the back floater short"
    params = ['floater', 'pick and roll', 'jam', 'behind the back']

    assert find_params(text, params[0]) == [(97, 104)]
    assert find_params(text, params[1]) == [(34, 47)]
    assert find_params(text, params[2]) == []
    assert find_params(text, params[3]) == [(81, 96)]


def test_double_find_params():
    text = "the pick and roll is a great play, pick and roll"
    params = ['pick and roll']

    assert find_params(text, params[0]) == [(4, 17), (35, 48)]


def test_start_of_sentence_find_params():
    text = "pick and roll by Jordan"
    params = ['pick and roll']
    assert find_params(text, params[0]) == [(0, 13)]


def test_empty_find_params():
    text = "Pick and roll by Jordan"
    params = ['pick and roll']
    assert find_params("", params[0]) == []
    assert find_params(text, "")


def text_sanity_find_tokens():
    text_tokens = ['[CLS]', 'here', 'comes', 'called', 'her', 'on', 'the', 'high', 'pick', 'and', 'roll', 'rolling',
                   'inside', 'le', '##bron', 'james', 'year', 'behind', 'the', 'back', 'float', '##er', 'short']
    phrase_tokens = ['back', 'float', '##er']
    assert find_tokens(text_tokens, phrase_tokens) == [18, 19, 20]


def test_empty_find_tokens():
    text_tokens = []
    phrase_tokens = ['back', 'float', '##er']
    assert find_tokens(text_tokens, phrase_tokens) == []


def test_not_exist_find_tokens():
    text_tokens = ['[CLS]', 'here', 'comes', 'called', 'her', 'on', 'the', 'high', 'pick', 'and', 'roll', 'rolling',
                   'inside', 'le', '##bron', 'james', 'year', 'behind', 'the', 'back', 'float', '##er', 'short']
    phrase_tokens = ['back', 'floating', '##er']
    assert find_tokens(text_tokens, phrase_tokens) == []
