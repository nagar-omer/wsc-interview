from wsc_interview.utils.eda_utils import label_distribution, text_length_distribution


def test_sanity_label_distribution():
    labels = ['a', 'b', 'c', 'a', 'b', 'a']
    assert label_distribution(labels) == {'a': 3, 'b': 2, 'c': 1}

    labels = ['a', 'b', 'c', 'a', 'b', 'a', 'c', 'c', 'c']
    assert label_distribution(labels) == {'a': 3, 'b': 2, 'c': 4}


def test_empty_label_distribution():
    labels = []
    assert label_distribution(labels) == {}


def test_sanity_text_length_distribution():
    texts = ['this is a test', 'this is another test', 'this is the last test']
    assert text_length_distribution(texts) == {4: 2, 5: 1}

    texts = ['this is a test', 'this is another test', 'this is the last test', 'this is the last test']
    assert text_length_distribution(texts) == {4: 2, 5: 2}


def test_empty_text_length_distribution():
    texts = []
    assert text_length_distribution(texts) == {}


def test_special_characters_text_length_distribution():
    texts = ['this is $ %a test', 'this is another  - test', 'this is , the last test!']
    assert text_length_distribution(texts) == {4: 2, 5: 1}

    texts = ['this is a test###', '!@? this is another test? ', 'this is the last test? ', 'this is the last test']
    assert text_length_distribution(texts) == {4: 2, 5: 2}