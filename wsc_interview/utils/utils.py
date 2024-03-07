import re
from wsc_interview import logger


def find_params(text: str, param: str) -> list:
    """
    Find parameters in a given text.
    """
    logger.info(f"Finding param in text: {text}")

    assert isinstance(text, str), "text must be a string"
    assert isinstance(param, str), "param must be a string"

    pattern = r'\b' + re.escape(param) + r'\b'

    # Search for the pattern in the sentence
    matches = re.finditer(pattern, text)

    return [(match.start(), match.end())for match in matches]


def find_tokens(text_tokens: list, phrase_tokens: list) -> list:
    """
    This function search the phrase in the text and return the tokens idx (first appearance) of the phrase in the text
    :param text_tokens: list of tokens (or ids)
    :param phrase_tokens: list of tokens (or ids)
    :return: list of tokens idx
    """
    # skip CLS token

    for i in range(len(text_tokens) - len(phrase_tokens) + 1):
        if text_tokens[i:i + len(phrase_tokens)] == phrase_tokens:
            return list(range(i, i + len(phrase_tokens)))
    return []


if __name__ == '__main__':
    text = "here comes called her on the high pick and roll rolling inside lebron james year behind the back floater short"
    params = ['floater', 'pick and roll', 'jam', 'behind the back']

    print(find_params(text, params[0]))  # ['Python', 'Java', 'Django', 'Flask', 'Spring Boot']