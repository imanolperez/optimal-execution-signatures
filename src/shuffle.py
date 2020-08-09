def concatenate(u, list_words):
    """Concatenates a letter with each word in a list."""

    return [tuple([u] + list(word)) for word in list_words]

def shuffle(w1, w2):
    """Computes the shuffle product of two words."""

    if len(w1) == 0:
        return [w2]

    if len(w2) == 0:
        return [w1]

    return concatenate(w1[0], shuffle(w1[1:], w2)) + \
           concatenate(w2[0], shuffle(w1, w2[1:]))
