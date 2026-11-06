def touching_counts(a, b):
    return max(a[0], b[0]) <= min(a[1], b[1])


def strict_overlap(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])


def needs_two(a, b):
    return min(a[1], b[1]) - max(a[0], b[0]) >= 2


def positive_length(a, b):
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    return end - start > 0
