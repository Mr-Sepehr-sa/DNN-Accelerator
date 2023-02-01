
class ListElement:
    def __init__(self, value, next):
        self.value = value
        self.next = next
    def nth(self, n):
        o = self
        i = 0
        while i < n and o.next is not None:
            o = o.next
            i += 1
        return o

def init(multiset):
    multiset.sort() # ensures proper non-increasing order
    h = ListElement(multiset[0], None)
    for item in multiset[1:]:
        h = ListElement(item, h)
    return h, h.nth(len(multiset) - 2), h.nth(len(multiset) - 1)

def visit(h):
    """Converts our bespoke linked list to a python list."""
    o = h
    l = []
    while o is not None:
        l.append(o.value)
        o = o.next
    return l

def permutations(multiset):
    """Generator providing all multiset permutations of a multiset."""
    h, i, j = init(multiset)
    yield visit(h)
    while j.next is not None or j.value < h.value:
        if j.next is not None and i.value >= j.next.value:
            s = j
        else:
            s = i
        t = s.next
        s.next = t.next
        t.next = h
        if t.value < h.value:
            i = t
        j = i.next
        h = t
        yield visit(h)

if __name__ == "__main__":
    multiset = [('OX', 2), ('OX', 2), ('OX', 3), ('OY', 2), ('K', 2), ('K', 2), ('K', 2), ('K', 3), ('K', 3), ('K', 3), ('K', 3), ('C', 2), ('C', 2), ('C', 2), ('C', 2), ('C', 2)]
    i = 0
    for ordering in permutations(multiset):
        # print(ordering)
        i += 1
    print(i)