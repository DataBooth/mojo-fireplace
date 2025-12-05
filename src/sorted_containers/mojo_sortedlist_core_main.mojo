from mojo_sortedlist_core import SortedList


fn demo_int() raises:
    var sl = SortedList[Int]()

    # Basic insertion
    sl.add(3)
    sl.add(1)
    sl.add(2)
    sl.add(2)

    print("Int demo – after inserts (expected [1, 2, 2, 3]):")
    for i in range(sl.__len__()):
        print(" ", sl.get_item(i))

    # Removal
    sl.remove(2)
    print("Int demo – after remove(2) (expected [1, 2, 3]):")
    for i in range(sl.__len__()):
        print(" ", sl.get_item(i))

    # Negative index
    var last = sl.get_item(-1)
    print("Int demo – last element (expected 3):", last)


fn demo_float() raises:
    var sl = SortedList[Float64]()

    sl.add(3.5)
    sl.add(1.0)
    sl.add(2.25)

    print(
        "Float demo: Create list by add 3.5, then 1.0, then 2.25 – after"
        " inserts (expected [1.0, 2.25, 3.5]):"
    )
    for i in range(sl.__len__()):
        print(" ", sl.get_item(i))

    var last = sl.get_item(-1)
    print("Float demo – last element (expected 3.5):", last)


fn main() raises:
    demo_int()
    demo_float()
