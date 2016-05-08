import setup_pl2

def test_num_parts_to_len_per_part():
    assert setup_pl2.num_parts_to_len_per_part(2, 2) == 1
    assert setup_pl2.num_parts_to_len_per_part(3, 2) == 2
    assert setup_pl2.num_parts_to_len_per_part(4, 2) == 2
    assert setup_pl2.num_parts_to_len_per_part(5, 2) == 3
    assert setup_pl2.num_parts_to_len_per_part(6, 2) == 3

    assert setup_pl2.num_parts_to_len_per_part(14, 2) == 7
    assert setup_pl2.num_parts_to_len_per_part(15, 2) == 8
    assert setup_pl2.num_parts_to_len_per_part(16, 2) == 8
    assert setup_pl2.num_parts_to_len_per_part(17, 2) == 9
    assert setup_pl2.num_parts_to_len_per_part(18, 2) == 9

    assert setup_pl2.num_parts_to_len_per_part(20, 2) == 10

def test_append_reverse_edges():
    c = append_reverse_edges([], {})
