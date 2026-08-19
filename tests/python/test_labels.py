"""
Covers the full SAMLabels API: clean_labels, masks, merge, to_binary,
class_distribution, name resolution, one-hot, stratified split, take,
relabel and dict serialization.
"""

import numpy as np
import pytest

from samcore import SAMLabels, merge_labels


# ── clean labels ────────────────────────────────────────────────────────────

def test_clean_already_consecutive():
    sl = SAMLabels(np.array([0, 1, 2], dtype=np.int8), ["healthy", "A", "B"])
    sl.clean_labels()
    np.testing.assert_array_equal(sl.labels, [0, 1, 2])
    assert sl.label_names == ["healthy", "A", "B"]


def test_clean_with_gaps():
    sl = SAMLabels(np.array([0, 3, 5, 3], dtype=np.int8),
                   ["healthy", "a", "b", "c", "d", "e"])
    sl.clean_labels()
    np.testing.assert_array_equal(sl.labels, [0, 1, 2, 1])
    assert sl.label_names == ["healthy", "c", "e"]


def test_clean_all_unlabeled():
    sl = SAMLabels(np.array([-1, -1, -1], dtype=np.int8), ["healthy", "A"])
    sl.clean_labels()
    np.testing.assert_array_equal(sl.labels, [-1, -1, -1])
    assert sl.label_names == ["healthy"]


def test_clean_all_healthy():
    sl = SAMLabels(np.array([0, 0, 0], dtype=np.int8), ["healthy", "A", "B"])
    sl.clean_labels()
    np.testing.assert_array_equal(sl.labels, [0, 0, 0])
    assert sl.label_names == ["healthy"]


def test_clean_single_class():
    sl = SAMLabels(np.array([1, 1, 1], dtype=np.int8), ["healthy", "Only"])
    sl.clean_labels()
    np.testing.assert_array_equal(sl.labels, [1, 1, 1])
    assert sl.label_names == ["healthy", "Only"]


def test_clean_multigap_sorted_remap():
    sl = SAMLabels(np.array([-1, 2, 7, 2, -1, 7], dtype=np.int8),
                   ["healthy", "X", "A", "Y", "Z", "W", "V", "B"])
    sl.clean_labels()
    np.testing.assert_array_equal(sl.labels, [-1, 1, 2, 1, -1, 2])
    assert sl.label_names == ["healthy", "A", "B"]


def test_clean_no_label_names():
    sl = SAMLabels(np.array([3, 5, 3], dtype=np.int8), [])
    sl.clean_labels()
    np.testing.assert_array_equal(sl.labels, [1, 2, 1])
    assert sl.label_names == ["healthy", "label3", "label5"]


def test_clean_keeps_healthy_at_zero():
    sl = SAMLabels(np.array([0, 4, 0, 4], dtype=np.int8),
                   ["healthy", "a", "b", "c", "Z"])
    sl.clean_labels()
    np.testing.assert_array_equal(sl.labels, [0, 1, 0, 1])
    assert sl.label_names == ["healthy", "Z"]


def test_clean_verify_integrity_after():
    sl = SAMLabels(np.array([-1, 0, 9, 0, 9, -1], dtype=np.int8),
                   ["healthy"] + [f"x{i}" for i in range(10)])
    sl.clean_labels()
    sl.verify_integrity()


def test_clean_large_random():
    rng = np.random.default_rng(42)
    labels = np.concatenate([
        np.full(20, -1, dtype=np.int8),
        rng.integers(0, 50, size=80, dtype=np.int8),
    ])
    rng.shuffle(labels)
    sl = SAMLabels(labels, ["healthy"] + [f"L{i}" for i in range(50)])
    sl.clean_labels()
    sl.verify_integrity()
    unique = np.unique(sl.labels)
    assert -1 in unique
    expected = [-1] + list(range(len(np.setdiff1d(unique, [-1]))))
    np.testing.assert_array_equal(unique, expected)
    for v in sl.labels:
        if v > 0:
            assert sl.label_names[v] != "healthy"


# ── basic properties & dunders ──────────────────────────────────────────────

def test_is_labeled_true():
    sl = SAMLabels(np.array([0, 1, -1], dtype=np.int8), ["healthy", "Defect"])
    assert sl.is_labeled()


def test_is_labeled_false():
    sl = SAMLabels.create_unlabeled(5)
    assert not sl.is_labeled()


def test_max_label():
    sl = SAMLabels(np.array([0, 3, 1, 5, -1], dtype=np.int8),
                   ["healthy", "a", "b", "c", "d", "e"])
    assert sl.max_label() == 5


def test_max_label_all_unlabeled():
    sl = SAMLabels.create_unlabeled(3)
    assert sl.max_label() == -1


def test_healthy_mask():
    sl = SAMLabels(np.array([0, 1, 0, -1, 2], dtype=np.int8),
                   ["healthy", "A", "B"])
    np.testing.assert_array_equal(sl.healthy_mask(),
                                  [True, False, True, False, False])


def test_len_dunder():
    sl = SAMLabels(np.array([0, 1, 2], dtype=np.int8), ["healthy", "A", "B"])
    assert len(sl) == 3


def test_getitem():
    sl = SAMLabels(np.array([0, 1, 2, -1], dtype=np.int8),
                   ["healthy", "A", "B"])
    assert sl[0] == 0
    assert sl[1] == 1
    assert sl[3] == -1


def test_iter():
    sl = SAMLabels(np.array([0, 1, 2], dtype=np.int8), ["healthy", "A", "B"])
    assert list(sl) == [0, 1, 2]


def test_label_name_method():
    sl = SAMLabels(np.array([0, 1, 2], dtype=np.int8), ["healthy", "A", "B"])
    assert sl.label_name(0) == "healthy"
    assert sl.label_name(1) == "A"
    assert sl.label_name(2) == "B"


def test_label_name_unlabeled():
    sl = SAMLabels(np.array([0, -1, 1], dtype=np.int8), ["healthy", "Defect"])
    assert sl.label_name(1) == "unlabeled"


def test_label_name_fallback():
    sl = SAMLabels(np.array([0, 5], dtype=np.int8), [])
    assert sl.label_name(1) == "label5"


def test_label_name_val():
    sl = SAMLabels(np.array([0, 1, 2, -1], dtype=np.int8),
                   ["healthy", "A", "B"])
    assert sl.label_name_val(0) == "healthy"
    assert sl.label_name_val(1) == "A"
    assert sl.label_name_val(2) == "B"
    assert sl.label_name_val(-1) == "unlabeled"


class TestCopy:
    def test_copy_creates_independent(self):
        sl = SAMLabels(np.array([0, 1, 2, -1], dtype=np.int8),
                       ["healthy", "A", "B"])
        c = sl.copy()
        np.testing.assert_array_equal(c.labels, sl.labels)
        assert c.label_names == sl.label_names
        c.labels[0] = -1
        assert sl.labels[0] != c.labels[0]

    def test_copy_label_names_independent(self):
        sl = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "Defect"])
        c = sl.copy()
        c.label_names[1] = "Modified"
        assert sl.label_names[1] == "Defect"


# ── merge ───────────────────────────────────────────────────────────────────

class TestMergeBasic:
    def test_different_names(self):
        a = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "disease_A"])
        b = SAMLabels(np.array([0, 2], dtype=np.int8),
                      ["healthy", "disease_B", "disease_C"])
        m = a.merge(b)
        np.testing.assert_array_equal(m.labels, [0, 1, 0, 2])
        assert m.label_names == ["healthy", "disease_A", "disease_C"]

    def test_overlapping_names_same_values(self):
        a = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "shared"])
        b = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "shared"])
        m = a.merge(b)
        np.testing.assert_array_equal(m.labels, [0, 1, 0, 1])
        assert m.label_names == ["healthy", "shared"]

    def test_overlapping_names_different_values(self):
        a = SAMLabels(np.array([1], dtype=np.int8), ["healthy", "common"])
        b = SAMLabels(np.array([2], dtype=np.int8), ["healthy", "X", "common"])
        m = a.merge(b)
        np.testing.assert_array_equal(m.labels, [1, 1])
        assert m.label_names == ["healthy", "common"]


class TestMergeEdgeCases:
    def test_both_unlabeled(self):
        a = SAMLabels.create_unlabeled(3)
        b = SAMLabels.create_unlabeled(2)
        m = a.merge(b)
        np.testing.assert_array_equal(m.labels, [-1, -1, -1, -1, -1])
        assert m.label_names == ["healthy"]

    def test_self_unlabeled_other_labeled(self):
        a = SAMLabels.create_unlabeled(2)
        b = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "Defect"])
        m = a.merge(b)
        np.testing.assert_array_equal(m.labels, [-1, -1, 0, 1])
        assert m.label_names == ["healthy", "Defect"]

    def test_self_labeled_other_unlabeled(self):
        a = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "Defect"])
        b = SAMLabels.create_unlabeled(3)
        m = a.merge(b)
        np.testing.assert_array_equal(m.labels, [0, 1, -1, -1, -1])
        assert m.label_names == ["healthy", "Defect"]

    def test_only_healthy(self):
        a = SAMLabels(np.array([0, 0], dtype=np.int8), ["healthy"])
        b = SAMLabels(np.array([0, 0, 0], dtype=np.int8), ["healthy"])
        m = a.merge(b)
        np.testing.assert_array_equal(m.labels, [0, 0, 0, 0, 0])
        assert m.label_names == ["healthy"]

    def test_empty_label_names(self):
        a = SAMLabels(np.array([1, 1], dtype=np.int8), [])
        b = SAMLabels(np.array([2, 2], dtype=np.int8), [])
        m = a.merge(b)
        np.testing.assert_array_equal(m.labels, [1, 1, 2, 2])
        assert m.label_names == ["healthy", "label1", "label2"]


class TestMergeImmutability:
    def test_originals_unchanged(self):
        a = SAMLabels(np.array([-1, 0, 2], dtype=np.int8),
                      ["healthy", "A", "B"])
        b = SAMLabels(np.array([0, 3], dtype=np.int8),
                      ["healthy", "X", "Y", "Z"])
        a_labels_before = a.labels.copy()
        a_names_before = a.label_names.copy()
        b_labels_before = b.labels.copy()
        b_names_before = b.label_names.copy()
        a.merge(b)
        np.testing.assert_array_equal(a.labels, a_labels_before)
        assert a.label_names == a_names_before
        np.testing.assert_array_equal(b.labels, b_labels_before)
        assert b.label_names == b_names_before


class TestMergeLength:
    @pytest.mark.parametrize("n1,n2", [(1, 1), (5, 5), (7, 3)])
    def test_result_length(self, n1, n2):
        a = SAMLabels(np.zeros(n1, dtype=np.int8), ["healthy"])
        b = SAMLabels(np.zeros(n2, dtype=np.int8), ["healthy"])
        m = a.merge(b)
        assert len(m) == len(a) + len(b)

    def test_result_length_exact(self):
        a = SAMLabels(np.array([-1, 0, 1, 2], dtype=np.int8),
                      ["healthy", "A", "B"])
        b = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "C"])
        m = a.merge(b)
        assert len(m) == 6


class TestMergeOrder:
    def test_order_preserved(self):
        a = SAMLabels(np.array([2, 1, 0], dtype=np.int8),
                      ["healthy", "First", "Second"])
        b = SAMLabels(np.array([0, 3, 1], dtype=np.int8),
                      ["healthy", "Third", "X", "Fourth"])
        m = a.merge(b)
        np.testing.assert_array_equal(m.labels, [2, 1, 0, 0, 4, 3])
        assert m.label_names == ["healthy", "First", "Second", "Third",
                                 "Fourth"]

    def test_order_self_before_other(self):
        a = SAMLabels(np.array([1], dtype=np.int8), ["healthy", "A"])
        b = SAMLabels(np.array([1], dtype=np.int8), ["healthy", "B"])
        m = a.merge(b)
        np.testing.assert_array_equal(m.labels, [1, 2])
        assert m.label_names == ["healthy", "A", "B"]


class TestMergeNameResolution:
    def test_same_name_different_old_values_map_same(self):
        a = SAMLabels(np.array([5], dtype=np.int8),
                      ["healthy", "X", "Y", "Z", "W", "match"])
        b = SAMLabels(np.array([3], dtype=np.int8),
                      ["healthy", "P", "Q", "match"])
        m = a.merge(b)
        np.testing.assert_array_equal(m.labels, [1, 1])
        assert m.label_names == ["healthy", "match"]

    def test_name_matching_handles_labels_without_names(self):
        a = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "Known"])
        b = SAMLabels(np.array([0, 1, 2], dtype=np.int8), [])
        m = a.merge(b)
        np.testing.assert_array_equal(m.labels, [0, 1, 0, 2, 3])
        assert m.label_names == ["healthy", "Known", "label1", "label2"]

    def test_merge_cleans_result(self):
        a = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "Keep"])
        b = SAMLabels(np.array([0, 1], dtype=np.int8),
                      ["healthy", "Used", "Unused"])
        m = a.merge(b)
        assert "Unused" not in m.label_names
        np.testing.assert_array_equal(m.labels, [0, 1, 0, 2])
        assert m.label_names == ["healthy", "Keep", "Used"]


class TestMergeVerifyIntegrity:
    def test_merge_result_passes_verify(self):
        a = SAMLabels(np.array([0, 1, 2, -1], dtype=np.int8),
                      ["healthy", "Alpha", "Beta"])
        b = SAMLabels(np.array([-1, 0, 2, 3], dtype=np.int8),
                      ["healthy", "Gamma", "Beta", "Delta"])
        m = a.merge(b)
        m.verify_integrity()

    def test_merge_random_pairs(self):
        rng = np.random.default_rng(123)
        for _ in range(20):
            n1 = rng.integers(1, 30)
            n2 = rng.integers(1, 30)
            max_l1 = rng.integers(0, 8)
            max_l2 = rng.integers(0, 8)
            labs1 = rng.integers(-1, max_l1 + 1, size=n1, dtype=np.int8)
            labs2 = rng.integers(-1, max_l2 + 1, size=n2, dtype=np.int8)
            names1 = ["healthy"] + [f"a_{i}" for i in range(max_l1)]
            names2 = ["healthy"] + [f"b_{i}" for i in range(max_l2)]
            a = SAMLabels(labs1, names1)
            b = SAMLabels(labs2, names2)
            m = a.merge(b)
            m.verify_integrity()
            assert len(m) == n1 + n2


class TestMergeCleanComposition:
    def test_clean_merge_clean(self):
        a = SAMLabels(np.array([-1, 0, 5, 5], dtype=np.int8),
                      ["healthy", "u1", "u2", "u3", "u4", "A"])
        b = SAMLabels(np.array([0, 3, 3], dtype=np.int8),
                      ["healthy", "x1", "x2", "B"])
        a.clean_labels()
        b.clean_labels()
        m = a.merge(b)
        m.verify_integrity()
        np.testing.assert_array_equal(m.labels, [-1, 0, 1, 1, 0, 2, 2])
        assert m.label_names == ["healthy", "A", "B"]


class TestMergeLabels:
    def test_merge_two_same_as_method(self):
        a = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "A"])
        b = SAMLabels(np.array([0, 2], dtype=np.int8), ["healthy", "X", "B"])
        via_method = a.merge(b)
        via_function = merge_labels([a, b])
        np.testing.assert_array_equal(via_function.labels, via_method.labels)
        assert via_function.label_names == via_method.label_names

    def test_merge_three(self):
        a = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "A"])
        b = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "B"])
        c = SAMLabels(np.array([0, 2], dtype=np.int8), ["healthy", "C", "D"])
        m = merge_labels([a, b, c])
        np.testing.assert_array_equal(m.labels, [0, 1, 0, 2, 0, 3])
        assert m.label_names == ["healthy", "A", "B", "D"]
        assert len(m) == len(a) + len(b) + len(c)

    def test_merge_three_overlapping_names(self):
        a = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "Shared"])
        b = SAMLabels(np.array([1], dtype=np.int8), ["healthy", "Shared"])
        c = SAMLabels(np.array([3], dtype=np.int8),
                      ["healthy", "X", "Y", "Shared"])
        m = merge_labels([a, b, c])
        np.testing.assert_array_equal(m.labels, [0, 1, 1, 1])
        assert m.label_names == ["healthy", "Shared"]

    def test_merge_five(self):
        instances = []
        for i in range(5):
            instances.append(SAMLabels(
                np.array([0, i + 1], dtype=np.int8),
                ["healthy"] + [f"L{j}" for j in range(i + 1)] + [f"U{i}"]
            ))
        m = merge_labels(instances)
        m.verify_integrity()
        assert len(m) == sum(len(inst) for inst in instances)

    def test_merge_all_unlabeled(self):
        instances = [SAMLabels.create_unlabeled(n) for n in [2, 3, 1]]
        m = merge_labels(instances)
        np.testing.assert_array_equal(m.labels, [-1, -1, -1, -1, -1, -1])
        assert m.label_names == ["healthy"]

    def test_merge_preserves_order(self):
        a = SAMLabels(np.array([0], dtype=np.int8), ["healthy"])
        b = SAMLabels(np.array([1], dtype=np.int8), ["healthy", "B"])
        c = SAMLabels(np.array([2], dtype=np.int8), ["healthy", "X", "C"])
        m = merge_labels([a, b, c])
        np.testing.assert_array_equal(m.labels, [0, 1, 2])
        assert m.label_names == ["healthy", "B", "C"]

    def test_merge_case_insensitive(self):
        a = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "Defect"])
        b = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "defect"])
        c = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "DEFECT"])
        m = merge_labels([a, b, c])
        np.testing.assert_array_equal(m.labels, [0, 1, 0, 1, 0, 1])
        assert m.label_names == ["healthy", "Defect"]

    def test_merge_originals_unchanged(self):
        a = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "A"])
        b = SAMLabels(np.array([0, 2], dtype=np.int8), ["healthy", "X", "B"])
        c = SAMLabels(np.array([0, 3], dtype=np.int8),
                      ["healthy", "Y", "Z", "C"])
        a_before = (a.labels.copy(), a.label_names.copy())
        b_before = (b.labels.copy(), b.label_names.copy())
        c_before = (c.labels.copy(), c.label_names.copy())
        merge_labels([a, b, c])
        np.testing.assert_array_equal(a.labels, a_before[0])
        assert a.label_names == a_before[1]
        np.testing.assert_array_equal(b.labels, b_before[0])
        assert b.label_names == b_before[1]
        np.testing.assert_array_equal(c.labels, c_before[0])
        assert c.label_names == c_before[1]

    def test_merge_random_many(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            n_instances = rng.integers(3, 7)
            instances = []
            for _ in range(n_instances):
                n = rng.integers(2, 20)
                max_l = rng.integers(0, 6)
                labs = rng.integers(-1, max_l + 1, size=n, dtype=np.int8)
                names = ["healthy"] + [f"lab_{i}" for i in range(max_l)]
                instances.append(SAMLabels(labs, names))
            m = merge_labels(instances)
            m.verify_integrity()
            assert len(m) == sum(len(inst) for inst in instances)

    def test_merge_too_few_instances(self):
        with pytest.raises(ValueError, match="At least two"):
            merge_labels([SAMLabels.create_unlabeled(3)])


class TestToBinary:
    def test_binary_by_int(self):
        sl = SAMLabels(np.array([0, 1, 2, -1], dtype=np.int8),
                       ["healthy", "A", "B"])
        b = sl.to_binary(1)
        np.testing.assert_array_equal(b, [0, 1, 0, -1])

    def test_binary_by_str(self):
        sl = SAMLabels(np.array([0, 2, -1, 2], dtype=np.int8),
                       ["healthy", "X", "target"])
        b = sl.to_binary("target")
        np.testing.assert_array_equal(b, [0, 1, -1, 1])

    def test_binary_case_insensitive(self):
        sl = SAMLabels(np.array([0, 1, 1], dtype=np.int8), ["healthy", "Defect"])
        b = sl.to_binary("DEFECT")
        np.testing.assert_array_equal(b, [0, 1, 1])

    def test_binary_positive_is_healthy(self):
        sl = SAMLabels(np.array([0, 0, 1], dtype=np.int8), ["healthy", "A"])
        b = sl.to_binary(0)
        np.testing.assert_array_equal(b, [1, 1, 0])

    def test_binary_unknown_str_returns_all_zeros(self):
        sl = SAMLabels(np.array([0, 1, 2], dtype=np.int8), ["healthy", "A", "B"])
        b = sl.to_binary("not_found")
        np.testing.assert_array_equal(b, [0, 0, 0])

    def test_binary_bad_type_raises(self):
        sl = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "A"])
        with pytest.raises(TypeError):
            sl.to_binary(1.5)


class TestClassDistribution:
    def test_basic(self):
        sl = SAMLabels(np.array([0, 1, 1, -1, 0], dtype=np.int8),
                       ["healthy", "Defect"])
        dist = sl.class_distribution()
        assert dist == {"unlabeled": 1, "healthy": 2, "Defect": 2}

    def test_all_unlabeled(self):
        sl = SAMLabels(np.array([-1, -1], dtype=np.int8), ["healthy"])
        dist = sl.class_distribution()
        assert dist == {"unlabeled": 2}

    def test_no_label_names(self):
        sl = SAMLabels(np.array([0, 1], dtype=np.int8), [])
        dist = sl.class_distribution()
        assert dist == {"healthy": 1, "label1": 1}


class TestNameToValue:
    def test_healthy(self):
        sl = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "Defect"])
        assert sl.name_to_value("healthy") == 0
        assert sl.name_to_value("HEALTHY") == 0
        assert sl.name_to_value("Healthy") == 0

    def test_registered_name(self):
        sl = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "Defect"])
        assert sl.name_to_value("Defect") == 1
        assert sl.name_to_value("defect") == 1

    def test_unlabeled(self):
        sl = SAMLabels(np.array([0, -1], dtype=np.int8), ["healthy"])
        assert sl.name_to_value("unlabeled") == -1
        assert sl.name_to_value("UNLABELED") == -1

    def test_unknown_returns_unlabeled(self):
        sl = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "Defect"])
        assert sl.name_to_value("nonexistent") == -1

    def test_empty_string(self):
        sl = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "Defect"])
        assert sl.name_to_value("") == -1

    def test_no_label_names(self):
        sl = SAMLabels(np.array([0, 1], dtype=np.int8), [])
        assert sl.name_to_value("healthy") == 0
        assert sl.name_to_value("label1") == 1


class TestHasName:
    def test_known(self):
        sl = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "Defect"])
        assert sl.has_name("Defect")
        assert sl.has_name("defect")
        assert sl.has_name("healthy")
        assert sl.has_name("unlabeled")

    def test_unknown(self):
        sl = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "Defect"])
        assert not sl.has_name("nonexistent")

    def test_no_label_names(self):
        sl = SAMLabels(np.array([0, 1], dtype=np.int8), [])
        assert sl.has_name("healthy")
        assert sl.has_name("unlabeled")
        assert not sl.has_name("Defect")


class TestMask:
    def test_by_int(self):
        sl = SAMLabels(np.array([0, 1, 2, 1, -1], dtype=np.int8),
                       ["healthy", "A", "B"])
        m = sl.mask(1)
        np.testing.assert_array_equal(m, [False, True, False, True, False])

    def test_by_str(self):
        sl = SAMLabels(np.array([0, 1, 2, 1, -1], dtype=np.int8),
                       ["healthy", "A", "B"])
        m = sl.mask("A")
        np.testing.assert_array_equal(m, [False, True, False, True, False])

    def test_by_str_case_insensitive(self):
        sl = SAMLabels(np.array([0, 1, 2], dtype=np.int8),
                       ["healthy", "Defect", "B"])
        m = sl.mask("DEFECT")
        np.testing.assert_array_equal(m, [False, True, False])

    def test_healthy(self):
        sl = SAMLabels(np.array([0, 1, 0], dtype=np.int8), ["healthy", "A"])
        m = sl.mask(0)
        np.testing.assert_array_equal(m, [True, False, True])

    def test_unlabeled(self):
        sl = SAMLabels(np.array([-1, 0, -1], dtype=np.int8), ["healthy"])
        m = sl.mask(-1)
        np.testing.assert_array_equal(m, [True, False, True])

    def test_unlabeled_by_str(self):
        sl = SAMLabels(np.array([-1, 0, -1], dtype=np.int8), ["healthy"])
        m = sl.mask("unlabeled")
        np.testing.assert_array_equal(m, [True, False, True])


class TestLabeledUnlabeledMask:
    def test_labeled(self):
        sl = SAMLabels(np.array([-1, 0, 1, -1, 2], dtype=np.int8),
                       ["healthy", "A", "B"])
        np.testing.assert_array_equal(
            sl.labeled_mask(), [False, True, True, False, True])

    def test_unlabeled(self):
        sl = SAMLabels(np.array([-1, 0, 1, -1, 2], dtype=np.int8),
                       ["healthy", "A", "B"])
        np.testing.assert_array_equal(
            sl.unlabeled_mask(), [True, False, False, True, False])

    def test_all_labeled(self):
        sl = SAMLabels(np.array([0, 1, 2], dtype=np.int8), ["healthy", "A", "B"])
        assert np.all(sl.labeled_mask())
        assert not np.any(sl.unlabeled_mask())

    def test_all_unlabeled(self):
        sl = SAMLabels.create_unlabeled(5)
        assert not np.any(sl.labeled_mask())
        assert np.all(sl.unlabeled_mask())


class TestNumClasses:
    def test_three_classes(self):
        sl = SAMLabels(np.array([0, 1, 2, -1], dtype=np.int8),
                       ["healthy", "A", "B"])
        assert sl.num_classes == 3

    def test_all_unlabeled(self):
        sl = SAMLabels.create_unlabeled(5)
        assert sl.num_classes == 0

    def test_only_healthy(self):
        sl = SAMLabels(np.array([0, 0, 0], dtype=np.int8), ["healthy"])
        assert sl.num_classes == 1

    def test_no_label_names(self):
        sl = SAMLabels(np.array([0, 1, 2, 0], dtype=np.int8), [])
        assert sl.num_classes == 3


class TestUniqueLabels:
    def test_mixed(self):
        sl = SAMLabels(np.array([0, 2, 1, -1, 0, 2], dtype=np.int8),
                       ["healthy", "A", "B"])
        np.testing.assert_array_equal(sl.unique_labels, [0, 1, 2])

    def test_all_unlabeled(self):
        sl = SAMLabels.create_unlabeled(3)
        assert len(sl.unique_labels) == 0

    def test_healthy_only(self):
        sl = SAMLabels(np.array([0, 0, 0], dtype=np.int8), ["healthy"])
        np.testing.assert_array_equal(sl.unique_labels, [0])

    def test_sorted(self):
        sl = SAMLabels(np.array([3, 0, 1, 5], dtype=np.int8),
                       ["healthy", "a", "b", "c", "d", "e"])
        np.testing.assert_array_equal(sl.unique_labels, [0, 1, 3, 5])


class TestToOneHot:
    def test_basic(self):
        sl = SAMLabels(np.array([0, 1, 2, 0], dtype=np.int8),
                       ["healthy", "A", "B"])
        oh = sl.to_one_hot()
        assert oh.shape == (4, 3)
        assert oh.dtype == np.float32
        np.testing.assert_array_equal(oh[0], [1, 0, 0])
        np.testing.assert_array_equal(oh[1], [0, 1, 0])
        np.testing.assert_array_equal(oh[2], [0, 0, 1])
        np.testing.assert_array_equal(oh[3], [1, 0, 0])

    def test_unlabeled_all_zero(self):
        sl = SAMLabels(np.array([-1, 0, 1, -1], dtype=np.int8),
                       ["healthy", "A"])
        oh = sl.to_one_hot()
        assert oh.shape == (4, 2)
        np.testing.assert_array_equal(oh[0], [0, 0])
        np.testing.assert_array_equal(oh[3], [0, 0])

    def test_no_classes(self):
        sl = SAMLabels.create_unlabeled(3)
        oh = sl.to_one_hot()
        assert oh.shape == (3, 1)
        np.testing.assert_array_equal(oh, np.zeros((3, 1), dtype=np.float32))

    def test_only_healthy(self):
        sl = SAMLabels(np.array([0, 0, 0], dtype=np.int8), ["healthy"])
        oh = sl.to_one_hot()
        assert oh.shape == (3, 1)
        np.testing.assert_array_equal(oh, np.ones((3, 1), dtype=np.float32))


class TestStratifiedSplit:
    def test_preserves_class_proportion_approximately(self):
        sl = SAMLabels(np.array([0] * 30 + [1] * 10 + [2] * 10, dtype=np.int8),
                       ["healthy", "A", "B"])
        train, test = sl.stratified_split(test_size=0.2, random_state=42)
        assert len(test) == 10
        assert len(train) == 40

    def test_each_class_has_at_least_one_test(self):
        sl = SAMLabels(np.array([0] * 5 + [1] * 5 + [2] * 5, dtype=np.int8),
                       ["healthy", "A", "B"])
        train, test = sl.stratified_split(test_size=0.2, random_state=42)
        test_labels = sl.labels[test]
        assert 0 in test_labels
        assert 1 in test_labels
        assert 2 in test_labels

    def test_split_disjoint(self):
        sl = SAMLabels(np.array([0] * 20 + [1] * 10, dtype=np.int8),
                       ["healthy", "A"])
        train, test = sl.stratified_split(test_size=0.3, random_state=42)
        assert len(np.intersect1d(train, test)) == 0
        assert len(train) + len(test) == len(sl)

    def test_unlabeled_distributed(self):
        sl = SAMLabels(np.array([-1] * 10 + [0] * 10 + [1] * 10, dtype=np.int8),
                       ["healthy", "A"])
        train, test = sl.stratified_split(test_size=0.3, random_state=7)
        test_labels = sl.labels[test]
        assert -1 in test_labels

    def test_all_unlabeled(self):
        sl = SAMLabels.create_unlabeled(10)
        train, test = sl.stratified_split(test_size=0.3, random_state=42)
        assert len(test) >= 1
        assert len(train) + len(test) == 10
        np.testing.assert_array_equal(sl.labels[train], -1)
        np.testing.assert_array_equal(sl.labels[test], -1)

    def test_reproducible(self):
        sl = SAMLabels(np.array([0] * 10 + [1] * 10, dtype=np.int8),
                       ["healthy", "A"])
        t1a, t1t = sl.stratified_split(test_size=0.3, random_state=42)
        t2a, t2t = sl.stratified_split(test_size=0.3, random_state=42)
        np.testing.assert_array_equal(t1a, t2a)
        np.testing.assert_array_equal(t1t, t2t)

    def test_bad_test_size_raises(self):
        sl = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "A"])
        with pytest.raises(ValueError):
            sl.stratified_split(test_size=0)
        with pytest.raises(ValueError):
            sl.stratified_split(test_size=1)

    def test_class_with_one_sample(self):
        sl = SAMLabels(np.array([0] * 10 + [1] * 10 + [2], dtype=np.int8),
                       ["healthy", "A", "B"])
        train, test = sl.stratified_split(test_size=0.3, random_state=42)
        test_labels = sl.labels[test]
        assert 2 in test_labels


class TestTake:
    def test_subset(self):
        sl = SAMLabels(np.array([0, 1, 2, -1, 1], dtype=np.int8),
                       ["healthy", "A", "B"])
        sub = sl.take(np.array([0, 2, 4]))
        np.testing.assert_array_equal(sub.labels, [0, 2, 1])
        assert sub.label_names == ["healthy", "A", "B"]

    def test_empty_raises(self):
        sl = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "A"])
        with pytest.raises(ValueError, match="Cannot take an empty"):
            sl.take(np.array([], dtype=np.int64))

    def test_does_not_modify_original(self):
        sl = SAMLabels(np.array([0, 1, 2], dtype=np.int8), ["healthy", "A", "B"])
        labels_before = sl.labels.copy()
        sl.take(np.array([0, 2]))
        np.testing.assert_array_equal(sl.labels, labels_before)


class TestRelabel:
    def test_int_to_int_merge(self):
        sl = SAMLabels(np.array([0, 1, 2, 1, 2], dtype=np.int8),
                       ["healthy", "A", "B"])
        sl.relabel({2: 1})
        np.testing.assert_array_equal(sl.labels, [0, 1, 1, 1, 1])
        assert sl.label_names == ["healthy", "A"]

    def test_int_to_str_rename(self):
        sl = SAMLabels(np.array([0, 1, 2], dtype=np.int8),
                       ["healthy", "Old", "B"])
        sl.relabel({1: "Renamed"})
        np.testing.assert_array_equal(sl.labels, [0, 1, 2])
        assert sl.label_names == ["healthy", "Renamed", "B"]

    def test_int_to_new_str(self):
        sl = SAMLabels(np.array([0, 1, 0, 1], dtype=np.int8), ["healthy", "label1"])
        sl.relabel({1: "Crack"})
        assert sl.label_names == ["healthy", "Crack"]

    def test_str_to_int_merge(self):
        sl = SAMLabels(np.array([0, 1, 2], dtype=np.int8),
                       ["healthy", "A", "B"])
        sl.relabel({"B": 1})
        np.testing.assert_array_equal(sl.labels, [0, 1, 1])
        assert sl.label_names == ["healthy", "A"]

    def test_str_to_str_rename(self):
        sl = SAMLabels(np.array([0, 1, 2], dtype=np.int8),
                       ["healthy", "OldA", "OldB"])
        sl.relabel({"OldA": "NewA", "OldB": "NewB"})
        assert sl.label_names == ["healthy", "NewA", "NewB"]

    def test_str_case_insensitive(self):
        sl = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "Defect"])
        sl.relabel({"DEFECT": "Flaw"})
        assert sl.label_names == ["healthy", "Flaw"]

    def test_multiple_mappings(self):
        sl = SAMLabels(np.array([0, 1, 2, 3], dtype=np.int8),
                       ["healthy", "A", "B", "C"])
        sl.relabel({1: "X", 2: 1})
        np.testing.assert_array_equal(sl.labels, [0, 1, 1, 2])
        assert sl.label_names == ["healthy", "X", "C"]

    def test_unknown_key_raises(self):
        sl = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "A"])
        with pytest.raises(KeyError, match="not found"):
            sl.relabel({"NotThere": 0})

    def test_relabel_cleans_result(self):
        sl = SAMLabels(np.array([0, 1, 2, 1], dtype=np.int8),
                       ["healthy", "A", "B"])
        sl.relabel({2: 0})
        np.testing.assert_array_equal(sl.labels, [0, 1, 0, 1])
        assert sl.label_names == ["healthy", "A"]

    def test_relabel_to_new_name_same_value(self):
        sl = SAMLabels(np.array([0, 1, 1, 0], dtype=np.int8), ["healthy", "Old"])
        sl.relabel({1: "New"})
        np.testing.assert_array_equal(sl.labels, [0, 1, 1, 0])
        assert sl.label_names == ["healthy", "New"]

    def test_relabel_empty_mapping(self):
        sl = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "A"])
        labels_before = sl.labels.copy()
        sl.relabel({})
        np.testing.assert_array_equal(sl.labels, labels_before)
        assert sl.label_names == ["healthy", "A"]


class TestDictSerialization:
    def test_roundtrip(self):
        sl = SAMLabels(np.array([0, 1, 2, -1, 0], dtype=np.int8),
                       ["healthy", "A", "B"])
        d = sl.to_dict()
        sl2 = SAMLabels.from_dict(d)
        np.testing.assert_array_equal(sl.labels, sl2.labels)
        assert sl.label_names == sl2.label_names

    def test_roundtrip_no_names(self):
        sl = SAMLabels(np.array([0, 1, 0], dtype=np.int8), [])
        d = sl.to_dict()
        sl2 = SAMLabels.from_dict(d)
        np.testing.assert_array_equal(sl.labels, sl2.labels)
        assert sl2.label_names == []

    def test_roundtrip_all_unlabeled(self):
        sl = SAMLabels.create_unlabeled(5)
        d = sl.to_dict()
        sl2 = SAMLabels.from_dict(d)
        np.testing.assert_array_equal(sl.labels, sl2.labels)
        assert sl.label_names == sl2.label_names

    def test_dict_structure(self):
        sl = SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "A"])
        d = sl.to_dict()
        assert "labels" in d
        assert "label_names" in d
        assert isinstance(d["labels"], list)
        assert d["labels"] == [0, 1]
        assert d["label_names"] == ["healthy", "A"]

    def test_from_dict_missing_names(self):
        sl = SAMLabels.from_dict({"labels": [0, 1, 0]})
        np.testing.assert_array_equal(sl.labels, [0, 1, 0])
        assert sl.label_names == ["healthy"]
