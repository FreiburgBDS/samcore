#include <gtest/gtest.h>

#include <samcore/sam_labels.hpp>

using namespace samcore;

namespace {

sam_labels make_labels(std::vector<std::int8_t> vals) {
    return sam_labels(std::move(vals), {"healthy", "defect_a", "defect_b"});
}

} // namespace

TEST(sam_labels, Constants) {
    EXPECT_EQ(sam_labels::label_unlabeled, -1);
    EXPECT_EQ(sam_labels::label_healthy, 0);
    EXPECT_EQ(std::string(sam_labels::label_name_unlabeled), "unlabeled");
    EXPECT_EQ(std::string(sam_labels::label_name_healthy), "healthy");
}

TEST(sam_labels, EmptyThrows) {
    EXPECT_THROW(sam_labels(std::vector<std::int8_t>{}), std::invalid_argument);
}

TEST(sam_labels, FirstNameForcedToHealthy) {
    sam_labels l(std::vector<std::int8_t>{0}, {"foo"});
    EXPECT_EQ(l.label_names()[0], "healthy");
}

TEST(sam_labels, LabelNameResolution) {
    auto l = make_labels({-1, 0, 1, 2, 3, 4});
    EXPECT_EQ(l.label_name(0), "unlabeled");
    EXPECT_EQ(l.label_name(1), "healthy");
    EXPECT_EQ(l.label_name(2), "defect_a");
    EXPECT_EQ(l.label_name(3), "defect_b");
    EXPECT_EQ(l.label_name(4), "label3");
    EXPECT_EQ(l.label_name(5), "label4");
}

TEST(sam_labels, NameToValue) {
    auto l = make_labels({-1, 0, 1, 2, 2});
    EXPECT_EQ(l.name_to_value("unlabeled"), -1);
    EXPECT_EQ(l.name_to_value("healthy"), 0);
    EXPECT_EQ(l.name_to_value("DEFECT_A"), 1);
    EXPECT_EQ(l.name_to_value("defect_b"), 2);
    EXPECT_EQ(l.name_to_value("label2"), 2); // present in labels
    EXPECT_EQ(l.name_to_value("label9"), -1); // not present
    EXPECT_EQ(l.name_to_value("nope"), -1);
    EXPECT_EQ(l.name_to_value(""), -1);
}

TEST(sam_labels, HasName) {
    auto l = make_labels({-1, 0, 1, 2});
    EXPECT_TRUE(l.has_name("healthy"));
    EXPECT_TRUE(l.has_name("UNLABELED"));
    EXPECT_TRUE(l.has_name("Defect_A"));
    EXPECT_FALSE(l.has_name("defect_c"));
}

TEST(sam_labels, Masks) {
    auto l = make_labels({-1, 0, 1, 2});
    EXPECT_EQ(l.healthy_mask(), (std::vector<uint8_t>{0, 1, 0, 0}));
    EXPECT_EQ(l.mask(static_cast<std::int8_t>(1)),
              (std::vector<uint8_t>{0, 0, 1, 0}));
    EXPECT_EQ(l.mask("defect_b"), (std::vector<uint8_t>{0, 0, 0, 1}));
    EXPECT_EQ(l.labeled_mask(), (std::vector<uint8_t>{0, 1, 1, 1}));
    EXPECT_EQ(l.unlabeled_mask(), (std::vector<uint8_t>{1, 0, 0, 0}));
}

TEST(sam_labels, Integrity) {
    EXPECT_NO_THROW(make_labels({0, 1}).verify_integrity());
    EXPECT_THROW(make_labels({0, 1}).take({}).verify_integrity(),
                 std::invalid_argument);
    sam_labels bad({0, 1});
    bad.set_label_names({"x", "y"}); // "x" != healthy
    EXPECT_THROW(bad.verify_integrity(), std::invalid_argument);
    sam_labels dup({0, 1});
    dup.set_label_names({"healthy", "healthy"});
    EXPECT_THROW(dup.verify_integrity(), std::invalid_argument);
    sam_labels neg({-2, 0});
    EXPECT_THROW(neg.verify_integrity(), std::invalid_argument);
}

TEST(sam_labels, IsLabeledAndCounts) {
    EXPECT_FALSE(sam_labels::create_unlabeled(5).is_labeled());
    auto l = make_labels({-1, 0, 1, 2, 2, 2});
    EXPECT_TRUE(l.is_labeled());
    EXPECT_EQ(l.max_label(), 2);
    EXPECT_EQ(l.num_classes(), 3);
    EXPECT_EQ(l.unique_labels(), (std::vector<std::int8_t>{0, 1, 2}));
}

TEST(sam_labels, ToBinary) {
    auto l = make_labels({-1, 0, 1, 2, 2});
    EXPECT_EQ(l.to_binary(static_cast<std::int8_t>(2)),
              (std::vector<std::int8_t>{-1, 0, 0, 1, 1}));
    EXPECT_EQ(l.to_binary(std::string("defect_a")),
              (std::vector<std::int8_t>{-1, 0, 1, 0, 0}));
    EXPECT_EQ(l.to_binary(std::string("unlabeled")),
              (std::vector<std::int8_t>{1, 0, 0, 0, 0}));
}

TEST(sam_labels, ClassDistribution) {
    auto l = make_labels({-1, 0, 1, 2, 2});
    auto dist = l.class_distribution();
    EXPECT_EQ(dist.at("unlabeled"), 1);
    EXPECT_EQ(dist.at("healthy"), 1);
    EXPECT_EQ(dist.at("defect_a"), 1);
    EXPECT_EQ(dist.at("defect_b"), 2);
}

TEST(sam_labels, OneHot) {
    sam_labels l(std::vector<std::int8_t>{-1, 0, 1}, {"healthy", "a", "b"});
    auto oh = l.to_one_hot();
    ASSERT_EQ(oh.rows(), 3);
    ASSERT_EQ(oh.cols(), 2); // classes {0, 1}
    EXPECT_FLOAT_EQ(oh[0][0], 0.0f);
    EXPECT_FLOAT_EQ(oh[1][0], 1.0f);
    EXPECT_FLOAT_EQ(oh[2][1], 1.0f);
    auto empty = sam_labels::create_unlabeled(3);
    auto oh0 = empty.to_one_hot();
    EXPECT_EQ(oh0.cols(), 1);
}

TEST(sam_labels, CleanLabels) {
    auto l = make_labels({-1, 0, 3, 5});
    l.clean_labels();
    EXPECT_EQ(l.labels(), (std::vector<std::int8_t>{-1, 0, 1, 2}));
    EXPECT_EQ(l.label_names(), (std::vector<std::string>{"healthy", "label3", "label5"}));
    auto un = sam_labels::create_unlabeled(3);
    un.clean_labels();
    EXPECT_EQ(un.label_names(), (std::vector<std::string>{"healthy"}));
}

TEST(sam_labels, RelabelByName) {
    auto l = make_labels({-1, 0, 1, 2});
    l.relabel({{std::string("defect_a"), std::string("defect_a2")}});
    EXPECT_EQ(l.label_names()[1], "defect_a2");
    l.relabel({{std::string("defect_b"), std::string("defect_a2")}});
    EXPECT_EQ(l.labels(), (std::vector<std::int8_t>{-1, 0, 1, 1}));
    EXPECT_EQ(l.label_names(), (std::vector<std::string>{"healthy", "defect_a2"}));
}

TEST(sam_labels, RelabelByValue) {
    auto l = make_labels({-1, 0, 1, 2});
    l.relabel({{std::int64_t(2), std::int64_t(1)}});
    EXPECT_EQ(l.labels(), (std::vector<std::int8_t>{-1, 0, 1, 1}));
}

TEST(sam_labels, Merge) {
    sam_labels a({0, 1, -1}, {"healthy", "defect"});
    sam_labels b({0, 2}, {"healthy", "OTHER"});
    auto m = merge_labels({a, b});
    EXPECT_EQ(m.labels(), (std::vector<std::int8_t>{0, 1, -1, 0, 2}));
    // label value 2 has no registered name in b -> auto name "label2"
    EXPECT_EQ(m.label_names(),
              (std::vector<std::string>{"healthy", "defect", "label2"}));
}

TEST(sam_labels, MergeCaseInsensitive) {
    sam_labels a({0, 1}, {"healthy", "Defect"});
    sam_labels b({0, 1}, {"healthy", "defect"});
    auto m = merge_labels({a, b});
    EXPECT_EQ(m.labels(), (std::vector<std::int8_t>{0, 1, 0, 1}));
    EXPECT_EQ(m.label_names(), (std::vector<std::string>{"healthy", "Defect"}));
}

TEST(sam_labels, MergeRequiresTwo) {
    sam_labels a({0, 1}, {"healthy", "d"});
    EXPECT_THROW(merge_labels({a}), std::invalid_argument);
}

TEST(sam_labels, MergeByNameAcrossInstances) {
    // same name -> unified; unnamed value 5 -> auto name "label5"
    sam_labels a({0, 1}, {"healthy", "same"});
    sam_labels b({0, 5}, {"healthy", "same"});
    auto m = merge_labels({a, b});
    EXPECT_EQ(m.labels(), (std::vector<std::int8_t>{0, 1, 0, 2}));
    EXPECT_EQ(m.label_names(),
              (std::vector<std::string>{"healthy", "same", "label5"}));
}

TEST(sam_labels, Take) {
    auto l = make_labels({-1, 0, 1, 2});
    auto t = l.take({3, 1});
    EXPECT_EQ(t.labels(), (std::vector<std::int8_t>{2, 0}));
    EXPECT_THROW(l.take({}), std::invalid_argument);
}

TEST(sam_labels, CreateUnlabeled) {
    auto l = sam_labels::create_unlabeled(4);
    EXPECT_EQ(l.labels(), (std::vector<std::int8_t>{-1, -1, -1, -1}));
    EXPECT_EQ(l.label_names(), (std::vector<std::string>{"healthy"}));
    EXPECT_FALSE(l.is_labeled());
}

TEST(sam_labels, DictRoundTrip) {
    auto l = make_labels({-1, 0, 1});
    auto d = l.to_dict();
    auto l2 = sam_labels::from_dict(d);
    EXPECT_EQ(l2.labels(), l.labels());
    EXPECT_EQ(l2.label_names(), l.label_names());
}

TEST(sam_labels, Copy) {
    auto l = make_labels({-1, 0, 1});
    auto c = l.copy();
    EXPECT_EQ(c.labels(), l.labels());
    EXPECT_EQ(c.label_names(), l.label_names());
}

TEST(sam_labels, SetLabelsSanitizes) {
    sam_labels l(std::vector<std::int8_t>{1}, {"not_healthy"});
    EXPECT_EQ(l.label_names()[0], "healthy");
}
