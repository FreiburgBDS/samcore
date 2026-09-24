#include <gtest/gtest.h>

#include <samcore/array.hpp>

using samcore::array2d;
using samcore::array3d;

TEST(array2d, Construction) {
    array2d<int> a(3, 4);
    EXPECT_EQ(a.rows(), 3);
    EXPECT_EQ(a.cols(), 4);
    EXPECT_EQ(a.size(), 12);
    EXPECT_FALSE(a.empty());
}

TEST(array2d, FillAndAccess) {
    array2d<int> a(2, 3, 7);
    EXPECT_EQ(a[1][2], 7);
    a[0][1] = 42;
    EXPECT_EQ(a.flat()[1], 42);
    EXPECT_EQ(a.data()[1], 42);
}

TEST(array2d, DefaultEmpty) {
    array2d<double> a;
    EXPECT_TRUE(a.empty());
    EXPECT_EQ(a.rows(), 0);
}

TEST(array2d, RowSpanMutability) {
    array2d<int> a(2, 2, 0);
    auto row = a[1];
    row[0] = 5;
    EXPECT_EQ(a[1][0], 5);
}

TEST(array2d, BufferSizeMismatchThrows) {
    EXPECT_THROW(array2d<int>(2, 2, std::vector<int>(3)), std::invalid_argument);
}

TEST(array2d, Equality) {
    array2d<int> a(2, 2, 1);
    array2d<int> b(2, 2, 1);
    array2d<int> c(2, 2, 2);
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(array2d, NonOwningViewMetadata) {
    int ext[6] = {1, 2, 3, 4, 5, 6};
    array2d<int> v(ext, 2, 3, samcore::non_owning);
    EXPECT_TRUE(v.is_view());
    EXPECT_EQ(v.rows(), 2);
    EXPECT_EQ(v.cols(), 3);
    EXPECT_EQ(v.size(), 6);
    EXPECT_FALSE(v.empty());
    EXPECT_EQ(v.flat().size(), 6);
    EXPECT_EQ(v.flat()[4], 5);
    EXPECT_EQ(v[1][2], 6);
    EXPECT_EQ(v.data(), ext);
}

TEST(array2d, NonOwningViewEquality) {
    int a_data[4] = {1, 2, 3, 4};
    int b_data[4] = {1, 2, 3, 4};
    int c_data[4] = {1, 2, 3, 5};
    array2d<int> a(a_data, 2, 2, samcore::non_owning);
    array2d<int> b(b_data, 2, 2, samcore::non_owning);
    array2d<int> c(c_data, 2, 2, samcore::non_owning);
    array2d<int> own(2, 2, 0);
    own[0][0] = 1;
    own[0][1] = 2;
    own[1][0] = 3;
    own[1][1] = 4;

    EXPECT_EQ(a, b);          // same contents, different memory
    EXPECT_NE(a, c);          // same shape, different contents
    EXPECT_EQ(a, own);        // view vs owning, same contents
    EXPECT_NE(c, own);
}

TEST(array2d, ViewCopyMaterializes) {
    int ext[4] = {1, 2, 3, 4};
    array2d<int> v(ext, 2, 2, samcore::non_owning);

    array2d<int> copy = v; // deep copy, not an alias
    EXPECT_FALSE(copy.is_view());
    EXPECT_EQ(copy, v);

    ext[0] = 99;
    EXPECT_EQ(v[0][0], 99);
    EXPECT_EQ(copy[0][0], 1);

    copy[1][1] = 42;
    EXPECT_EQ(ext[3], 4);

    array2d<int> assigned(1, 1, 0);
    assigned = v;
    EXPECT_FALSE(assigned.is_view());
    EXPECT_EQ(assigned.rows(), 2);
    EXPECT_EQ(assigned[0][0], 99);
}

TEST(array2d, ViewMutatorsThrow) {
    int ext[4] = {1, 2, 3, 4};
    array2d<int> v(ext, 2, 2, samcore::non_owning);
    EXPECT_THROW(v.resize(1, 1), std::logic_error);
    EXPECT_THROW(v.fill(0), std::logic_error);
    EXPECT_THROW((void)v.release(), std::logic_error);
    EXPECT_EQ(ext[0], 1); // untouched
}

TEST(array2d, MovedFromIsEmpty) {
    array2d<int> a(2, 2, 7);
    array2d<int> b(std::move(a));
    EXPECT_TRUE(a.empty());
    EXPECT_EQ(a.size(), 0);
    EXPECT_EQ(a.rows(), 0);
    EXPECT_FALSE(a.is_view());
    EXPECT_EQ(b.size(), 4);
    EXPECT_EQ(b[1][1], 7);
}

TEST(array2d, Release) {
    array2d<int> a(2, 3, 9);
    auto buf = a.release();
    EXPECT_EQ(buf.size(), 6);
    EXPECT_TRUE(a.empty());
}

TEST(array3d, Basic) {
    array3d<int> a(2, 3, 4);
    EXPECT_EQ(a.size(), 24);
    a.flat()[23] = 1;
    EXPECT_EQ(a.plane(1)[11], 1);
}
