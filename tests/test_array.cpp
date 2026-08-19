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
