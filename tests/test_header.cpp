#include <gtest/gtest.h>

#include <samcore/sam_header.hpp>

using namespace samcore;

TEST(sam_header, ConstructorAndDefaults) {
    sam_header h(100, 50, 2000, 100.0, 100, 1.5);
    EXPECT_EQ(h.scanspline, 100);
    EXPECT_EQ(h.nlines, 50);
    EXPECT_EQ(h.scanlen, 2000);
    EXPECT_EQ(h.samplerate, 100.0);
    EXPECT_EQ(h.tzero, 100);
    EXPECT_EQ(h.resolution, 1.5);
    EXPECT_FALSE(h.interpolated);
    EXPECT_TRUE(h.quality);
    EXPECT_EQ(h.downsample_factor, 1);
    EXPECT_TRUE(h.extra.empty());
}

TEST(sam_header, TimeAxis) {
    sam_header h(1, 1, 4, 1000.0, 100, 1.0);
    // linspace(tzero, tzero + scanlen/samplerate*1e3, scanlen)
    auto t = h.time();
    ASSERT_EQ(t.size(), 4);
    EXPECT_DOUBLE_EQ(t[0], 100.0);
    EXPECT_DOUBLE_EQ(t[3], 104.0);
}

TEST(sam_header, TimeAxisWithRange) {
    sam_header h(1, 1, 100, 100.0, 0, 1.0);
    auto t = h.time(10, 20);
    ASSERT_EQ(t.size(), 10);
    EXPECT_DOUBLE_EQ(t[0], 100.0); // 10 ns/sample
    EXPECT_DOUBLE_EQ(t[9], 200.0); // endpoint = tzero + end/samplerate*1e3
}

TEST(sam_header, Equality) {
    sam_header a(1, 1, 10, 1, 0, 1);
    sam_header b(1, 1, 10, 1, 0, 1);
    sam_header c(1, 1, 11, 1, 0, 1);
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(sam_header, HashStable) {
    sam_header a(1, 1, 10, 1, 0, 1);
    sam_header b(1, 1, 10, 1, 0, 1);
    EXPECT_EQ(a.hash(), b.hash());
}

TEST(sam_header, HashDistinguishesFields) {
    sam_header a(1, 1, 10, 1, 0, 1);
    sam_header b = a;
    b.nlines = 2;
    EXPECT_NE(a.hash(), b.hash());
}

TEST(sam_header, HashIncludesExtra) {
    sam_header a(1, 1, 10, 1, 0, 1);
    sam_header b = a;
    a.extra["gain"] = std::int64_t{3};
    EXPECT_NE(a.hash(), b.hash());
    sam_header c = a;
    EXPECT_EQ(a.hash(), c.hash());
}
