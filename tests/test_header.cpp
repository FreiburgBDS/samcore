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

TEST(sam_header, JsonRoundTrip) {
    sam_header h(64, 32, 512, 250.0, 42, 2.0, true, false,
                 "echo", "in", "through", "cell-1", 2,
                 {{"gain", 3.5}, {"notes", "[\"a\", 1]"}});
    auto j = h.to_json();
    auto h2 = sam_header::from_json(j);
    EXPECT_EQ(h2, h);
}

TEST(sam_header, JsonParsesBoolsAndInts) {
    auto j = nlohmann::json{{"version", "x"},  {"scanspline", 10},
                            {"nlines", 20},   {"scanlen", 100},
                            {"samplerate", 50}, {"tzero", 7},
                            {"resolution", 1.0}, {"interpolated", 1},
                            {"quality", 0}};
    auto h = sam_header::from_json(j);
    EXPECT_TRUE(h.interpolated);
    EXPECT_FALSE(h.quality);
}

TEST(sam_header, IgnoresLegacyKeys) {
    auto j = nlohmann::json{{"version", "x"},
                            {"scanspline", 1},
                            {"nlines", 1},
                            {"scanlen", 10},
                            {"samplerate", 1},
                            {"tzero", 0},
                            {"resolution", 1},
                            {"headerlen", 12345},
                            {"bytes_p_sample", 1}};
    auto h = sam_header::from_json(j);
    EXPECT_EQ(h.extra.count("headerlen"), 0);
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

TEST(sam_header, ExtraJsonStringRoundTrip) {
    sam_header h(1, 1, 10, 1, 0, 1);
    h.extra["array_meta"] = std::string("{\"x\": [1, 2]}");
    auto j = h.to_json();
    EXPECT_TRUE(j["array_meta"].is_string());
    auto h2 = sam_header::from_json(j);
    EXPECT_EQ(std::get<std::string>(h2.extra.at("array_meta")),
              "{\"x\": [1, 2]}");
}
