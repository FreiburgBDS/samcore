// OpenMP scaling smoke benchmark for libsamcore.
// Build with -DSAMCORE_BUILD_EXECUTABLES=ON; run ./bench [scans_x scans_y scanlen]

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include <samcore/sam_scan.hpp>

#ifdef SAMCORE_HAS_OPENMP
#include <omp.h>
#endif

using namespace samcore;

namespace {

template <class F>
double time_ms(F&& f) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

} // namespace

int main(int argc, char** argv) {
    const std::int64_t nl = argc > 1 ? std::atol(argv[1]) : 256;
    const std::int64_t nc = argc > 2 ? std::atol(argv[2]) : 256;
    const std::int64_t sl = argc > 3 ? std::atol(argv[3]) : 2000;

    sam_header header(nc, nl, sl, 100.0, 15000, 1.0);
    array2d<std::int8_t> data(static_cast<size_t>(nl * nc),
                              static_cast<size_t>(sl));
    for (size_t i = 0; i < data.rows(); ++i) {
        for (size_t j = 0; j < data.cols(); ++j) {
            data[i][j] = static_cast<std::int8_t>(
                ((i * 13 + j * 7) ^ static_cast<size_t>(j >> 3)) % 120 - 60);
        }
    }
    sam_scan scan = sam_scan::from_data(std::move(data), header);

    std::printf("cube: %ld x %ld scans, scanlen %ld\n", (long)nl, (long)nc,
                (long)sl);

#ifdef SAMCORE_HAS_OPENMP
    std::printf("OpenMP threads: %d\n", omp_get_max_threads());
#else
    std::printf("OpenMP: disabled\n");
#endif

    double t;
    t = time_ms([&] { auto img = scan.image_absmax(); (void)img; });
    std::printf("image(absmax)   : %8.2f ms\n", t);

    t = time_ms([&] { auto z = scan.zgate(0.5, 500); (void)z; });
    std::printf("zgate(0.5, 500) : %8.2f ms\n", t);

    t = time_ms([&] { auto d = scan.downsampled(4, downsample_mode::mean); (void)d; });
    std::printf("downsample(mean): %8.2f ms\n", t);

    t = time_ms([&] { auto st = scan.compute_stft(256, 128); (void)st; });
    std::printf("compute_stft    : %8.2f ms\n", t);

    t = time_ms([&] { auto p = scan.psd(256, 128); (void)p; });
    std::printf("psd             : %8.2f ms\n", t);

    return 0;
}
