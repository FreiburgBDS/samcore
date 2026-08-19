// gen_data - deterministic random SAM-like test data generator.
//
// Generates synthetic A-scans that mimic real scanning-acoustic-microscopy
// data: each scan is a gated sine burst (the surface/interface echo) plus
// low-amplitude gaussian noise, quantized to int8.
//
//   * .h5sam output: a single cube of (scans_y x scans_x) scans, each with a
//     per-scan start index (the echo onset; -1 for unaligned scans).
//   * .h5samd output: a multi-cube dataset (--cubes cubes), unsupervised,
//     padded to a common scan length.
//
// Build with -DSAMCORE_BUILD_EXECUTABLES=ON; the RNG is seeded deterministically
// so regenerating test data is reproducible.
//
// Usage:
//   gen_data <output.h5sam|output.h5samd> [options]
//
// Options (defaults reproduce the bundled testdata):
//   --scans-x N     scans per line (cols)          default 20
//   --scans-y N     number of lines                default 20
//   --scanlen N     samples per scan               default 5000
//   --samplerate F  sampling rate [MHz]            default 2500
//   --tzero N       time origin [ns]               default 15000
//   --resolution F  lateral resolution [um/px]     default 400
//   --period N      sine period [samples]          default 400
//   --amplitude N   sine amplitude [int8 LSB]      default 82
//   --noise-mean F  gaussian noise mean            default 19
//   --noise-std F   gaussian noise std             default 2
//   --seed N        RNG seed                       default 42
//   --cubes N       cubes in .h5samd output        default 3
//   --empty-prob F  fraction of scans with start -1 default 0.05

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <samcore/array.hpp>
#include <samcore/io/fileio.hpp>
#include <samcore/sam_labels.hpp>

namespace {

struct config {
    int scans_x = 20;
    int scans_y = 20;
    int scanlen = 5000;
    double samplerate = 2500.0;
    int tzero = 15000;
    double resolution = 400.0;
    int period = 400;
    signed char amplitude = 82;
    double noise_mean = 19.0;
    double noise_std = 2.0;
    unsigned seed = 42;
    int cubes = 3;
    double empty_prob = 0.05;
};

const char* g_prog = "gen_data";

void usage() {
    std::printf(
        "Usage: %s <output.h5sam|output.h5samd> [options]\n"
        "\n"
        "Generates deterministic random SAM-like data (gated sine burst + gaussian\n"
        "noise, int8) and writes it as a .h5sam (with per-scan starts) or a\n"
        "multi-cube .h5samd dataset.\n"
        "\n"
        "Options (defaults reproduce the bundled testdata):\n"
        "  --scans-x N      scans per line (cols)          default 20\n"
        "  --scans-y N      number of lines                default 20\n"
        "  --scanlen N      samples per scan               default 5000\n"
        "  --samplerate F   sampling rate [MHz]            default 2500\n"
        "  --tzero N        time origin [ns]               default 15000\n"
        "  --resolution F   lateral resolution [um/px]     default 400\n"
        "  --period N       sine period [samples]          default 400\n"
        "  --amplitude N    sine amplitude [int8 LSB]      default 82\n"
        "  --noise-mean F   gaussian noise mean            default 19\n"
        "  --noise-std F    gaussian noise std             default 2\n"
        "  --seed N         RNG seed                       default 42\n"
        "  --cubes N        cubes in .h5samd output        default 3\n"
        "  --empty-prob F   fraction of scans with start -1 default 0.05\n",
        g_prog);
}

double arg_double(const char* name, const char* value) {
    char* end = nullptr;
    const double v = std::strtod(value, &end);
    if (!end || *end != '\0') {
        std::fprintf(stderr, "%s: invalid value for %s: '%s'\n", g_prog, name,
                     value);
        std::exit(1);
    }
    return v;
}

long arg_long(const char* name, const char* value) {
    char* end = nullptr;
    const long v = std::strtol(value, &end, 10);
    if (!end || *end != '\0') {
        std::fprintf(stderr, "%s: invalid value for %s: '%s'\n", g_prog, name,
                     value);
        std::exit(1);
    }
    return v;
}

config parse_args(int argc, char** argv) {
    config c;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* opt) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s: missing value for %s\n", g_prog, opt);
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--scans-x") {
            c.scans_x = static_cast<int>(arg_long("--scans-x", need("--scans-x")));
        } else if (a == "--scans-y") {
            c.scans_y = static_cast<int>(arg_long("--scans-y", need("--scans-y")));
        } else if (a == "--scanlen") {
            c.scanlen = static_cast<int>(arg_long("--scanlen", need("--scanlen")));
        } else if (a == "--samplerate") {
            c.samplerate = arg_double("--samplerate", need("--samplerate"));
        } else if (a == "--tzero") {
            c.tzero = static_cast<int>(arg_long("--tzero", need("--tzero")));
        } else if (a == "--resolution") {
            c.resolution = arg_double("--resolution", need("--resolution"));
        } else if (a == "--period") {
            c.period = static_cast<int>(arg_long("--period", need("--period")));
        } else if (a == "--amplitude") {
            c.amplitude = static_cast<signed char>(
                arg_long("--amplitude", need("--amplitude")));
        } else if (a == "--noise-mean") {
            c.noise_mean = arg_double("--noise-mean", need("--noise-mean"));
        } else if (a == "--noise-std") {
            c.noise_std = arg_double("--noise-std", need("--noise-std"));
        } else if (a == "--seed") {
            c.seed = static_cast<unsigned>(
                arg_long("--seed", need("--seed")));
        } else if (a == "--cubes") {
            c.cubes = static_cast<int>(arg_long("--cubes", need("--cubes")));
        } else if (a == "--empty-prob") {
            c.empty_prob = arg_double("--empty-prob", need("--empty-prob"));
        } else if (a == "-h" || a == "--help") {
            usage();
            std::exit(0);
        } else {
            std::fprintf(stderr, "%s: unknown option: %s\n", g_prog, a.c_str());
            usage();
            std::exit(1);
        }
    }
    if (c.scans_x < 1 || c.scans_y < 1 || c.scanlen < 1 || c.period < 1) {
        std::fprintf(stderr, "%s: scans-x, scans-y, scanlen and period must be >= 1\n",
                     g_prog);
        std::exit(1);
    }
    if (c.empty_prob < 0.0 || c.empty_prob > 1.0) {
        std::fprintf(stderr, "%s: --empty-prob must be in [0, 1]\n", g_prog);
        std::exit(1);
    }
    return c;
}

// One A-scan: a gated sine burst starting at `start` with a gaussian-noise
// baseline, quantized to int8 (the same construction used for the bundled
// testdata).  Returns the cube and, per scan, the burst onset (the natural
// per-scan start index, or -1 for a small fraction of unaligned scans).
struct cube_result {
    samcore::array2d<std::int8_t> data;
    std::vector<std::int32_t> starts;
};

cube_result generate_cube(const config& c, std::mt19937& gen) {
    const size_t n = static_cast<size_t>(c.scans_y) * static_cast<size_t>(c.scans_x);
    samcore::array2d<std::int8_t> data(n, static_cast<size_t>(c.scanlen));
    std::vector<std::int32_t> starts(n);

    std::normal_distribution<double> noise(c.noise_mean, c.noise_std);
    std::uniform_int_distribution<int> start_dis(c.scanlen / 4, c.scanlen / 2);
    std::uniform_int_distribution<int> end_dis(3 * c.scanlen / 4,
                                               c.scanlen - 1);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    const double angular_freq = 2.0 * M_PI / c.period;

    for (size_t i = 0; i < n; ++i) {
        const int start = start_dis(gen);
        const int end = end_dis(gen);
        starts[i] = prob(gen) < c.empty_prob ? -1
                                             : static_cast<std::int32_t>(start);
        for (int j = 0; j < c.scanlen; ++j) {
            double v = 0.0;
            if (j >= start && j < end) {
                v = c.amplitude * std::sin(angular_freq * j);
            }
            v += noise(gen);
            data[i][static_cast<size_t>(j)] = static_cast<std::int8_t>(v);
        }
    }
    return {std::move(data), std::move(starts)};
}

void write_h5sam(const config& c, const std::filesystem::path& out) {
    std::mt19937 gen(c.seed);
    const size_t n = static_cast<size_t>(c.scans_y) * static_cast<size_t>(c.scans_x);

    samcore::sam_header header(c.scans_x, c.scans_y, c.scanlen, c.samplerate,
                               c.tzero, c.resolution, true, false);
    cube_result cube = generate_cube(c, gen);
    samcore::sam_labels labels = samcore::sam_labels::create_unlabeled(n);

    samcore::io::write_h5sam(out, cube.data, header, labels, cube.starts);

    const size_t aligned = n - static_cast<size_t>(
                                  std::count(cube.starts.begin(),
                                             cube.starts.end(), -1));
    std::printf("wrote %s: %d x %d scans, scanlen %d, %zu with starts\n",
                out.c_str(), c.scans_x, c.scans_y, c.scanlen, aligned);
}

void write_h5samd(const config& c, const std::filesystem::path& out) {
    std::mt19937 gen(c.seed);
    const size_t per_cube =
        static_cast<size_t>(c.scans_y) * static_cast<size_t>(c.scans_x);

    std::vector<std::pair<std::int32_t, std::int32_t>> shapes;
    std::vector<double> resolutions;
    std::vector<std::int32_t> scanlens;
    samcore::array2d<float> x(per_cube * static_cast<size_t>(c.cubes),
                              static_cast<size_t>(c.scanlen));

    for (int ci = 0; ci < c.cubes; ++ci) {
        cube_result cube = generate_cube(c, gen);
        for (size_t i = 0; i < per_cube; ++i) {
            for (size_t j = 0; j < static_cast<size_t>(c.scanlen); ++j) {
                x[ci * per_cube + i][j] = static_cast<float>(cube.data[i][j]);
            }
        }
        shapes.emplace_back(static_cast<std::int32_t>(c.scans_y),
                            static_cast<std::int32_t>(c.scans_x));
        resolutions.push_back(c.resolution);
        scanlens.push_back(static_cast<std::int32_t>(c.scanlen));
    }

    // Unsupervised dataset: no labels.
    samcore::io::write_h5samd(out, x, std::nullopt, shapes, resolutions,
                              scanlens, true, std::nullopt, std::nullopt);

    std::printf("wrote %s: %d cubes of %d x %d scans, scanlen %d\n", out.c_str(),
                c.cubes, c.scans_x, c.scans_y, c.scanlen);
}

} // namespace

int main(int argc, char** argv) {
    g_prog = argv[0];
    if (argc < 2) {
        usage();
        return 1;
    }

    const config c = parse_args(argc, argv);
    const std::filesystem::path out = argv[1];
    const std::string ext = out.extension().string();

    if (ext == ".h5sam") {
        write_h5sam(c, out);
    } else if (ext == ".h5samd") {
        write_h5samd(c, out);
    } else {
        std::fprintf(stderr,
                     "%s: unsupported output extension '%s' (use .h5sam or .h5samd)\n",
                     g_prog, ext.c_str());
        return 1;
    }
    return 0;
}
