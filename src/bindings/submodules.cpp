// nanobind bindings for preprocessing / utils / io.

#include "common.hpp"

void bind_submodules(nb::module_& m) {
    // preprocessing

    nb::module_ pp = m.def_submodule(
        "preprocessing",
        "Signal preprocessing primitives operating on (n_signals, scanlen) "
        "float32 arrays.");
    pp.def("lp",
           [](in_f32_2 data, double cutoff, double fs) {
               return to_numpy(preprocessing::lp(view_in<float>(data), cutoff, fs));
           },
           nb::arg("data"), nb::arg("cutoff"), nb::arg("fs"),
           "Butterworth low-pass filter.");
    pp.def("bp",
           [](in_f32_2 data, double cutoff_low, double cutoff_high, double fs) {
               return to_numpy(preprocessing::bp(view_in<float>(data), cutoff_low,
                                                 cutoff_high, fs));
           },
           nb::arg("data"), nb::arg("cutoff_low"), nb::arg("cutoff_high"),
           nb::arg("fs"),
           "Butterworth band-pass filter.");
    pp.def("normalize",
           [](in_f32_2 data, const std::string& mode) {
               return to_numpy(preprocessing::normalize(view_in<float>(data), mode));
           },
           nb::arg("data"), nb::arg("mode") = "minmax",
           "Normalize each signal with 'max', 'zscore' or 'minmax'.");
    pp.def("savgol",
           [](in_f32_2 data, size_t window_length, size_t polyorder) {
               return to_numpy(
                   preprocessing::savgol(view_in<float>(data), window_length,
                                         polyorder));
           },
           nb::arg("data"), nb::arg("window_length") = 5,
           nb::arg("polyorder") = 2,
           "Savitzky-Golay smoothing filter.");
    pp.def("medfilt",
           [](in_f32_2 data, size_t kernel_size) {
               return to_numpy(preprocessing::medfilt(view_in<float>(data),
                                                      kernel_size));
           },
           nb::arg("data"), nb::arg("kernel_size") = 3,
           "Median filter with an odd kernel size.");
    pp.def("gate",
           [](in_f32_2 data, size_t start, size_t end) {
               return to_numpy(preprocessing::gate(view_in<float>(data), start, end));
           },
           nb::arg("data"), nb::arg("start") = 0, nb::arg("end") = 0,
           "Keep samples ``start..end`` of every signal.");
    pp.def("detrend", [](in_f32_2 data) {
        return to_numpy(preprocessing::detrend(view_in<float>(data)));
    }, "Remove a linear trend from every signal.");
    pp.def("envelope", [](in_f32_2 data) {
        return to_numpy(preprocessing::envelope(view_in<float>(data)));
    }, "Hilbert envelope of every signal.");
    pp.def("moving_average",
           [](in_f32_2 data, size_t window) {
               return to_numpy(
                   preprocessing::moving_average(view_in<float>(data), window));
           },
           nb::arg("data"), nb::arg("window") = 5,
           "Moving-average smoothing with the given window size.");

    // utils

    nb::module_ ut = m.def_submodule(
        "utils", "Utility functions for signal analysis.");
    ut.def("kurt", [](in_f32_2 data) {
        return to_numpy(utils::kurt(view_in<float>(data)));
    }, nb::arg("data"),
       "Kurtosis of each signal (rows of ``data``).");
    ut.def("time_index",
           [](double tzero, double delta_t, size_t num) {
               return to_numpy(utils::time_index(tzero, delta_t, num));
           },
           nb::arg("tzero"), nb::arg("delta_t"), nb::arg("num"),
           "Time axis ``tzero + i * delta_t`` for ``num`` samples.");
    ut.def("fft_spec",
           [](in_f32_1 scan, double d) {
               std::vector<float> v = copy_in<float>(scan);
               auto [mag, freqs] = utils::fft_spec(v, d);
               return nb::make_tuple(to_numpy(std::move(mag)),
                                     to_numpy(std::move(freqs)));
           },
           nb::arg("scan"), nb::arg("d") = 1.0,
           "Magnitude spectrum and frequencies of a single signal.");
    ut.def("spectral_entropy",
           [](in_f32_2 psd, double base) {
               return to_numpy(utils::spectral_entropy(view_in<float>(psd), base));
           },
           nb::arg("psd"), nb::arg("base") = 2.0,
           "Spectral entropy per row of a PSD matrix.");
    ut.def("spectral_flatness", [](in_f32_2 psd) {
        return to_numpy(utils::spectral_flatness(view_in<float>(psd)));
    }, nb::arg("psd"),
       "Spectral flatness per row of a PSD matrix.");
    ut.def("spectral_centroid",
           [](in_f32_1 freqs, in_f32_2 psd) {
               return to_numpy(utils::spectral_centroid(
                   copy_in<float>(freqs), view_in<float>(psd)));
           },
           nb::arg("freqs"), nb::arg("psd"),
           "Weighted mean frequency per row of a PSD matrix.");
    ut.def("spectral_energy_ratio",
           [](in_f32_1 freqs, in_f32_2 psd, double critical_freq) {
               return to_numpy(utils::spectral_energy_ratio(
                   copy_in<float>(freqs), view_in<float>(psd), critical_freq));
           },
           nb::arg("freqs"), nb::arg("psd"), nb::arg("critical_freq"),
           "Ratio of energy below ``critical_freq`` to the total "
                   "energy per row.");

    // file I/O

    nb::module_ io = m.def_submodule(
        "io", "Reading and writing .h5sam / .h5samd files.");
    io.def("read_h5sam", [](const std::string& path) {
        auto r = samcore::io::read_h5sam(path);
        return nb::make_tuple(to_numpy(std::move(r.data)),
                              nb::cast(std::move(r.header)),
                              nb::cast(std::move(r.labels)),
                              r.starts ? to_numpy(std::move(*r.starts))
                                       : nb::object(nb::none()));
    }, nb::arg("path"),
       "Read a .h5sam file.  Returns "
               "``(data, header, labels, starts)``.");
    io.def("write_h5sam",
           [](const std::string& path, in_i8_2 data, sam_header header,
              sam_labels labels,
              std::optional<std::vector<std::int32_t>> starts) {
               samcore::io::write_h5sam(path, copy_in<std::int8_t>(data),
                                        std::move(header), std::move(labels),
                                        std::move(starts));
           },
           nb::arg("path"), nb::arg("data"), nb::arg("header"),
           nb::arg("samlabels"), nb::arg("starts") = nb::none(),
           "Write an int8 signal array plus header, labels and "
                   "optional starts to a .h5sam file.");
    io.def("read_h5samd", [](const std::string& path) {
        return sam_dataset::load(path);
    }, nb::arg("path"),
       "Read a .h5samd file as a :class:`SAMDataset`.");
    io.def("convert_h5sam_to_h5samd",
           [](const std::vector<std::string>& input_paths,
              const std::string& output_path, float pad_value,
              std::optional<bool> unsupervised) {
               std::vector<std::filesystem::path> paths;
               for (const auto& p : input_paths) paths.emplace_back(p);
               samcore::io::convert_h5sam_to_h5samd(paths, output_path,
                                                    pad_value, unsupervised);
           },
           nb::arg("input_paths"), nb::arg("output_path"),
           nb::arg("pad_value") = 0.0f, nb::arg("unsupervised") = nb::none(),
           "Convert one or more .h5sam files into a .h5samd "
                   "dataset, padding to the longest scan length.");
}
