// nanobind bindings for preprocessing / utils / io.

#include "common.hpp"

void bind_submodules(nb::module_& m) {
    // preprocessing

    nb::module_ pp = m.def_submodule(
        "preprocessing",
        "Signal preprocessing primitives operating on (n_signals, scanlen) "
        "float32 arrays.\n\n"
        "Every function returns a freshly computed float32 array of the "
        "same shape as ``data``.");
    pp.def("lp",
           [](in_f32_2 data, double cutoff, double fs) {
               return to_numpy(preprocessing::lp(view_in<float>(data), cutoff, fs));
           },
           nb::arg("data"), nb::arg("cutoff"), nb::arg("fs"),
           nb::sig(
               "def lp(data: numpy.typing.NDArray[numpy.float32], cutoff: float, fs: float) -> numpy.typing.NDArray[numpy.float32]"),
           "Butterworth low-pass filter.\n\n"
                   "Parameters\n"
                   "----------\n"
                   "data : ndarray (float32)\n"
                   "    Signals of shape (n_signals, scanlen).\n"
                   "cutoff : float\n"
                   "    Cutoff frequency in Hz.\n"
                   "fs : float\n"
                   "    Sampling rate in Hz.");
    pp.def("bp",
           [](in_f32_2 data, double cutoff_low, double cutoff_high, double fs) {
               return to_numpy(preprocessing::bp(view_in<float>(data), cutoff_low,
                                                 cutoff_high, fs));
           },
           nb::arg("data"), nb::arg("cutoff_low"), nb::arg("cutoff_high"),
           nb::arg("fs"),
           nb::sig(
               "def bp(data: numpy.typing.NDArray[numpy.float32], cutoff_low: float, cutoff_high: float, fs: float) -> numpy.typing.NDArray[numpy.float32]"),
           "Butterworth band-pass filter.\n\n"
                   "Parameters\n"
                   "----------\n"
                   "data : ndarray (float32)\n"
                   "    Signals of shape (n_signals, scanlen).\n"
                   "cutoff_low : float\n"
                   "    Lower cutoff frequency in Hz.\n"
                   "cutoff_high : float\n"
                   "    Upper cutoff frequency in Hz.\n"
                   "fs : float\n"
                   "    Sampling rate in Hz.");
    pp.def("normalize",
           [](in_f32_2 data, const std::string& mode) {
               return to_numpy(preprocessing::normalize(view_in<float>(data), mode));
           },
           nb::arg("data"), nb::arg("mode") = "minmax",
           nb::sig(
               "def normalize(data: numpy.typing.NDArray[numpy.float32], mode: str = 'minmax') -> numpy.typing.NDArray[numpy.float32]"),
           "Normalize each signal with 'max', 'zscore' or 'minmax'.\n\n"
                   "Parameters\n"
                   "----------\n"
                   "data : ndarray (float32)\n"
                   "    Signals of shape (n_signals, scanlen).\n"
                   "mode : str, optional\n"
                   "    One of 'max' (divide by the global maximum), "
                   "'zscore' (zero mean, unit variance) or 'minmax' "
                   "(scale to [0, 1]).");
    pp.def("savgol",
           [](in_f32_2 data, size_t window_length, size_t polyorder) {
               return to_numpy(
                   preprocessing::savgol(view_in<float>(data), window_length,
                                         polyorder));
           },
           nb::arg("data"), nb::arg("window_length") = 5,
           nb::arg("polyorder") = 2,
           nb::sig(
               "def savgol(data: numpy.typing.NDArray[numpy.float32], window_length: int = 5, polyorder: int = 2) -> numpy.typing.NDArray[numpy.float32]"),
           "Savitzky-Golay smoothing filter.\n\n"
                   "Parameters\n"
                   "----------\n"
                   "data : ndarray (float32)\n"
                   "    Signals of shape (n_signals, scanlen).\n"
                   "window_length : int, optional\n"
                   "    Odd window length.\n"
                   "polyorder : int, optional\n"
                   "    Polynomial order.");
    pp.def("medfilt",
           [](in_f32_2 data, size_t kernel_size) {
               return to_numpy(preprocessing::medfilt(view_in<float>(data),
                                                      kernel_size));
           },
           nb::arg("data"), nb::arg("kernel_size") = 3,
           nb::sig(
               "def medfilt(data: numpy.typing.NDArray[numpy.float32], kernel_size: int = 3) -> numpy.typing.NDArray[numpy.float32]"),
           "Median filter with an odd kernel size.\n\n"
                   "Parameters\n"
                   "----------\n"
                   "data : ndarray (float32)\n"
                   "    Signals of shape (n_signals, scanlen).\n"
                   "kernel_size : int, optional\n"
                   "    Odd kernel size.");
    pp.def("gate",
           [](in_f32_2 data, size_t start, size_t end) {
               return to_numpy(preprocessing::gate(view_in<float>(data), start, end));
           },
           nb::arg("data"), nb::arg("start") = 0, nb::arg("end") = 0,
           nb::sig(
               "def gate(data: numpy.typing.NDArray[numpy.float32], start: int = 0, end: int = 0) -> numpy.typing.NDArray[numpy.float32]"),
           "Keep samples ``start..end`` of every signal (in place).\n\n"
                   "Parameters\n"
                   "----------\n"
                   "data : ndarray (float32)\n"
                   "    Signals of shape (n_signals, scanlen).\n"
                   "start : int, optional\n"
                   "    First sample to keep.\n"
                   "end : int, optional\n"
                   "    Last sample to keep (exclusive); 0 means the full "
                   "scan length.");
    pp.def("detrend", [](in_f32_2 data) {
        return to_numpy(preprocessing::detrend(view_in<float>(data)));
    }, nb::sig(
           "def detrend(data: numpy.typing.NDArray[numpy.float32]) -> numpy.typing.NDArray[numpy.float32]"),
       "Remove a linear trend from every signal.");
    pp.def("envelope", [](in_f32_2 data) {
        return to_numpy(preprocessing::envelope(view_in<float>(data)));
    }, nb::sig(
           "def envelope(data: numpy.typing.NDArray[numpy.float32]) -> numpy.typing.NDArray[numpy.float32]"),
       "Hilbert envelope of every signal.");
    pp.def("moving_average",
           [](in_f32_2 data, size_t window) {
               return to_numpy(
                   preprocessing::moving_average(view_in<float>(data), window));
           },
           nb::arg("data"), nb::arg("window") = 5,
           nb::sig(
               "def moving_average(data: numpy.typing.NDArray[numpy.float32], window: int = 5) -> numpy.typing.NDArray[numpy.float32]"),
           "Moving-average smoothing with the given window size.\n\n"
                   "Parameters\n"
                   "----------\n"
                   "data : ndarray (float32)\n"
                   "    Signals of shape (n_signals, scanlen).\n"
                   "window : int, optional\n"
                   "    Window size in samples.");

    // utils

    nb::module_ ut = m.def_submodule(
        "utils", "Utility functions for signal analysis.");
    ut.def("kurt", [](in_f32_2 data) {
        return to_numpy(utils::kurt(view_in<float>(data)));
    }, nb::arg("data"),
       nb::sig(
           "def kurt(data: numpy.typing.NDArray[numpy.float32]) -> numpy.typing.NDArray[numpy.float64]"),
       "Kurtosis of each signal (rows of ``data``).\n\n"
               "Pearson kurtosis (bias-corrected = False, fisher = False), "
               "matching ``scipy.stats.kurtosis(..., fisher=False)``.");
    ut.def("time_index",
           [](double tzero, double delta_t, size_t num) {
               return to_numpy(utils::time_index(tzero, delta_t, num));
           },
           nb::arg("tzero"), nb::arg("delta_t"), nb::arg("num"),
           nb::sig(
               "def time_index(tzero: float, delta_t: float, num: int) -> numpy.typing.NDArray[numpy.float64]"),
           "Time axis ``tzero + i * delta_t`` for ``num`` samples.\n\n"
                   "Parameters\n"
                   "----------\n"
                   "tzero : float\n"
                   "    Time of the first sample.\n"
                   "delta_t : float\n"
                   "    Sample spacing.\n"
                   "num : int\n"
                   "    Number of samples.");
    ut.def("fft_spec",
           [](in_f32_1 scan, double d) {
               std::vector<float> v = copy_in<float>(scan);
               auto [mag, freqs] = utils::fft_spec(v, d);
               return nb::make_tuple(to_numpy(std::move(mag)),
                                     to_numpy(std::move(freqs)));
           },
           nb::arg("scan"), nb::arg("d") = 1.0,
           nb::sig(
               "def fft_spec(scan: numpy.typing.NDArray[numpy.float32], d: float = 1.0) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]"),
           "Magnitude spectrum and frequencies of a single signal.\n\n"
                   "Parameters\n"
                   "----------\n"
                   "scan : ndarray (float32)\n"
                   "    One signal.\n"
                   "d : float, optional\n"
                   "    Sample spacing (inverse of the sampling rate).\n\n"
                   "Returns\n"
                   "-------\n"
                   "(magnitude, freqs) : tuple of ndarray\n"
                   "    The magnitude spectrum and its frequency bins "
                   "(numpy rfft/rfftfreq parity).");
    ut.def("spectral_entropy",
           [](in_f32_2 psd, double base) {
               return to_numpy(utils::spectral_entropy(view_in<float>(psd), base));
           },
           nb::arg("psd"), nb::arg("base") = 2.0,
           nb::sig(
               "def spectral_entropy(psd: numpy.typing.NDArray[numpy.float32], base: float = 2.0) -> numpy.typing.NDArray[numpy.float64]"),
           "Spectral entropy per row of a PSD matrix.\n\n"
                   "Normalised Shannon entropy along the last axis; silent "
                   "rows return 0.\n\n"
                   "Parameters\n"
                   "----------\n"
                   "psd : ndarray (float32)\n"
                   "    PSD of shape (n_signals, n_freqs).\n"
                   "base : float, optional\n"
                   "    Logarithm base.");
    ut.def("spectral_flatness", [](in_f32_2 psd) {
        return to_numpy(utils::spectral_flatness(view_in<float>(psd)));
    }, nb::arg("psd"),
       nb::sig(
           "def spectral_flatness(psd: numpy.typing.NDArray[numpy.float32]) -> numpy.typing.NDArray[numpy.float64]"),
       "Spectral flatness (geometric / arithmetic mean) per row of a PSD "
               "matrix.");
    ut.def("spectral_centroid",
           [](in_f32_1 freqs, in_f32_2 psd) {
               return to_numpy(utils::spectral_centroid(
                   copy_in<float>(freqs), view_in<float>(psd)));
           },
           nb::arg("freqs"), nb::arg("psd"),
           nb::sig(
               "def spectral_centroid(freqs: numpy.typing.NDArray[numpy.float32], psd: numpy.typing.NDArray[numpy.float32]) -> numpy.typing.NDArray[numpy.float64]"),
           "Weighted mean frequency per row of a PSD matrix; silent rows "
                   "return 0.\n\n"
                   "Parameters\n"
                   "----------\n"
                   "freqs : ndarray (float32)\n"
                   "    Frequency bins (n_freqs,).\n"
                   "psd : ndarray (float32)\n"
                   "    PSD of shape (n_signals, n_freqs).");
    ut.def("spectral_energy_ratio",
           [](in_f32_1 freqs, in_f32_2 psd, double critical_freq) {
               return to_numpy(utils::spectral_energy_ratio(
                   copy_in<float>(freqs), view_in<float>(psd), critical_freq));
           },
           nb::arg("freqs"), nb::arg("psd"), nb::arg("critical_freq"),
           nb::sig(
               "def spectral_energy_ratio(freqs: numpy.typing.NDArray[numpy.float32], psd: numpy.typing.NDArray[numpy.float32], critical_freq: float) -> numpy.typing.NDArray[numpy.float64]"),
           "Ratio of energy below ``critical_freq`` to the total "
                   "energy per row.\n\n"
                   "Rows with zero below-critical energy return 0.\n\n"
                   "Parameters\n"
                   "----------\n"
                   "freqs : ndarray (float32)\n"
                   "    Frequency bins (n_freqs,).\n"
                   "psd : ndarray (float32)\n"
                   "    PSD of shape (n_signals, n_freqs).\n"
                   "critical_freq : float\n"
                   "    Frequency separating the two energy bands.");

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
       nb::sig(
           "def read_h5sam(path: str) -> tuple[numpy.typing.NDArray[numpy.int8], SAMHeader, SAMLabels, numpy.typing.NDArray[numpy.int32] | None]"),
       "Read a .h5sam file.\n\n"
               "Parameters\n"
               "----------\n"
               "path : str\n"
               "    Path to the .h5sam file.\n\n"
               "Returns\n"
               "-------\n"
               "(data, header, labels, starts) : tuple\n"
               "    The int8 signal array (n_signals, scanlen), the header, "
               "the labels and the per-scan start indices (or None).");
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
           nb::sig(
               "def write_h5sam(path: str, data: numpy.typing.NDArray[numpy.int8], header: SAMHeader, samlabels: SAMLabels, starts: collections.abc.Sequence[int] | None = None) -> None"),
           "Write an int8 signal array plus header, labels and "
                   "optional starts to a .h5sam file.");
    io.def("read_h5samd", [](const std::string& path) {
        return sam_dataset::load(path);
    }, nb::arg("path"),
       nb::sig("def read_h5samd(path: str) -> SAMDataset"),
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
           nb::sig(
               "def convert_h5sam_to_h5samd(input_paths: collections.abc.Sequence[str], output_path: str, pad_value: float = 0.0, unsupervised: bool | None = None) -> None"),
           "Convert one or more .h5sam files into a .h5samd "
                   "dataset, padding to the longest scan length.");
}
