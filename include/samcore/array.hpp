#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace samcore {

// Non-owning view over external memory (used by the Python bindings for
// zero-copy inputs).  Views must not be resized/filled/released (enforced);
// their lifetime must not exceed the underlying buffer.  Copying a view
// materializes it, so value copies never alias external memory.
struct non_owning_t {
    explicit non_owning_t() = default;
};
inline constexpr non_owning_t non_owning{};

// Minimal contiguous row-major 2D array.  The buffer is a single
// std::vector so the whole contents can be exposed as raw pointers for
// zero-copy interop with the future samcore Python package (nanobind).
// An array is either owning (backed by buf_) or a non-owning view (ptr_);
// all metadata, element access and comparison are storage-independent, and
// copying a view deep-copies the elements.
template <class T>
class array2d {
public:
    using value_type = T;

    array2d() = default;

    array2d(size_t rows, size_t cols)
        : rows_(rows), cols_(cols), buf_(checked_size(rows, cols)) {}

    array2d(size_t rows, size_t cols, T fill)
        : rows_(rows), cols_(cols), buf_(checked_size(rows, cols), fill) {}

    array2d(size_t rows, size_t cols, std::vector<T> buf)
        : rows_(rows), cols_(cols), buf_(std::move(buf)) {
        if (buf_.size() != rows_ * cols_) {
            throw std::invalid_argument(
                "array2d: buffer size does not match rows*cols");
        }
    }

    // Non-owning view over external row-major memory.
    array2d(T* data, size_t rows, size_t cols, non_owning_t)
        : rows_(rows), cols_(cols), ptr_(data) {}

    // Copying an owning array copies its buffer; copying a view allocates
    // and copies the borrowed elements (deep copy, never aliases).
    array2d(const array2d& o) : rows_(o.rows_), cols_(o.cols_) {
        copy_from(o);
    }
    array2d& operator=(const array2d& o) {
        if (this == &o) return *this;
        rows_ = o.rows_;
        cols_ = o.cols_;
        copy_from(o);
        return *this;
    }
    array2d(array2d&& o) noexcept
        : rows_(o.rows_), cols_(o.cols_), buf_(std::move(o.buf_)),
          ptr_(o.ptr_) {
        o.reset();
    }
    array2d& operator=(array2d&& o) noexcept {
        if (this == &o) return *this;
        rows_ = o.rows_;
        cols_ = o.cols_;
        buf_ = std::move(o.buf_);
        ptr_ = o.ptr_;
        o.reset();
        return *this;
    }
    ~array2d() = default;

    [[nodiscard]] size_t rows() const noexcept { return rows_; }
    [[nodiscard]] size_t cols() const noexcept { return cols_; }
    [[nodiscard]] size_t size() const noexcept { return rows_ * cols_; }
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] bool is_view() const noexcept { return ptr_ != nullptr; }

    [[nodiscard]] T* data() noexcept { return ptr_ ? ptr_ : buf_.data(); }
    [[nodiscard]] const T* data() const noexcept {
        return ptr_ ? ptr_ : buf_.data();
    }

    std::span<T> operator[](size_t r) noexcept {
        return std::span<T>(data() + r * cols_, cols_);
    }
    std::span<const T> operator[](size_t r) const noexcept {
        return std::span<const T>(data() + r * cols_, cols_);
    }

    [[nodiscard]] std::span<T> flat() noexcept {
        return std::span<T>(data(), size());
    }
    [[nodiscard]] std::span<const T> flat() const noexcept {
        return std::span<const T>(data(), size());
    }

    void resize(size_t rows, size_t cols) {
        require_owner("resize");
        rows_ = rows;
        cols_ = cols;
        buf_.resize(checked_size(rows, cols));
    }

    void fill(T v) {
        require_owner("fill");
        std::fill(buf_.begin(), buf_.end(), v);
    }

    [[nodiscard]] bool operator==(const array2d& o) const {
        if (rows_ != o.rows_ || cols_ != o.cols_) return false;
        if (size() == 0) return true;
        return std::equal(flat().begin(), flat().end(), o.flat().begin());
    }

    // Take ownership of the buffer, detaching from this array.
    [[nodiscard]] std::vector<T> release() {
        require_owner("release");
        rows_ = 0;
        cols_ = 0;
        return std::move(buf_);
    }

private:
    static size_t checked_size(size_t rows, size_t cols) {
        if (rows != 0 && cols > SIZE_MAX / rows) {
            throw std::overflow_error("array2d: rows*cols overflows size_t");
        }
        return rows * cols;
    }

    // Reset to a consistent empty owning state (used by move-out).
    void reset() noexcept {
        rows_ = 0;
        cols_ = 0;
        buf_.clear();
        ptr_ = nullptr;
    }

    void require_owner(const char* what) const {
        if (is_view()) {
            throw std::logic_error(std::string("array2d: cannot ") + what +
                                   " a non-owning view");
        }
    }

    // Deep-copy o into this array, materializing views into buf_.
    void copy_from(const array2d& o) {
        ptr_ = nullptr;
        if (o.is_view()) {
            if (o.size() > 0) {
                buf_.assign(o.data(), o.data() + o.size());
            } else {
                buf_.clear();
            }
        } else {
            buf_ = o.buf_;
        }
    }

    size_t rows_ = 0;
    size_t cols_ = 0;
    std::vector<T> buf_;
    T* ptr_ = nullptr; // non-owning views only
};

// Minimal contiguous row-major 3D array (used for STFT / spectrogram cubes).
template <class T>
class array3d {
public:
    using value_type = T;

    array3d() = default;

    array3d(size_t d0, size_t d1, size_t d2)
        : d0_(d0), d1_(d1), d2_(d2), buf_(checked_size(d0, d1, d2)) {}

    [[nodiscard]] size_t size0() const noexcept { return d0_; }
    [[nodiscard]] size_t size1() const noexcept { return d1_; }
    [[nodiscard]] size_t size2() const noexcept { return d2_; }
    [[nodiscard]] size_t size() const noexcept { return buf_.size(); }
    [[nodiscard]] bool empty() const noexcept { return buf_.empty(); }

    [[nodiscard]] T* data() noexcept { return buf_.data(); }
    [[nodiscard]] const T* data() const noexcept { return buf_.data(); }

    [[nodiscard]] std::span<T> flat() noexcept { return std::span<T>(buf_); }
    [[nodiscard]] std::span<const T> flat() const noexcept {
        return std::span<const T>(buf_);
    }

    // Row along the first two axes (e.g. one signal's f x t plane).
    [[nodiscard]] std::span<const T> plane(size_t i) const noexcept {
        return std::span<const T>(buf_.data() + i * d1_ * d2_, d1_ * d2_);
    }

private:
    static size_t checked_size(size_t a, size_t b, size_t c) {
        if (a != 0 && b != 0 && c != 0) {
            if (a > SIZE_MAX / b) throw std::overflow_error("array3d overflow");
            if (a * b > SIZE_MAX / c) throw std::overflow_error("array3d overflow");
        }
        return a * b * c;
    }

    size_t d0_ = 0;
    size_t d1_ = 0;
    size_t d2_ = 0;
    std::vector<T> buf_;
};

} // namespace samcore
