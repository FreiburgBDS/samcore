#pragma once

// Internal: lazy (deferred-read) h5sam/h5samd loading.  The HDF5 file stays
// open and data is decoded on demand in cached row blocks.  Note: the signal
// datasets are chunked + compressed, so real memory mapping is not possible;
// this is paged lazy reading, not mmap.  Not installed; not public API.

#include <H5Cpp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <list>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include <samcore/array.hpp>
#include <samcore/sam_header.hpp>
#include <samcore/sam_labels.hpp>

namespace samcore::io {

// Paged, cached reader over a chunked (compressed) 2-D HDF5 dataset.  Reads
// are decoded per row block and kept in an LRU cache so repeated access to
// the same rows does not re-decompress.  Not thread-safe.
template <class T>
class paged_reader {
public:
    paged_reader() = default;

    paged_reader(H5::H5File file, H5::DataSet dset, size_t rows, size_t cols,
                 H5::PredType mem_type, size_t target_block_bytes = 4u << 20,
                 size_t max_blocks = 8)
        : file_(std::move(file)),
          dset_(std::move(dset)),
          rows_(rows),
          cols_(cols),
          mem_type_(std::move(mem_type)) {
        const size_t row_bytes = cols_ * sizeof(T);
        block_rows_ = row_bytes == 0
                          ? 1
                          : std::max<size_t>(1, target_block_bytes / row_bytes);
        max_blocks_ = std::max<size_t>(1, max_blocks);
    }

    [[nodiscard]] size_t rows() const noexcept { return rows_; }
    [[nodiscard]] size_t cols() const noexcept { return cols_; }
    [[nodiscard]] size_t block_rows() const noexcept { return block_rows_; }
    // Number of blocks decoded from disk so far (read accounting for tests).
    [[nodiscard]] size_t blocks_read() const noexcept { return blocks_read_; }

    // Read rows [first, first + count) into `out` (row-major, count*cols).
    void read_rows(size_t first, size_t count, T* out) {
        if (first + count > rows_) {
            throw std::out_of_range("paged_reader: row range out of bounds");
        }
        size_t done = 0;
        while (done < count) {
            const size_t index = first + done;
            const size_t b = index / block_rows_;
            const block& blk = load_block(b);
            const size_t in_block = index - b * block_rows_;
            const size_t n = std::min(count - done, blk.rows - in_block);
            std::memcpy(out + done * cols_, blk.data.data() + in_block * cols_,
                        n * cols_ * sizeof(T));
            done += n;
        }
    }

    [[nodiscard]] std::vector<T> read_row(size_t index) {
        std::vector<T> out(cols_);
        read_rows(index, 1, out.data());
        return out;
    }

    // Read arbitrary rows (one row per index) into `out`.
    void read_selected(std::span<const std::int64_t> indices, T* out) {
        for (size_t i = 0; i < indices.size(); ++i) {
            const auto index = static_cast<size_t>(indices[i]);
            if (index >= rows_) {
                throw std::out_of_range("paged_reader: row index out of bounds");
            }
            const size_t b = index / block_rows_;
            const block& blk = load_block(b);
            std::memcpy(out + i * cols_,
                        blk.data.data() + (index - b * block_rows_) * cols_,
                        cols_ * sizeof(T));
        }
    }

private:
    struct block {
        std::vector<T> data;
        size_t rows = 0;
    };

    const block& load_block(size_t b) {
        auto it = cache_.find(b);
        if (it != cache_.end()) {
            touch(b);
            return it->second;
        }
        while (cache_.size() >= max_blocks_) {
            const size_t victim = lru_.back();
            lru_.pop_back();
            index_.erase(victim);
            cache_.erase(victim);
        }
        const size_t first = b * block_rows_;
        const size_t count = std::min(block_rows_, rows_ - first);
        block blk;
        blk.rows = count;
        blk.data.resize(count * cols_);
        const hsize_t offset[2] = {static_cast<hsize_t>(first), 0};
        const hsize_t sel[2] = {static_cast<hsize_t>(count),
                                static_cast<hsize_t>(cols_)};
        H5::DataSpace fspace = dset_.getSpace();
        fspace.selectHyperslab(H5S_SELECT_SET, sel, offset);
        H5::DataSpace mspace(2, sel);
        dset_.read(blk.data.data(), *mem_type_, mspace, fspace);
        ++blocks_read_;
        auto [ins, _] = cache_.emplace(b, std::move(blk));
        lru_.push_front(b);
        index_[b] = lru_.begin();
        return ins->second;
    }

    void touch(size_t b) {
        auto it = index_.find(b);
        if (it == index_.end()) return;
        lru_.erase(it->second);
        lru_.push_front(b);
        it->second = lru_.begin();
    }

    H5::H5File file_;
    H5::DataSet dset_;
    size_t rows_ = 0;
    size_t cols_ = 0;
    // PredType has a protected default constructor, hence the optional.
    std::optional<H5::PredType> mem_type_;
    size_t block_rows_ = 1;
    size_t max_blocks_ = 8;
    size_t blocks_read_ = 0;
    std::unordered_map<size_t, block> cache_;
    std::list<size_t> lru_;
    std::unordered_map<size_t, std::list<size_t>::iterator> index_;
};

struct h5sam_lazy_state {
    paged_reader<std::int8_t> reader;
};

struct h5sam_lazy_handle {
    sam_header header;
    sam_labels labels;
    std::optional<std::vector<std::int32_t>> starts;
    std::unique_ptr<h5sam_lazy_state> data;
};

// Open an .h5sam file, parse header/labels/starts eagerly and leave the
// signal data paged for on-demand reads.
[[nodiscard]] h5sam_lazy_handle read_h5sam_lazy(
    const std::filesystem::path& path);

// --- .h5samd -------------------------------------------------------------

// Open HDF5 handles for a .h5samd file.  X/Z/V stay on disk until read.
struct h5samd_lazy_state {
    std::optional<paged_reader<float>> x;
    std::optional<paged_reader<float>> z;
    std::optional<paged_reader<float>> v;
    size_t x_rows = 0;
    size_t x_cols = 0;
    size_t z_cols = 0;
    size_t v_cols = 0;
};

struct h5samd_lazy_handle {
    std::optional<sam_labels> labels;
    std::vector<std::pair<std::int32_t, std::int32_t>> cube_shapes;
    std::vector<double> cube_resolutions;
    std::vector<std::int32_t> scanlens;
    bool unsupervised = false;
    std::unique_ptr<h5samd_lazy_state> data;
};

// Owned X/Z/V read out of a lazy handle.
struct h5samd_lazy_data {
    array2d<float> x;
    std::optional<array2d<float>> z;
    std::optional<array2d<float>> v;
};

// Open a .h5samd file, parse labels/shapes/resolutions/scanlens eagerly and
// leave X/Z/V paged for on-demand reads.
[[nodiscard]] h5samd_lazy_handle read_h5samd_lazy(
    const std::filesystem::path& path);

// Read X/Z/V out of an open lazy state (full materialization).
[[nodiscard]] h5samd_lazy_data read_h5samd_lazy_data(h5samd_lazy_state& state);

} // namespace samcore::io
