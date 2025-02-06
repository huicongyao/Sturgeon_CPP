//
// Created by yaohc on 2024/12/10.
//

#ifndef STURGEON_DATALOADER_H
#define STURGEON_DATALOADER_H
#include <boost/process.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/map.hpp>

#include <string>
#include <torch/torch.h>
#include <filesystem>
#include <utility>
#include <unordered_map>
#include <spdlog/spdlog.h>
#include <H5Cpp.h>
#include <hdf5.h>
//#include <mpi.h>
#include "../utils.h"

namespace bp = boost::process;
namespace bip = boost::interprocess;
namespace fs = std::filesystem;

namespace Sturgeon{

struct chunk_meta {
    std::string read_id;
    std::string file_name;
    int32_t stride;
    int32_t overlap;
    int32_t trim_start;
    size_t signal_size;

    chunk_meta(std::string read_id, std::string file_name,
               int32_t stide, int32_t overlap, int32_t trim_start, size_t signal_size)
            : read_id(std::move(read_id)), file_name(std::move(file_name)),
            stride(stide), overlap(overlap),
            trim_start(trim_start), signal_size(signal_size) {}
};

struct Chunk {
    std::shared_ptr<chunk_meta> meta;
    size_t start;
    size_t end;
    torch::Tensor data;
};

struct Sig_Batch {
    std::vector<Chunk> chunks;
    std::vector<torch::Tensor> batches;
//    torch::Tensor results;
    std::vector<torch::Tensor> results;

    Sig_Batch() = default;
    Sig_Batch(std::vector<Chunk> && chunks, int64_t batch_size);
    bool empty() const;
    // move constructors
    Sig_Batch(std::vector<Chunk>  && chunks_, std::vector<torch::Tensor> && batches_, std::vector<torch::Tensor> && results_)
            : chunks(std::move(chunks_)), batches(std::move(batches_)), results(std::move(results_)) {}
};

struct Signal {
    std::string file_name;
    std::string read_id;
    int32_t trim_start;
    float s_scale;
    float s_shift;
    torch::Tensor data;

    Signal(std::string file_name, std::string  read_id, std::vector<long> size)
            : file_name(file_name), read_id(read_id), data(torch::empty(size, torch::kFloat)) {}

    torch::Tensor get_data() const { return data; }

    std::string get_file_name() const { return file_name; }

    std::string get_read_id() const { return read_id; }

    std::vector<Chunk> get_chunks(int32_t chunk_len, int32_t stride, int32_t overlap);

};

    void sub_loaderr(fs::path file, size_t batch_size,
                     Sturgeon::ThreadSafeQueue<Sturgeon::Sig_Batch> &data_queue);

class DataLoader {
private:
    std::vector<fs::path> h5_files;
    fs::path h5_directory;
    int32_t batch_size;
    int32_t num_workers;
    int32_t chunk_len;
    int32_t stride;
    int32_t overlap;

public:
    DataLoader(const fs::path &h5_directory, int32_t batch_size,
               int32_t num_workers, int32_t chunk_len,
               int32_t overlap, int32_t stride);

    void generate_batch(Sturgeon::ThreadSafeQueue<Sturgeon::Sig_Batch> & data_queue);
};

}


#endif //STURGEON_DATALOADER_H
