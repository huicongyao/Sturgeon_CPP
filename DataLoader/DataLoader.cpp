//
// Created by yaohc on 2024/12/10.
//

#include "DataLoader.h"
#include "../3rd_party/thread_pool.hpp"

std::vector<Sturgeon::Chunk> Sturgeon::Signal::get_chunks(int32_t chunk_len, int32_t stride, int32_t overlap) {
    // Normalize signals with MAD
//    if (this->read_id == "002919FF-7ADA-48ED-9688-8AA9367077B7") {
//        spdlog::info("1111");
//    }
    auto factors = Sturgeon::get_norm_factors(data);
    s_shift = std::get<0>(factors);
    s_scale = std::get<1>(factors);

    trim_start = Sturgeon::compute_trim_start(data, 40, s_scale * 2.0 + s_shift, 10, 3, 8000, 0.3);
    data.add_(-s_shift).div_(s_scale);

    std::vector<Sturgeon::Chunk> chunks;

    // handle metadata with shared_ptr
    auto info = std::make_shared<chunk_meta>(read_id,
                                             file_name,
                                             stride,
                                             overlap,
                                             trim_start,
                                             static_cast<size_t>(data.size(0)));
    if (data.size(0) - trim_start - 200 < chunk_len) return chunks;
    size_t st = trim_start, ed = trim_start + chunk_len ;
    while (static_cast<int64_t> (ed) < data.size(0) - 200) { // ignore post 200 signals at the end
        chunks.emplace_back(Sturgeon::Chunk{
            info,
            st,
            ed,
            data.slice(0, st, ed),
        });
        st += chunk_len - overlap;
        ed += chunk_len - overlap;
    }
    return chunks;
}


Sturgeon::DataLoader::DataLoader(const fs::path &h5_directory, int32_t batch_size, int32_t num_workers,
                                 int32_t chunk_len, int32_t overlap, int32_t stride) :
        h5_directory(h5_directory), batch_size(batch_size),
        num_workers(num_workers), chunk_len(chunk_len),
        stride(stride), overlap(overlap) {
    // Gather all h5 files
    for (const auto& entry : fs::directory_iterator(h5_directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".h5") {
            h5_files.push_back(entry.path());
        }
    }
}


bool Sturgeon::Sig_Batch::empty() const {
    return chunks.empty();
}


void Sturgeon::DataLoader::generate_batch(Sturgeon::ThreadSafeQueue<Sturgeon::Sig_Batch> & data_queue) {
    std::atomic<int64_t> chunk_cnt{0};
    std::string child_path = bp::search_path("Sturgeon_sub_loader", {"./"}).string();
    if (child_path.empty()) {
        std::cerr << "Child program not found!" << std::endl;
        child_path = "./Sturgeon_sub_loader"; // 使用相对路径
    }
    std::cout << "Child program path: " << child_path << std::endl;
    {
        ThreadPool pool(num_workers);
        spdlog::info("datalader worker size-{}", num_workers);
        auto sub_loader = [&](fs::path file, size_t batch_size,
                              Sturgeon::ThreadSafeQueue<Sturgeon::Sig_Batch> &data_queue) -> void {
            std::string h5_file_name = file.filename().string();
            bp::child child(child_path, h5_file_name, file.string());
            child.wait();
            auto st1 = std::chrono::high_resolution_clock::now();
            bip::managed_shared_memory segment(bip::open_only, h5_file_name.c_str());
            std::pair<char*, std::size_t> shm_data = segment.find<char>("SerializedData");
            std::string serialized_data(shm_data.first, shm_data.second);
            std::unordered_map<std::string, std::vector<float>> data;
            std::istringstream iss(serialized_data);
            boost::archive::binary_iarchive ia(iss);
            ia >> data;
            bip::shared_memory_object::remove(h5_file_name.c_str());
            auto st2 = std::chrono::high_resolution_clock::now();

            std::vector<Sturgeon::Chunk> batch_curr;
            for (const auto & [read_id, signals] : data) {
                Sturgeon::Signal signal(file.filename(), read_id, {static_cast<long>(signals.size())});
                std::memcpy(signal.data.data_ptr<float>(), signals.data(), signals.size() * sizeof(float));
                auto chunk_tmp = signal.get_chunks(chunk_len, stride, overlap);
                batch_curr.insert(batch_curr.end(), std::make_move_iterator(chunk_tmp.begin()),
                                  std::make_move_iterator(chunk_tmp.end()));
                if (batch_curr.size() >= batch_size * 10) {
                    auto item = Sturgeon::Sig_Batch(std::move(batch_curr), batch_size);
                    data_queue.push(std::move(item));
                    batch_curr = {};
                }
            }
            if (!batch_curr.empty()) {
                auto item = Sturgeon::Sig_Batch(std::move(batch_curr), batch_size);
                data_queue.push(std::move(item));
//                batch_curr = {};
            }
            auto st3 = std::chrono::high_resolution_clock::now();
            auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(st2 - st1);
            auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(st3 - st2);

//            spdlog::info("Get data chunks for file-{}, finished, "
//                         "time cost(milliseconds): load data from "
//                         "shared mem {}, split into unified chunks {}",
//                         file.string(), duration1.count(), duration2.count());
        };
        // 在程序初始化时设置
        H5::Exception::dontPrint();
        int cnt = 0;
        for (auto &file: h5_files) {
//            if (file.filename().string() == "reads_0050_3A9F1EA1-726C-407C-B3C4-0D743B1A046F.h5")
            pool.enqueue(sub_loader, file, batch_size, std::ref(data_queue));
            cnt++;
            if (cnt >= 700) break;
        }
    }
    for (size_t i = 0; i < torch::cuda::device_count(); i++)
        data_queue.push({});
    spdlog::info("DataLoader::generate_batch finished");
}

Sturgeon::Sig_Batch::Sig_Batch(std::vector<Chunk> && chunks_, int64_t batch_size)
        : chunks(std::move(chunks_)) {
    batches.reserve(chunks.size() / batch_size + 1);
    for (size_t st = 0; st < chunks.size(); st += batch_size) {
        size_t ed = std::min(chunks.size(), st + batch_size);
        std::vector<torch::Tensor> batch_data;
        batch_data.reserve(ed - st);
        for (size_t i = st; i < ed; i++) {
            batch_data.push_back(chunks[i].data.reshape({1, 1, 6000}));
        }
        torch::Tensor batch_tensor = torch::cat(batch_data, 0);
        batches.push_back(batch_tensor);
    }
    results.reserve(batches.size());
}


