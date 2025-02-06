//
// Created by yaohc on 2024/12/23.
//

#include "basecall.h"

Sturgeon::Basecall::Basecall(const std::string model_path) : model_path(model_path){
}


void Sturgeon::Basecall::basecall(Sturgeon::ThreadSafeQueue<Sturgeon::Sig_Batch> &data_queue,
                                  Sturgeon::ThreadSafeQueue<Sturgeon::Sig_Batch> & decode_queue,
                                  bool is_half) {
    std::atomic<int64_t> batch_cnt{0};
    std::atomic<int64_t> sample_cnt{0};
    auto worker = [&] (Sturgeon::ThreadSafeQueue<Sturgeon::Sig_Batch> &data_queue,
                        fs::path & model_path,
                        torch::Device device){
        auto model = std::make_shared<Sturgeon::CTC_encoder>();
        try {
            torch::load(model, model_path);
//            spdlog::info("Model loaded successfully.");
        } catch (const c10::Error& e) {
            spdlog::error("Error loading the model: {}", e.what());
            return;
        }
        if (is_half) model->to(torch::kHalf);
        model->to(device);
        model->eval();

        torch::NoGradGuard no_grad;
        while (true) {
            auto sig_batch = data_queue.pop_and_get();
            if (sig_batch.chunks.empty()) break;
            for (torch::Tensor & batch_tensor : sig_batch.batches) {
                if (is_half) {
                    batch_tensor = batch_tensor.to(torch::kHalf);
                }
                torch::Tensor output = model->forward(batch_tensor.to(device));
                output = output.to(torch::kCPU).to(torch::kFloat);
                sig_batch.results.push_back(output);
            }
            decode_queue.push(std::move(sig_batch));
            sample_cnt.fetch_add(sig_batch.chunks.size(), std::memory_order_seq_cst);
            batch_cnt++;
        }
    };

    auto st = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> workers;
    for(size_t i = 0; i < torch::cuda::device_count() ; i++) {
        torch::Device device(torch::kCUDA, i);
        workers.emplace_back(worker, std::ref(data_queue), std::ref(model_path), device);
    }
    for (auto &worker : workers) {
        worker.join();
    }
    decode_queue.push({});
    auto ed = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(ed - st);
    spdlog::info("Basecall finished, batch_cnt: {}", batch_cnt.load());
    spdlog::info("Basecall received total samples {}, receive samples speed: {}", sample_cnt, sample_cnt.load() * 1000 / duration.count());
}
