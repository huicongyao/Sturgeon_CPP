//
// Created by yaohc on 2024/12/25.
//


/*
 * Implement CTC_encoder in C++ with libtorch.
 * The parameter could be loaded from python trained model
 */
#ifndef STURGEON_MODEL_H
#define STURGEON_MODEL_H
#include <torch/torch.h>
#include <torch/script.h>

namespace Sturgeon {
    struct Permute : torch::nn::Module {
        std::vector<int64_t> dims;

        explicit Permute(std::vector<int64_t> dims_) : dims(std::move(dims_)) {}

        torch::Tensor forward(torch::Tensor x);
    };

    struct ReverseLSTM : torch::nn::Module {
        torch::nn::LSTM lstm{nullptr};
        bool reverse;

        ReverseLSTM(int64_t input_size, int64_t hidden_size, int64_t num_layers = 1,
                    bool bidirectional = false, bool reverse_ = false);

        torch::Tensor forward(torch::Tensor x);
    };

    struct CTC_encoder : torch::nn::Module {
        torch::nn::Sequential encoder;

        explicit CTC_encoder(std::vector<int64_t> conv = {8, 64}, int64_t n_hid = 512, double dropout = 0.2);

        torch::Tensor forward(torch::Tensor x);
    };

} // Sturgeon

#endif //STURGEON_MODEL_H
