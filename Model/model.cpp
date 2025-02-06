//
// Created by yaohc on 2024/12/25.
//

#include "model.h"



namespace Sturgeon {

torch::Tensor Permute::forward(torch::Tensor x) {
    return x.permute(dims);
}

ReverseLSTM::ReverseLSTM(int64_t input_size, int64_t hidden_size, int64_t num_layers, bool bidirectional,
                         bool reverse_)
        : reverse(reverse_) {
    lstm = torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_size)
                                   .num_layers(num_layers)
                                   .bidirectional(bidirectional)
                                   .batch_first(false));
    register_module("lstm", lstm);
}

torch::Tensor ReverseLSTM::forward(torch::Tensor x) {
    if (reverse) {
        x = x.flip(0);  // Reverse along the time dimension
    }
    auto output = std::get<0>(lstm->forward(x));
    if (reverse) {
        output = output.flip(0);  // Reverse back along the time dimension
    }
    return output;
}

CTC_encoder::CTC_encoder(std::vector<int64_t> conv, int64_t n_hid, double dropout) {
    encoder = torch::nn::Sequential(
        torch::nn::Conv1d(torch::nn::Conv1dOptions(1, conv[0], 3).stride(1).padding(1)),
        torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(conv[0]).track_running_stats(true)),
        torch::nn::SiLU(),
        torch::nn::Conv1d(torch::nn::Conv1dOptions(conv[0], conv[1], 5).stride(1).padding(2)),
        torch::nn::BatchNorm1d( torch::nn::BatchNorm1dOptions(conv[1]).track_running_stats(true)),
        torch::nn::SiLU(),
        torch::nn::Conv1d(torch::nn::Conv1dOptions(conv[1], n_hid, 15).stride(5).padding(7)),
        torch::nn::BatchNorm1d( torch::nn::BatchNorm1dOptions(n_hid).track_running_stats(true)),
        torch::nn::SiLU(),
        torch::nn::Conv1d(torch::nn::Conv1dOptions(n_hid, n_hid, 5).stride(1).padding(2)),
        torch::nn::BatchNorm1d( torch::nn::BatchNorm1dOptions(n_hid).track_running_stats(true)),
        torch::nn::SiLU(),
        torch::nn::Conv1d(torch::nn::Conv1dOptions(n_hid, n_hid, 5).stride(1).padding(2)),
        torch::nn::BatchNorm1d(  torch::nn::BatchNorm1dOptions(n_hid).track_running_stats(true)),
        torch::nn::SiLU(),
        Permute(std::vector<int64_t>{2, 0, 1}),
        ReverseLSTM(n_hid, n_hid, 1, false, true),
        ReverseLSTM(n_hid, n_hid, 1, false, false),
        ReverseLSTM(n_hid, n_hid, 1, false, true),
        ReverseLSTM(n_hid, n_hid / 4, 1, false, false),
        ReverseLSTM(n_hid / 4, n_hid / 16, 1, false, true),
        torch::nn::Linear(n_hid / 16, 5)
    );

    register_module("encoder", encoder);
}

torch::Tensor CTC_encoder::forward(torch::Tensor x) {
    return encoder->forward(x).log_softmax(2);
}



} // Sturgeon