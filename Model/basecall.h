//
// Created by yaohc on 2024/12/23.
//

#ifndef STURGEON_BASECALL_H
#define STURGEON_BASECALL_H
#include <torch/torch.h>
#include <torch/script.h>
#include <spdlog/spdlog.h>
#include <random>
#include <atomic>
#include "../utils.h"
#include "../DataLoader/DataLoader.h"
#include "model.h"

namespace Sturgeon{
class Basecall {
public:
    Basecall(const std::string model_path);

    void basecall(Sturgeon::ThreadSafeQueue<Sturgeon::Sig_Batch> & data_queue,
                  Sturgeon::ThreadSafeQueue<Sturgeon::Sig_Batch> & decode_queue,
                  bool is_half=true);
private:
    fs::path model_path;
};

}


#endif //STURGEON_BASECALL_H
