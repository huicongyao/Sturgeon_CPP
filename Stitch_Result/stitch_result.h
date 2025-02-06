//
// Created by yaohc on 2024/12/30.
//

#ifndef STURGEON_STITCH_RESULT_H
#define STURGEON_STITCH_RESULT_H
#include <string>
#include "../DataLoader/DataLoader.h"
#include "../CTC_Decoder/CTC_Decoder.h"
#include "../Fastq_Writer/writer.h"


namespace Sturgeon {

constexpr std::array<uint8_t, 5> num2seq = {'N', 'A', 'C', 'G', 'T'};

torch::Tensor cat_tensors(torch::Tensor & tensors,
                                 int64_t start,
                                 int64_t end);

void stitch_result(Sturgeon::ThreadSafeQueue<std::pair<std::vector<Sturgeon::Chunk>, Sturgeon::Decode_Result>> & stitch_queue,
                   Sturgeon::ThreadSafeQueue<Fastq_read> & write_queue,
                   int64_t chunk_size = 6000,
                   int32_t overlap = 500,
                   int32_t stride = 5);

}


#endif //STURGEON_STITCH_RESULT_H
