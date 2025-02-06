//
// Created by yaohc on 2024/12/30.
//

#include "stitch_result.h"


torch::Tensor Sturgeon::cat_tensors(torch::Tensor &tensors, int64_t start, int64_t end) {
    std::vector<torch::Tensor> concatenated;
    concatenated.push_back(tensors[0].slice(0, 0, end));
    int64_t size = tensors.size(0);
    for (int64_t i = 1; i < size - 1; ++i) {
        concatenated.push_back(tensors[i].slice(0, start, end));
    }
    if (size > 2) {
        concatenated.push_back(tensors[tensors.size(0) - 1].slice(0, start));
    }
    return torch::cat(concatenated, 0);
}


void Sturgeon::stitch_result(
        Sturgeon::ThreadSafeQueue<std::pair<std::vector<Sturgeon::Chunk>, Sturgeon::Decode_Result>> &stitch_queue,
        Sturgeon::ThreadSafeQueue<Fastq_read> & write_queue,
        int64_t chunk_size, int32_t overlap, int32_t stride) {
    int64_t semi_overlap = overlap / 2;
    int64_t start = semi_overlap / stride, end = (chunk_size - semi_overlap) / stride;
    while (true) {
        auto [read_chunk, decode_result] = stitch_queue.pop_and_get();
        if (read_chunk.empty()) break;
//        std::cout << decode_result.seqs.sizes() << std::endl;
        auto seqs_ = Sturgeon::cat_tensors(decode_result.seqs, start, end);
        auto moves_ = Sturgeon::cat_tensors(decode_result.moves, start, end);
        auto quals_ = Sturgeon::cat_tensors(decode_result.quals, start, end);

        auto seqs_ptr = seqs_.accessor<uint8_t, 1>();
        auto moves_ptr = moves_.accessor<uint8_t, 1>();
        auto quals_ptr = quals_.accessor<uint8_t, 1>();

        std::string seq_str;
        for (int64_t i = 0; i < seqs_.size(0); ++i) {
            if (seqs_ptr[i] != 0) seq_str += Sturgeon::num2seq[seqs_ptr[i]];
        }

        std::string moves_str;
        moves_str.reserve(seq_str.length() * 2);
        for (int64_t i = 0; i < moves_.size(0); ++i) {
            if (moves_ptr[i]) {
                if (!moves_str.empty()) {
                    moves_str += ',';
                }
                moves_str += std::to_string(i * stride);
            }
        }

        float avg_qual = 0;
        std::string quals_str;
        quals_str.reserve(seq_str.length() * 2);
        for (int64_t i = 0; i < quals_.size(0); ++i) {
            if (quals_ptr[i]) {
                avg_qual += static_cast<float>(quals_ptr[i]) - 33;
                if (!quals_str.empty()) quals_str += ',';
                quals_str += static_cast<unsigned char>(quals_ptr[i]);
            }
        }
        if (!seq_str.empty())
            avg_qual /= static_cast<float>(seq_str.length());


        write_queue.push(Sturgeon::Fastq_read{
//                std::move(read_chunk[0].meta->read_id),
                read_chunk[0].meta->read_id,
                std::move(seq_str),
                std::move(quals_str),
                std::move(moves_str),
//                std::move(read_chunk[0].meta->file_name),
                read_chunk[0].meta->file_name,
                read_chunk[0].meta->trim_start,
                stride,
                avg_qual,
        });
    }
    write_queue.push(Sturgeon::Fastq_read());
}
