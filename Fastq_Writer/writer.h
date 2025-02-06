//
// Created by yaohc on 2024/12/30.
//

#ifndef STURGEON_WRITER_H
#define STURGEON_WRITER_H
#include <string>
#include <fstream>
#include "../utils.h"

namespace Sturgeon {

    struct Fastq_read {
        std::string read_id;
        std::string seq_str;
        std::string quals_str;
        std::string moves_str;
        std::string file_name;
        int32_t trim_start = -1;
        int32_t stride = -1;
        float avg_q = -1;

        void write_fastq(std::ofstream & fasq_file, bool output_moves);
        bool is_empty();
    };


    void write_fastq(Sturgeon::ThreadSafeQueue<Fastq_read> & write_queue, fs::path & output_dir, bool output_moves = false);
}


#endif //STURGEON_WRITER_H
