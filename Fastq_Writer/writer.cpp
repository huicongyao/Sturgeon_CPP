//
// Created by yaohc on 2024/12/30.
//

#include "writer.h"

void Sturgeon::Fastq_read::write_fastq(std::ofstream &fasq_file, bool output_moves) {
    std::string info_str = "@" + read_id + "\t" +  \
                "qs:i:" + std::to_string(static_cast<int>(avg_q)) + "\t" + \
                "ts:i:" + std::to_string(trim_start) + "\t" + \
                "fn:Z:" + file_name + "\t" + \
                "sd:i:" + std::to_string(stride);
    if (output_moves) {
        info_str += "\t";
        info_str += "base2signal:Z:" + moves_str + "\n";
    }
    else {
        info_str += "\n";
    }
    fasq_file << info_str << seq_str << "\n" << "+\n" << quals_str << "\n";
}

bool Sturgeon::Fastq_read::is_empty() {
    return trim_start == -1;
}

void Sturgeon::write_fastq(Sturgeon::ThreadSafeQueue<Fastq_read> &write_queue, fs::path &output_dir, bool output_moves) {
    if (!fs::exists(output_dir))
        fs::create_directories(output_dir);
    std::ofstream fastq_pass = std::ofstream(output_dir / "pass.fastq");
    std::ofstream fastq_fail = std::ofstream(output_dir / "fail.fastq");
    while(true) {
        auto read = write_queue.pop_and_get();
        if (read.is_empty()) break;
        if (read.avg_q < 10.0) {
            read.write_fastq(fastq_fail, output_moves);
        } else {
            read.write_fastq(fastq_pass, output_moves);
        }
    }
    fastq_pass.close();
    fastq_fail.close();
    spdlog::info("write fastq done");
}
