#include "DataLoader/DataLoader.h"
#include "CTC_Decoder/CTC_Decoder.h"
#include "Model/basecall.h"
#include "Stitch_Result/stitch_result.h"
#include "Fastq_Writer/writer.h"
#include <thread>
#include "3rd_party/argparse.hpp"
int main(int argc, char **argv) {
    argparse::ArgumentParser Sturgeon("Sturgeon", "1.0");

    argparse::ArgumentParser basecaller("basecall");
    basecaller.add_description("running basecalling for Nanopore sequencing data");
    basecaller.add_argument("h5_dir")
        .help("path to h5 directories");
    basecaller.add_argument("model_path")
        .help("trained module path");
    basecaller.add_argument("--output_dir")
        .help("output directory, sturgeon will output a pass.fastq and fail.fastq "
              "with a threshold of mean quality value of 10")
        .default_value("../");
    basecaller.add_argument("--num_sub_proc")
        .help("sub process to load h5 files, default set to 12")
        .default_value(static_cast<int32_t>(12))
        .scan<'i', int>();
    basecaller.add_argument("--batch_size")
        .help("batch size")
        .default_value(static_cast<int32_t>(1024))
        .scan<'i', int>();
    basecaller.add_argument("--chunk_size")
        .help("chunk size")
        .default_value(static_cast<int32_t>(6000))
        .scan<'i', int>();
    basecaller.add_argument("--overlap")
        .help("overlap size")
        .default_value(static_cast<int32_t>(500))
        .scan<'i', int>();
    basecaller.add_argument("--stride")
        .help("stride size")
        .default_value(static_cast<int32_t>(5))
        .scan<'i', int>();
    basecaller.add_argument("-output_moves")
            .help("output moves")
            .default_value(false)
            .implicit_value(true);
    basecaller.add_argument("-half")
            .help("whether use half precision for inference")
            .default_value(true)
            .implicit_value(false);

    Sturgeon.add_subparser(basecaller);

    try {
        Sturgeon.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << Sturgeon;
    }

    if (Sturgeon.is_subcommand_used("basecall")) {
        spdlog::info("Sturgeon basecalling start");

        fs::path h5_dir = basecaller.get<std::string>("h5_dir");
        fs::path model_path = basecaller.get<std::string>("model_path");
        fs::path output_dir = basecaller.get<std::string>("output_dir");
        auto num_sub_proc = basecaller.get<int32_t>("num_sub_proc");
        auto batch_size = basecaller.get<int32_t>("batch_size");
        auto chunk_size = basecaller.get<int32_t>("chunk_size");
        auto overlap = basecaller.get<int32_t>("overlap");
        auto stride = basecaller.get<int32_t>("stride");
        bool output_moves = basecaller.get<bool>("-output_moves");
        bool half = basecaller.get<bool>("-half");

        spdlog::info("Detected num of cuda device: {}", torch::cuda::device_count());

        at::set_num_threads(1);         // set libtorch intraop thread pool to 1
        at::set_num_interop_threads(1); // set libtorch interop thread pool to 1
        Sturgeon::ThreadSafeQueue<Sturgeon::Sig_Batch> data_queue;
        Sturgeon::ThreadSafeQueue<Sturgeon::Sig_Batch> decode_queue;
        Sturgeon::ThreadSafeQueue<std::pair<std::vector<Sturgeon::Chunk>, Sturgeon::Decode_Result>> stitch_queue;
        Sturgeon::ThreadSafeQueue<Sturgeon::Fastq_read>  write_queue;

        Sturgeon::DataLoader data_loader(h5_dir, batch_size, num_sub_proc, chunk_size, overlap, stride);
        Sturgeon::Basecall basecall(model_path);

        std::thread data_loader_thread(&Sturgeon::DataLoader::generate_batch, &data_loader, std::ref(data_queue));
        std::thread basecall_thread(&Sturgeon::Basecall::basecall, &basecall, std::ref(data_queue), std::ref(decode_queue), half);
        std::thread decode_thread(Sturgeon::ctc_decode_worker, std::ref(decode_queue), std::ref(stitch_queue));
        std::thread stitch_thread(Sturgeon::stitch_result, std::ref(stitch_queue), std::ref(write_queue), chunk_size, overlap, stride);
        std::thread write_thread(Sturgeon::write_fastq, std::ref(write_queue), std::ref(output_dir), output_moves);
        data_loader_thread.join();
        basecall_thread.join();
        decode_thread.join();
        stitch_thread.join();
        write_thread.join();
    }
    else {
        spdlog::error("wrong sub command!");
        return 1;
    }

    return 0;
}