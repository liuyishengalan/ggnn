#include <ggnn/base/gpu_instance.cuh>
#include <ggnn/base/dataset.cuh>
#include <ggnn/base/graph_config.h>
#include <ggnn/cuda_utils/check.cuh>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

using namespace ggnn;

using KeyT = int32_t;
using ValueT = float;
using BaseT = float;

DEFINE_string(base, "", "Path to sift_base.fvecs");
DEFINE_string(query, "", "Path to sift_query.fvecs");
DEFINE_string(graph_dir, "./ggnn_async_graph", "Directory for GGNN graph shards");

DEFINE_uint32(k_build, 24, "KBuild");
DEFINE_double(tau_build, 0.5, "tau_build");
DEFINE_uint32(refinement_iterations, 2, "refinement iterations");

DEFINE_uint32(k_query, 10, "KQuery");
DEFINE_uint32(max_iterations, 200, "max query iterations");
DEFINE_double(tau_query, 0.51, "tau_query");

DEFINE_uint32(shard_size, 0, "Shard size, 0 means no sharding (use full base as one shard)");
DEFINE_int32(gpu, 0, "CUDA device id");

#define TEST_CHECK(cond, msg)                                  \
  do {                                                         \
    if (!(cond)) {                                             \
      std::cerr << "[FAIL] " << msg << std::endl;              \
      std::exit(1);                                            \
    }                                                          \
  } while (0)

static bool compare_results(const Results<KeyT, ValueT>& a,
                            const Results<KeyT, ValueT>& b,
                            float eps = 1e-5f)
{
  auto to_cpu = [](const auto& data) {
    using T = typename std::remove_cv_t<std::remove_reference_t<decltype(data)>>::value_type;
    auto host = Dataset<T>::empty(data.N, data.D, true);
    data.copyTo(host);
    CHECK_CUDA(cudaStreamSynchronize(0));
    return host;
  };

  auto a_ids = to_cpu(a.ids);
  auto a_dists = to_cpu(a.dists);
  auto b_ids = to_cpu(b.ids);
  auto b_dists = to_cpu(b.dists);

  if (a_ids.numel() != b_ids.numel()) {
    std::cerr << "ids numel mismatch: " << a_ids.numel() << " vs " << b_ids.numel() << "\n";
    return false;
  }
  if (a_dists.numel() != b_dists.numel()) {
    std::cerr << "dists numel mismatch: " << a_dists.numel() << " vs " << b_dists.numel() << "\n";
    return false;
  }

  for (uint64_t i = 0; i < a_ids.numel(); ++i) {
    if (a_ids[i] != b_ids[i]) {
      std::cerr << "ID mismatch at " << i << ": " << a_ids[i] << " vs " << b_ids[i] << "\n";
      return false;
    }
  }

  for (uint64_t i = 0; i < a_dists.numel(); ++i) {
    if (std::fabs(a_dists[i] - b_dists[i]) > eps) {
      std::cerr << "Distance mismatch at " << i << ": "
                << a_dists[i] << " vs " << b_dists[i] << "\n";
      return false;
    }
  }

  return true;
}

__global__ void dummy_check_kernel(const KeyT* ids, const ValueT* dists, int* flag)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    KeyT id0 = ids[0];
    ValueT d0 = dists[0];
    (void)id0;
    (void)d0;
    flag[0] = 1;
  }
}

static size_t compute_reserved_gpu_memory(
    const Dataset<BaseT>& query,
    uint32_t k_query,
    uint32_t num_shards)
{
  const size_t query_size = query.required_size_bytes();
  const size_t result_size =
      static_cast<size_t>(query.N) * k_query * (sizeof(KeyT) + sizeof(ValueT));

  // 多 shard 时，本地结果是 Q x (K * num_shards)，排序时还会再来一份
  const size_t shard_result_size =
      (num_shards > 1) ? (result_size * num_shards * 2) : 0UL;

  return query_size + result_size + shard_result_size;
}

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
  google::InstallFailureSignalHandler();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  TEST_CHECK(!FLAGS_base.empty(), "Please provide --base");
  TEST_CHECK(!FLAGS_query.empty(), "Please provide --query");
  TEST_CHECK(std::filesystem::exists(FLAGS_base), "Base file does not exist");
  TEST_CHECK(std::filesystem::exists(FLAGS_query), "Query file does not exist");

  CHECK_CUDA(cudaSetDevice(FLAGS_gpu));

  std::filesystem::create_directories(FLAGS_graph_dir);

  std::cout << "[INFO] Loading datasets...\n";
  Dataset<BaseT> base = Dataset<BaseT>::load(FLAGS_base, 0, std::numeric_limits<uint32_t>::max(), true);
  Dataset<BaseT> query = Dataset<BaseT>::load(FLAGS_query, 0, std::numeric_limits<uint32_t>::max(), true);

  TEST_CHECK(base.data() != nullptr, "Failed to load base dataset");
  TEST_CHECK(query.data() != nullptr, "Failed to load query dataset");
  TEST_CHECK(base.D == query.D, "Base/query dimensions do not match");

  std::cout << "[INFO] base : N=" << base.N  << " D=" << base.D  << "\n";
  std::cout << "[INFO] query: N=" << query.N << " D=" << query.D << "\n";

  uint32_t N_shard = FLAGS_shard_size == 0 ? static_cast<uint32_t>(base.N) : FLAGS_shard_size;
  TEST_CHECK(base.N % N_shard == 0, "base.N must be divisible by shard_size");
  const uint32_t num_shards = static_cast<uint32_t>(base.N / N_shard);

  std::cout << "[INFO] shard size = " << N_shard << ", num_shards = " << num_shards << "\n";

  GraphParameters graph_params{
      .N = N_shard,
      .D = base.D,
      .KBuild = FLAGS_k_build,
      .graph_dir = FLAGS_graph_dir,
  };
  GraphConfig graph_config{graph_params};

  GPUContext gpu_ctx{FLAGS_gpu};

  ShardingConfiguration shard_config{
      .N_shard = N_shard,
      .device_index = 0,
      .num_shards = num_shards,
      .cpu_memory_limit = std::numeric_limits<size_t>::max(),
  };

  GPUInstance<KeyT, ValueT, BaseT> gpu(gpu_ctx, shard_config, graph_config);

  const size_t reserved_gpu_memory =
      compute_reserved_gpu_memory(query, FLAGS_k_query, num_shards);

  std::cout << "[INFO] reserved GPU memory = "
            << static_cast<double>(reserved_gpu_memory) / (1024.0 * 1024.0 * 1024.0)
            << " GiB\n";

  const auto graph_probe = std::filesystem::path(FLAGS_graph_dir) / "part_0.ggnn";
  if (std::filesystem::exists(graph_probe)) {
    std::cout << "[INFO] Loading existing graph from " << FLAGS_graph_dir << "\n";
    gpu.load(base, graph_config, reserved_gpu_memory);
  } else {
    std::cout << "[INFO] Building graph into " << FLAGS_graph_dir << "\n";
    gpu.build(base, graph_config,
              static_cast<float>(FLAGS_tau_build),
              FLAGS_refinement_iterations,
              DistanceMeasure::Euclidean,
              reserved_gpu_memory);
  }

  cudaStream_t stream_compute = nullptr;
  cudaStream_t stream_follow = nullptr;
  CHECK_CUDA(cudaStreamCreate(&stream_compute));
  CHECK_CUDA(cudaStreamCreate(&stream_follow));

  // ============================================================
  // Test 1: correctness
  // ============================================================
  std::cout << "\n[TEST 1] correctness: sync vs async\n";

  auto r_sync = gpu.query(query,
                          FLAGS_k_query,
                          FLAGS_max_iterations,
                          static_cast<float>(FLAGS_tau_query),
                          DistanceMeasure::Euclidean);

  auto h_async = gpu.queryLocalAsync(query,
                                     FLAGS_k_query,
                                     FLAGS_max_iterations,
                                     static_cast<float>(FLAGS_tau_query),
                                     DistanceMeasure::Euclidean,
                                     stream_compute);

  TEST_CHECK(h_async.done_event != nullptr, "async done_event is null");

  CHECK_CUDA(cudaEventSynchronize(h_async.done_event));

  bool ok = compare_results(r_sync, h_async.results);
  TEST_CHECK(ok, "sync query and async query results do not match");

  std::cout << "[PASS] correctness\n";

  // ============================================================
  // Test 2: return latency
  // ============================================================
  std::cout << "\n[TEST 2] async return latency\n";

  auto t0 = std::chrono::steady_clock::now();
  auto h_async2 = gpu.queryLocalAsync(query,
                                      FLAGS_k_query,
                                      FLAGS_max_iterations,
                                      static_cast<float>(FLAGS_tau_query),
                                      DistanceMeasure::Euclidean,
                                      stream_compute);
  auto t1 = std::chrono::steady_clock::now();

  TEST_CHECK(h_async2.done_event != nullptr, "async2 done_event is null");

  CHECK_CUDA(cudaEventSynchronize(h_async2.done_event));
  auto t2 = std::chrono::steady_clock::now();

  const auto launch_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  const auto total_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count();

  std::cout << "launch return time = " << launch_us << " us\n";
  std::cout << "total time        = " << total_us  << " us\n";

  TEST_CHECK(launch_us < total_us,
             "queryLocalAsync() does not appear to return before completion");

  std::cout << "[PASS] return latency\n";

  // ============================================================
  // Test 3: stream dependency
  // ============================================================
  std::cout << "\n[TEST 3] stream wait event\n";

  auto h_async3 = gpu.queryLocalAsync(query,
                                      FLAGS_k_query,
                                      FLAGS_max_iterations,
                                      static_cast<float>(FLAGS_tau_query),
                                      DistanceMeasure::Euclidean,
                                      stream_compute);

  TEST_CHECK(h_async3.done_event != nullptr, "async3 done_event is null");

  int* d_flag = nullptr;
  int h_flag = 0;
  CHECK_CUDA(cudaMalloc(&d_flag, sizeof(int)));
  CHECK_CUDA(cudaMemsetAsync(d_flag, 0, sizeof(int), stream_follow));

  CHECK_CUDA(cudaStreamWaitEvent(stream_follow, h_async3.done_event, 0));

  dummy_check_kernel<<<1, 1, 0, stream_follow>>>(
      h_async3.results.ids.data(),
      h_async3.results.dists.data(),
      d_flag);

  CHECK_CUDA(cudaMemcpyAsync(&h_flag, d_flag, sizeof(int),
                             cudaMemcpyDeviceToHost, stream_follow));
  CHECK_CUDA(cudaStreamSynchronize(stream_follow));

  TEST_CHECK(h_flag == 1, "stream_follow failed to consume async results after done_event");

  std::cout << "[PASS] stream dependency\n";

  CHECK_CUDA(cudaFree(d_flag));

  CHECK_CUDA(cudaEventDestroy(h_async.done_event));
  CHECK_CUDA(cudaEventDestroy(h_async2.done_event));
  CHECK_CUDA(cudaEventDestroy(h_async3.done_event));

  CHECK_CUDA(cudaStreamDestroy(stream_compute));
  CHECK_CUDA(cudaStreamDestroy(stream_follow));

  std::cout << "\nAll async tests passed.\n";
  return 0;
}