// Row gather for DSV41_ENGRAM_SSD (engram_ssd.py): out[i] = table[idx[i]] for one or two (table, idx, out) triples,
// on a pool of native threads. A cold row is a page fault on the mmap'd shard that blocks the thread taking it, so rows
// are handed out a few at a time to whichever thread is free. On the Python pool this replaces, a decode step (a few
// hundred rows, most of them in the page cache) cost mostly interpreter overhead.
#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

struct Part {
  torch::Tensor table, idx, out;  // held until the job is freed, so Python may drop its references early
  const uint8_t* src;
  const int64_t* ids;
  uint8_t* dst;
  int64_t row_bytes;
};

struct Job {
  std::vector<Part> parts;
  int64_t n = 0, block = 1, blocks = 0;
  std::atomic<int64_t> next{0}, left{0};
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;

  // Claims and copies blocks until none are left to claim.
  void work() {
    for (int64_t b; (b = next.fetch_add(1)) < blocks;) {
      const int64_t lo = b * block, hi = std::min(n, lo + block);
      for (const Part& p : parts)
        for (int64_t i = lo; i < hi; ++i) std::memcpy(p.dst + i * p.row_bytes, p.src + p.ids[i] * p.row_bytes, p.row_bytes);
      if (left.fetch_sub(1) == 1) {
        std::lock_guard<std::mutex> g(mu);
        done = true;
        cv.notify_all();
      }
    }
  }
};

class Pool {
 public:
  explicit Pool(int threads) : threads_(threads) {
    // Detached and never destroyed: a joinable std::thread destroyed at interpreter exit would std::terminate.
    for (int i = 0; i < threads; ++i) std::thread([this] { loop(); }).detach();
  }

  // detach: nobody will wait() (prefetch); the job is freed once its last block is copied.
  int64_t submit(const std::shared_ptr<Job>& job, bool detach) {
    int64_t id = -1;
    {
      std::lock_guard<std::mutex> g(mu_);
      if (!detach) jobs_[id = next_id_++] = job;
      queue_.push_back(job);
    }
    // One wake per block up to the pool size; past a few, one broadcast is cheaper for the caller.
    const int64_t wake = std::min<int64_t>(job->blocks, threads_);
    if (wake > 4) {
      cv_.notify_all();
    } else {
      for (int64_t i = 0; i < wake; ++i) cv_.notify_one();
    }
    return id;
  }

  void wait(int64_t id) {
    std::shared_ptr<Job> job;
    {
      std::lock_guard<std::mutex> g(mu_);
      auto it = jobs_.find(id);
      TORCH_CHECK(it != jobs_.end(), "engram gather: unknown job ", id);
      job = std::move(it->second);
      jobs_.erase(it);
    }
    job->work();  // the waiting thread takes what is left
    std::unique_lock<std::mutex> l(job->mu);
    job->cv.wait(l, [&] { return job->done; });
  }

 private:
  void loop() {
    for (;;) {
      std::shared_ptr<Job> job;
      {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [&] {
          while (!queue_.empty() && queue_.front()->next.load() >= queue_.front()->blocks) queue_.pop_front();
          return !queue_.empty();
        });
        job = queue_.front();
      }
      job->work();
    }
  }

  const int threads_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<Job>> queue_;
  std::unordered_map<int64_t, std::shared_ptr<Job>> jobs_;
  int64_t next_id_ = 0;
};

std::vector<Pool*> pools;  // leaked on purpose, see Pool()
std::mutex pools_mu;

Pool& pool(int64_t handle) {
  std::lock_guard<std::mutex> g(pools_mu);
  TORCH_CHECK(handle >= 0 && handle < static_cast<int64_t>(pools.size()), "engram gather: bad pool ", handle);
  return *pools[handle];
}

void add_part(Job& job, const torch::Tensor& table, const torch::Tensor& idx, const torch::Tensor& out) {
  TORCH_CHECK(table.device().is_cpu() && idx.device().is_cpu() && out.device().is_cpu(), "engram gather: CPU tensors only");
  TORCH_CHECK(table.dim() == 2 && out.dim() == 2 && table.scalar_type() == torch::kUInt8 && out.scalar_type() == torch::kUInt8,
              "engram gather: tables are [rows, row_bytes] uint8");
  TORCH_CHECK(table.is_contiguous() && out.is_contiguous() && idx.is_contiguous(), "engram gather: contiguous tensors only");
  TORCH_CHECK(idx.scalar_type() == torch::kInt64 && idx.dim() == 1, "engram gather: idx is 1-D int64");
  TORCH_CHECK(table.size(1) == out.size(1) && out.size(0) >= idx.numel(), "engram gather: shape mismatch");
  TORCH_CHECK(job.parts.empty() || idx.numel() == job.n, "engram gather: parts differ in length");
  job.n = idx.numel();
  Part p{table, idx, out, table.data_ptr<uint8_t>(), idx.data_ptr<int64_t>(), out.data_ptr<uint8_t>(), table.size(1)};
  // An out-of-range id would read past the mapping: check here rather than per copy.
  const int64_t rows = table.size(0);
  for (int64_t i = 0; i < job.n; ++i)
    TORCH_CHECK(p.ids[i] >= 0 && p.ids[i] < rows, "engram gather: row id ", p.ids[i], " out of range");
  job.parts.push_back(std::move(p));
}

}  // namespace

int64_t new_pool(int64_t threads) {
  TORCH_CHECK(threads > 0, "engram gather: need at least one thread");
  std::lock_guard<std::mutex> g(pools_mu);
  pools.push_back(new Pool(static_cast<int>(threads)));
  return static_cast<int64_t>(pools.size()) - 1;
}

// Starts out_a[i] = table_a[idx_a[i]] (and the same for b if table_b is given). Returns a job id for wait(), or -1 with
// detach=true.
int64_t start(int64_t handle, torch::Tensor table_a, torch::Tensor idx_a, torch::Tensor out_a,
              c10::optional<torch::Tensor> table_b, c10::optional<torch::Tensor> idx_b,
              c10::optional<torch::Tensor> out_b, int64_t block, bool detach) {
  auto job = std::make_shared<Job>();
  add_part(*job, table_a, idx_a, out_a);
  if (table_b.has_value()) {
    TORCH_CHECK(idx_b.has_value() && out_b.has_value(), "engram gather: part b needs table, idx and out");
    add_part(*job, *table_b, *idx_b, *out_b);
  }
  job->block = std::max<int64_t>(1, block);
  job->blocks = (job->n + job->block - 1) / job->block;
  job->left = job->blocks;
  job->done = job->blocks == 0;
  return pool(handle).submit(job, detach);
}

void wait(int64_t handle, int64_t job) {
  pybind11::gil_scoped_release release;
  pool(handle).wait(job);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("new_pool", &new_pool);
  m.def("start", &start);
  m.def("wait", &wait);
}
