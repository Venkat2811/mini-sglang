#include <torch/python.h>

#include <ATen/ATen.h>
#include <c10/util/intrusive_ptr.h>
#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gloo/allreduce.h"
#include "gloo/barrier.h"
#include "gloo/broadcast.h"
#include "gloo/config.h"
#include "gloo/math.h"
#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/store.h"
#include "gloo/transport/device.h"

#if GLOO_HAVE_TRANSPORT_UV
#include "gloo/transport/uv/device.h"
#endif

#if GLOO_HAVE_TRANSPORT_MYELON
#include "gloo/transport/myelon/device.h"
#endif

namespace py = pybind11;

namespace c10d {
namespace {

constexpr const char* kBackendName = "glooext";
constexpr const char* kTransportEnvVar = "MINISGL_C10D_GLOOEXT_TRANSPORT";
constexpr uint32_t kBroadcastTag = 1;
constexpr uint32_t kAllreduceTag = 2;
constexpr uint32_t kBarrierTag = 3;

class StoreAdapter final : public ::gloo::rendezvous::Store {
 public:
  explicit StoreAdapter(c10::intrusive_ptr<::c10d::Store> store)
      : store_(std::move(store)) {}

  void set(const std::string& key, const std::vector<char>& data) override {
    store_->set(key, toBytes(data));
  }

  std::vector<char> get(const std::string& key) override {
    return toChars(store_->get(key));
  }

  void wait(const std::vector<std::string>& keys) override {
    store_->wait(keys);
  }

  void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) override {
    store_->wait(keys, timeout);
  }

  bool has_v2_support() override {
    return store_->hasExtendedApi();
  }

  std::vector<std::vector<char>> multi_get(
      const std::vector<std::string>& keys) override {
    const auto values = store_->multiGet(keys);
    std::vector<std::vector<char>> out;
    out.reserve(values.size());
    for (const auto& value : values) {
      out.push_back(toChars(value));
    }
    return out;
  }

  void multi_set(
      const std::vector<std::string>& keys,
      const std::vector<std::vector<char>>& values) override {
    std::vector<std::vector<uint8_t>> converted;
    converted.reserve(values.size());
    for (const auto& value : values) {
      converted.push_back(toBytes(value));
    }
    store_->multiSet(keys, converted);
  }

  void append(const std::string& key, const std::vector<char>& data) override {
    store_->append(key, toBytes(data));
  }

  int64_t add(const std::string& key, int64_t value) override {
    return store_->add(key, value);
  }

 private:
  static std::vector<uint8_t> toBytes(const std::vector<char>& data) {
    return std::vector<uint8_t>(data.begin(), data.end());
  }

  static std::vector<char> toChars(const std::vector<uint8_t>& data) {
    return std::vector<char>(data.begin(), data.end());
  }

  c10::intrusive_ptr<::c10d::Store> store_;
};

class WorkGlooExt final : public ::c10d::Work {
 public:
  explicit WorkGlooExt(
      OpType opType,
      std::vector<at::Tensor> result = std::vector<at::Tensor>())
      : Work(-1, opType), result_(std::move(result)) {
    finish();
  }

  std::vector<at::Tensor> result() override {
    return result_;
  }

 private:
  std::vector<at::Tensor> result_;
};

void checkDenseCpuTensor(const at::Tensor& tensor, const char* opName) {
  TORCH_CHECK(
      tensor.device().is_cpu(),
      opName,
      " only supports CPU tensors in this prototype");
  TORCH_CHECK(
      tensor.layout() == c10::Layout::Strided,
      opName,
      " only supports dense tensors");
  TORCH_CHECK(
      tensor.is_contiguous(),
      opName,
      " only supports contiguous tensors");
}

template <typename T>
::gloo::AllreduceOptions::Func reduceFunctionFor(const ReduceOp& op) {
  switch (static_cast<ReduceOp::RedOpType>(op)) {
    case ReduceOp::SUM:
      return static_cast<void (*)(void*, const void*, const void*, size_t)>(
          &::gloo::sum<T>);
    case ReduceOp::MIN:
      return static_cast<void (*)(void*, const void*, const void*, size_t)>(
          &::gloo::min<T>);
    case ReduceOp::MAX:
      return static_cast<void (*)(void*, const void*, const void*, size_t)>(
          &::gloo::max<T>);
    case ReduceOp::PRODUCT:
      return static_cast<void (*)(void*, const void*, const void*, size_t)>(
          &::gloo::product<T>);
    default:
      TORCH_CHECK(false, "unsupported ReduceOp for glooext prototype");
  }
}

template <typename T>
void configureBroadcast(
    ::gloo::BroadcastOptions& opts,
    at::Tensor& tensor,
    int rank,
    int rootRank) {
  auto* ptr = tensor.data_ptr<T>();
  const auto elements = static_cast<size_t>(tensor.numel());
  opts.setOutput<T>(ptr, elements);
  if (rank == rootRank) {
    opts.setInput<T>(ptr, elements);
  }
}

template <typename T>
void configureAllreduce(
    ::gloo::AllreduceOptions& opts,
    at::Tensor& input,
    at::Tensor& output,
    const ReduceOp& reduceOp) {
  auto* inputPtr = input.data_ptr<T>();
  auto* outputPtr = output.data_ptr<T>();
  const auto elements = static_cast<size_t>(input.numel());
  opts.setInput<T>(inputPtr, elements);
  opts.setOutput<T>(outputPtr, elements);
  opts.setReduceFunction(reduceFunctionFor<T>(reduceOp));
  opts.setAlgorithm(::gloo::AllreduceOptions::Algorithm::RING);
}

std::shared_ptr<::gloo::transport::Device> createDeviceFromEnv() {
  const char* value = std::getenv(kTransportEnvVar);
  const std::string transport = value == nullptr ? "uv" : std::string(value);

#if GLOO_HAVE_TRANSPORT_MYELON
  if (transport == "myelon") {
    ::gloo::transport::myelon::attr attr;
    return ::gloo::transport::myelon::CreateDevice(attr);
  }
#endif

#if GLOO_HAVE_TRANSPORT_UV
  if (transport == "uv") {
    ::gloo::transport::uv::attr attr;
    return ::gloo::transport::uv::CreateDevice(attr);
  }
#endif

  TORCH_CHECK(
      false,
      "unsupported transport in ",
      kTransportEnvVar,
      ": ",
      transport);
}

class BackendGlooExt final : public Backend {
 public:
  BackendGlooExt(
      const c10::intrusive_ptr<::c10d::Store>& store,
      int rank,
      int size,
      const std::chrono::milliseconds& timeout)
      : Backend(rank, size),
        options_(c10::make_intrusive<Backend::Options>(kBackendName, timeout)),
        device_(createDeviceFromEnv()),
        context_(std::make_shared<::gloo::rendezvous::Context>(rank, size)) {
    auto rendezvousStore =
        std::make_shared<StoreAdapter>(store->clone());
    context_->setTimeout(timeout);
    context_->connectFullMesh(rendezvousStore, device_);
  }

  const std::string getBackendName() const override {
    return kBackendName;
  }

  c10::intrusive_ptr<Backend::Options> getBackendOptions() override {
    return options_;
  }

  void setTimeout(std::chrono::milliseconds timeout) override {
    options_->timeout = timeout;
    context_->setTimeout(timeout);
  }

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts) override {
    TORCH_CHECK(!tensors.empty(), "broadcast expects at least one tensor");
    TORCH_CHECK(opts.rootTensor == 0, "glooext only supports rootTensor == 0");
    for (auto& tensor : tensors) {
      checkDenseCpuTensor(tensor, "broadcast");
      ::gloo::BroadcastOptions glooOpts(context_);
      glooOpts.setRoot(static_cast<int>(opts.rootRank));
      glooOpts.setTag(kBroadcastTag);
      glooOpts.setTimeout(resolveTimeout(opts.timeout));
      switch (tensor.scalar_type()) {
        case at::ScalarType::Byte:
          configureBroadcast<uint8_t>(
              glooOpts, tensor, getRank(), static_cast<int>(opts.rootRank));
          break;
        case at::ScalarType::Char:
          configureBroadcast<int8_t>(
              glooOpts, tensor, getRank(), static_cast<int>(opts.rootRank));
          break;
        case at::ScalarType::Short:
          configureBroadcast<int16_t>(
              glooOpts, tensor, getRank(), static_cast<int>(opts.rootRank));
          break;
        case at::ScalarType::Int:
          configureBroadcast<int32_t>(
              glooOpts, tensor, getRank(), static_cast<int>(opts.rootRank));
          break;
        case at::ScalarType::Long:
          configureBroadcast<int64_t>(
              glooOpts, tensor, getRank(), static_cast<int>(opts.rootRank));
          break;
        case at::ScalarType::Float:
          configureBroadcast<float>(
              glooOpts, tensor, getRank(), static_cast<int>(opts.rootRank));
          break;
        case at::ScalarType::Double:
          configureBroadcast<double>(
              glooOpts, tensor, getRank(), static_cast<int>(opts.rootRank));
          break;
        case at::ScalarType::Bool:
          configureBroadcast<bool>(
              glooOpts, tensor, getRank(), static_cast<int>(opts.rootRank));
          break;
        default:
          TORCH_CHECK(false, "broadcast does not support dtype ", tensor.scalar_type());
      }
      ::gloo::broadcast(glooOpts);
    }
    return c10::make_intrusive<WorkGlooExt>(OpType::BROADCAST, tensors);
  }

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts) override {
    TORCH_CHECK(!tensors.empty(), "allreduce expects at least one tensor");
    for (auto& tensor : tensors) {
      checkDenseCpuTensor(tensor, "allreduce");
      auto output = at::empty_like(tensor);
      ::gloo::AllreduceOptions glooOpts(context_);
      glooOpts.setTag(kAllreduceTag);
      glooOpts.setTimeout(resolveTimeout(opts.timeout));
      switch (tensor.scalar_type()) {
        case at::ScalarType::Byte:
          configureAllreduce<uint8_t>(glooOpts, tensor, output, opts.reduceOp);
          break;
        case at::ScalarType::Char:
          configureAllreduce<int8_t>(glooOpts, tensor, output, opts.reduceOp);
          break;
        case at::ScalarType::Short:
          configureAllreduce<int16_t>(glooOpts, tensor, output, opts.reduceOp);
          break;
        case at::ScalarType::Int:
          configureAllreduce<int32_t>(glooOpts, tensor, output, opts.reduceOp);
          break;
        case at::ScalarType::Long:
          configureAllreduce<int64_t>(glooOpts, tensor, output, opts.reduceOp);
          break;
        case at::ScalarType::Float:
          configureAllreduce<float>(glooOpts, tensor, output, opts.reduceOp);
          break;
        case at::ScalarType::Double:
          configureAllreduce<double>(glooOpts, tensor, output, opts.reduceOp);
          break;
        case at::ScalarType::Bool:
          configureAllreduce<bool>(glooOpts, tensor, output, opts.reduceOp);
          break;
        default:
          TORCH_CHECK(false, "allreduce does not support dtype ", tensor.scalar_type());
      }
      ::gloo::allreduce(glooOpts);
      tensor.copy_(output);
    }
    return c10::make_intrusive<WorkGlooExt>(OpType::ALLREDUCE, tensors);
  }

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts) override {
    ::gloo::BarrierOptions glooOpts(context_);
    glooOpts.setTag(kBarrierTag);
    glooOpts.setTimeout(resolveTimeout(opts.timeout));
    ::gloo::barrier(glooOpts);
    return c10::make_intrusive<WorkGlooExt>(OpType::BARRIER);
  }

  static c10::intrusive_ptr<Backend> createBackend(
      const c10::intrusive_ptr<::c10d::Store>& store,
      int rank,
      int size,
      const std::chrono::duration<float>& timeout) {
    return c10::make_intrusive<BackendGlooExt>(
        store,
        rank,
        size,
        std::chrono::duration_cast<std::chrono::milliseconds>(timeout));
  }

 private:
  std::chrono::milliseconds resolveTimeout(std::chrono::milliseconds timeout) const {
    return timeout == kUnsetTimeout ? options_->timeout : timeout;
  }

  c10::intrusive_ptr<Backend::Options> options_;
  std::shared_ptr<::gloo::transport::Device> device_;
  std::shared_ptr<::gloo::rendezvous::Context> context_;
};

c10::intrusive_ptr<Backend> createBackendGlooExt(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int size,
    const std::chrono::duration<float>& timeout) {
  return BackendGlooExt::createBackend(store, rank, size, timeout);
}

} // namespace
} // namespace c10d

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  auto dist = py::module::import("torch.distributed");
  auto register_backend = dist.attr("Backend").attr("register_backend");
  register_backend(
      "glooext",
      py::cpp_function(&::c10d::createBackendGlooExt),
      false,
      std::vector<std::string>{"cpu"});

  m.def("createBackendGlooExt", &::c10d::createBackendGlooExt);
}
