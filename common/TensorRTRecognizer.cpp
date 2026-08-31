#include "TensorRTRecognizer.h"

#include "TensorRTVersion.h"

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include "cuda_runtime_api.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <process.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace plate_recognition {
namespace {

template <typename T> void DestroyTensorRTObject(T *&object) {
  if (object != nullptr) {
    object->destroy();
    object = nullptr;
  }
}

class ErrorOnlyLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *message) noexcept override {
    if (severity <= Severity::kERROR && message != nullptr) {
      try {
        std::lock_guard<std::mutex> lock(mutex_);
        last_message_ = message;
      } catch (...) {
        // TensorRT requires ILogger::log to be noexcept.  Diagnostics must
        // never terminate inference if allocation or locking fails.
      }
    }
  }

  std::string LastMessage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_message_;
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    last_message_.clear();
  }

private:
  mutable std::mutex mutex_;
  std::string last_message_;
};

bool FileExists(const std::string &path) {
  std::ifstream file(path.c_str(), std::ios::binary);
  return file.good();
}

struct FileFingerprint {
  std::uint64_t size = 0U;
  std::uint64_t fnv1a = 14695981039346656037ULL;
};

void UpdateFingerprint(FileFingerprint &fingerprint, const void *data,
                       std::size_t size) {
  const unsigned char *bytes = static_cast<const unsigned char *>(data);
  for (std::size_t index = 0U; index < size; ++index) {
    fingerprint.fnv1a ^= static_cast<std::uint64_t>(bytes[index]);
    fingerprint.fnv1a *= 1099511628211ULL;
  }
  fingerprint.size += static_cast<std::uint64_t>(size);
}

bool FingerprintFile(const std::string &path, FileFingerprint &fingerprint) {
  fingerprint = FileFingerprint();
  std::ifstream input(path.c_str(), std::ios::binary);
  if (!input.good()) {
    return false;
  }
  char buffer[64 * 1024];
  while (input.good()) {
    input.read(buffer, static_cast<std::streamsize>(sizeof(buffer)));
    const std::streamsize count = input.gcount();
    if (count > 0) {
      UpdateFingerprint(fingerprint, buffer, static_cast<std::size_t>(count));
    }
  }
  return input.eof();
}

std::string FingerprintText(const FileFingerprint &fingerprint) {
  std::ostringstream text;
  text << "size=" << fingerprint.size << ",fnv1a64=" << std::hex
       << std::setfill('0') << std::setw(16) << fingerprint.fnv1a;
  return text.str();
}

bool ReadTextFile(const std::string &path, std::string &contents) {
  std::ifstream input(path.c_str(), std::ios::binary);
  if (!input.good()) {
    return false;
  }
  std::ostringstream text;
  text << input.rdbuf();
  if (!input.eof() && !input.good()) {
    return false;
  }
  contents = text.str();
  return true;
}

unsigned long CurrentProcessId() {
#if defined(_WIN32)
  return static_cast<unsigned long>(_getpid());
#else
  return static_cast<unsigned long>(getpid());
#endif
}

std::string TemporaryPath(const std::string &destination) {
  static std::atomic<unsigned long long> counter(0U);
  const unsigned long long sequence = counter.fetch_add(1U);
  const unsigned long long timestamp =
      static_cast<unsigned long long>(
          std::chrono::high_resolution_clock::now().time_since_epoch().count());
  std::ostringstream path;
  path << destination << ".tmp." << CurrentProcessId() << '.' << timestamp
       << '.' << sequence;
  return path.str();
}

bool ReplaceFileAtomically(const std::string &temporary,
                           const std::string &destination) {
#if defined(_WIN32)
  return MoveFileExA(temporary.c_str(), destination.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
  return std::rename(temporary.c_str(), destination.c_str()) == 0;
#endif
}

bool EndsWithCaseInsensitive(const std::string &text,
                             const std::string &suffix) {
  if (text.size() < suffix.size()) {
    return false;
  }
  const std::size_t offset = text.size() - suffix.size();
  for (std::size_t index = 0U; index < suffix.size(); ++index) {
    const unsigned char left =
        static_cast<unsigned char>(text[offset + index]);
    const unsigned char right = static_cast<unsigned char>(suffix[index]);
    if (std::tolower(left) != std::tolower(right)) {
      return false;
    }
  }
  return true;
}

class EngineCacheFileLock {
public:
  explicit EngineCacheFileLock(const std::string &path) {
    if (path.empty()) {
      locked_ = true;
      return;
    }
#if defined(_WIN32)
    handle_ = CreateFileA(path.c_str(), GENERIC_READ | GENERIC_WRITE,
                          FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                          OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle_ != INVALID_HANDLE_VALUE) {
      locked_ = LockFileEx(handle_, LOCKFILE_EXCLUSIVE_LOCK, 0U, MAXDWORD,
                           MAXDWORD, &overlapped_) != 0;
    }
#else
    descriptor_ = open(path.c_str(), O_CREAT | O_RDWR, 0666);
    if (descriptor_ >= 0) {
      locked_ = flock(descriptor_, LOCK_EX) == 0;
    }
#endif
  }

  ~EngineCacheFileLock() {
#if defined(_WIN32)
    if (handle_ != INVALID_HANDLE_VALUE) {
      if (locked_) {
        UnlockFileEx(handle_, 0U, MAXDWORD, MAXDWORD, &overlapped_);
      }
      CloseHandle(handle_);
    }
#else
    if (descriptor_ >= 0) {
      if (locked_) {
        flock(descriptor_, LOCK_UN);
      }
      close(descriptor_);
    }
#endif
  }

  bool IsLocked() const { return locked_; }

  EngineCacheFileLock(const EngineCacheFileLock &) = delete;
  EngineCacheFileLock &operator=(const EngineCacheFileLock &) = delete;

private:
  bool locked_ = false;
#if defined(_WIN32)
  HANDLE handle_ = INVALID_HANDLE_VALUE;
  OVERLAPPED overlapped_ = {};
#else
  int descriptor_ = -1;
#endif
};

std::string ShapeText(const TensorRTShape &shape) {
  std::ostringstream text;
  text << shape.n << 'x' << shape.c << 'x' << shape.h << 'x' << shape.w;
  return text.str();
}

nvinfer1::Dims ToDims(const TensorRTShape &shape) {
  nvinfer1::Dims dims;
  dims.nbDims = 4;
  dims.d[0] = shape.n;
  dims.d[1] = shape.c;
  dims.d[2] = shape.h;
  dims.d[3] = shape.w;
  return dims;
}

bool HasDynamicDimension(const nvinfer1::Dims &dims) {
  for (int index = 0; index < dims.nbDims; ++index) {
    if (dims.d[index] < 0) {
      return true;
    }
  }
  return false;
}

bool ShapeMatchesNetwork(const nvinfer1::Dims &network_dims,
                         const nvinfer1::Dims &concrete_dims) {
  if (network_dims.nbDims != concrete_dims.nbDims) {
    return false;
  }
  for (int index = 0; index < network_dims.nbDims; ++index) {
    if (concrete_dims.d[index] <= 0) {
      return false;
    }
    if (network_dims.d[index] >= 0 &&
        network_dims.d[index] != concrete_dims.d[index]) {
      return false;
    }
  }
  return true;
}

bool IsOrderedProfile(const nvinfer1::Dims &min_dims,
                      const nvinfer1::Dims &opt_dims,
                      const nvinfer1::Dims &max_dims) {
  if (min_dims.nbDims != opt_dims.nbDims ||
      opt_dims.nbDims != max_dims.nbDims) {
    return false;
  }
  for (int index = 0; index < min_dims.nbDims; ++index) {
    if (min_dims.d[index] <= 0 || min_dims.d[index] > opt_dims.d[index] ||
        opt_dims.d[index] > max_dims.d[index]) {
      return false;
    }
  }
  return true;
}

bool DimsEqual(const nvinfer1::Dims &lhs, const nvinfer1::Dims &rhs) {
  if (lhs.nbDims != rhs.nbDims) {
    return false;
  }
  for (int index = 0; index < lhs.nbDims; ++index) {
    if (lhs.d[index] != rhs.d[index]) {
      return false;
    }
  }
  return true;
}

bool BindingVolume(const nvinfer1::Dims &dims, std::size_t &volume) {
  volume = 1U;
  for (int index = 0; index < dims.nbDims; ++index) {
    if (dims.d[index] <= 0) {
      return false;
    }
    const std::size_t dimension = static_cast<std::size_t>(dims.d[index]);
    if (volume > std::numeric_limits<std::size_t>::max() / dimension) {
      return false;
    }
    volume *= dimension;
  }
  return true;
}

std::string ParserErrors(const nvonnxparser::IParser &parser) {
  std::ostringstream message;
  for (int index = 0; index < parser.getNbErrors(); ++index) {
    const nvonnxparser::IParserError *error = parser.getError(index);
    if (error != nullptr) {
      if (index > 0) {
        message << " | ";
      }
      message << error->desc();
    }
  }
  return message.str();
}

} // namespace

class TensorRTRecognizer::Impl {
public:
  Impl() = default;

  ~Impl() { Release(); }

  bool Init(const TensorRTRecognizerConfig &config) {
    Release();
    last_error_.clear();
    logger_.Clear();
    config_ = config;

    if (config_.onnx_path.empty() || config_.engine_path.empty()) {
      return Fail("ONNX path and engine path must not be empty");
    }
    if (config_.gpu_id < 0) {
      return Fail("GPU id must be non-negative");
    }

    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
      return FailCuda("Could not query CUDA devices", status);
    }
    if (config_.gpu_id >= device_count) {
      return Fail("CUDA device index is outside the available range");
    }
    status = cudaSetDevice(config_.gpu_id);
    if (status != cudaSuccess) {
      return FailCuda("cudaSetDevice failed", status);
    }
    cudaDeviceProp device_properties;
    status = cudaGetDeviceProperties(&device_properties, config_.gpu_id);
    if (status != cudaSuccess) {
      return FailCuda("cudaGetDeviceProperties failed", status);
    }
    device_compute_major_ = device_properties.major;
    device_compute_minor_ = device_properties.minor;
    device_name_ = device_properties.name;
    device_multiprocessor_count_ = device_properties.multiProcessorCount;
    device_total_global_memory_ =
        static_cast<std::uint64_t>(device_properties.totalGlobalMem);

    initLibNvInferPlugins(&logger_, "");

    const bool has_onnx_source = HasOnnxSource();
    EngineCacheFileLock cache_lock(
        has_onnx_source ? config_.engine_path + ".lock" : std::string());
    if (!cache_lock.IsLocked()) {
      return Fail("Could not lock TensorRT engine cache: " +
                  config_.engine_path);
    }
    const bool cache_requires_build =
        !FileExists(config_.engine_path) ||
        (has_onnx_source && !EngineCacheMatches());
    if (cache_requires_build) {
      if (!BuildEngine()) {
        return false;
      }
      if (has_onnx_source && !EngineCacheMatches()) {
        return Fail("TensorRT engine cache validation failed after build");
      }
    }
    if (!LoadEngine()) {
      ReleaseRuntimeObjects();
      if (!has_onnx_source || !BuildEngine() || !EngineCacheMatches() ||
          !LoadEngine()) {
        ReleaseRuntimeObjects();
        bindings_.clear();
        binding_capacity_.clear();
        return false;
      }
    }

    cudaError_t stream_status = cudaStreamCreate(&stream_);
    if (stream_status != cudaSuccess) {
      Release();
      return FailCuda("cudaStreamCreate failed", stream_status);
    }
    initialized_ = true;
    return true;
  }

  bool Infer(const std::vector<float> &input, const TensorRTShape &input_shape,
             std::vector<float> &output, std::vector<int> &output_shape) {
    output.clear();
    output_shape.clear();
    last_error_.clear();

    if (!initialized_ || engine_ == nullptr || context_ == nullptr) {
      return Fail("TensorRT recognizer is not initialized");
    }
    cudaError_t status = cudaSetDevice(config_.gpu_id);
    if (status != cudaSuccess) {
      return FailCuda("cudaSetDevice failed", status);
    }

    const nvinfer1::Dims requested_dims = ToDims(input_shape);
    const nvinfer1::Dims engine_input_dims =
        engine_->getBindingDimensions(input_index_);
    if (!ShapeMatchesNetwork(engine_input_dims, requested_dims)) {
      return Fail("Input dimensions do not match the TensorRT network");
    }
    if (HasDynamicDimension(engine_input_dims)) {
      if (!context_->setBindingDimensions(input_index_, requested_dims)) {
        return Fail("TensorRT rejected the requested input dimensions");
      }
    } else if (!DimsEqual(engine_input_dims, requested_dims)) {
      return Fail("Static TensorRT input dimensions do not match the request");
    }
    if (!context_->allInputDimensionsSpecified()) {
      return Fail("Not all TensorRT input dimensions were specified");
    }

    std::size_t input_count = 0U;
    if (!BindingVolume(requested_dims, input_count) ||
        input.size() != input_count) {
      return Fail("Input tensor element count is invalid");
    }

    const nvinfer1::Dims output_dims =
        context_->getBindingDimensions(output_index_);
    std::size_t output_count = 0U;
    if (!BindingVolume(output_dims, output_count)) {
      return Fail("TensorRT returned invalid output dimensions");
    }

    if (!EnsureBuffer(input_index_, input_count * sizeof(float)) ||
        !EnsureBuffer(output_index_, output_count * sizeof(float))) {
      return false;
    }

    status = cudaMemcpyAsync(bindings_[input_index_], input.data(),
                             input_count * sizeof(float),
                             cudaMemcpyHostToDevice, stream_);
    if (status != cudaSuccess) {
      return FailCuda("Input cudaMemcpyAsync failed", status);
    }
    if (!context_->enqueueV2(bindings_.data(), stream_, nullptr)) {
      return Fail("TensorRT enqueueV2 failed");
    }

    output.resize(output_count);
    status = cudaMemcpyAsync(output.data(), bindings_[output_index_],
                             output_count * sizeof(float),
                             cudaMemcpyDeviceToHost, stream_);
    if (status != cudaSuccess) {
      output.clear();
      return FailCuda("Output cudaMemcpyAsync failed", status);
    }
    status = cudaStreamSynchronize(stream_);
    if (status != cudaSuccess) {
      output.clear();
      return FailCuda("cudaStreamSynchronize failed", status);
    }

    output_shape.reserve(static_cast<std::size_t>(output_dims.nbDims));
    for (int index = 0; index < output_dims.nbDims; ++index) {
      output_shape.push_back(output_dims.d[index]);
    }
    return true;
  }

  void Release() {
    if (config_.gpu_id >= 0) {
      cudaSetDevice(config_.gpu_id);
    }
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
    for (std::size_t index = 0U; index < bindings_.size(); ++index) {
      if (bindings_[index] != nullptr) {
        cudaFree(bindings_[index]);
        bindings_[index] = nullptr;
      }
    }
    bindings_.clear();
    binding_capacity_.clear();
    ReleaseRuntimeObjects();
    input_index_ = -1;
    output_index_ = -1;
    input_name_.clear();
    output_name_.clear();
    initialized_ = false;
  }

  bool IsInitialized() const { return initialized_; }

  const std::string &LastError() const { return last_error_; }

  const std::string &InputName() const { return input_name_; }

  const std::string &OutputName() const { return output_name_; }

  bool GetInputShape(TensorRTShape &shape) const {
    shape = TensorRTShape();
    if (!initialized_ || engine_ == nullptr || input_index_ < 0) {
      return false;
    }
    const nvinfer1::Dims dims = engine_->getBindingDimensions(input_index_);
    if (dims.nbDims != 4) {
      return false;
    }
    shape.n = dims.d[0];
    shape.c = dims.d[1];
    shape.h = dims.d[2];
    shape.w = dims.d[3];
    return true;
  }

private:
  bool HasOnnxSource() const {
    return EndsWithCaseInsensitive(config_.onnx_path, ".onnx") &&
           FileExists(config_.onnx_path);
  }

  bool BuildCachePrefix(std::string &prefix) const {
    FileFingerprint onnx_fingerprint;
    if (!FingerprintFile(config_.onnx_path, onnx_fingerprint)) {
      return false;
    }
    std::ostringstream metadata;
    metadata << "PlateRecognitionTensorRTEngineCacheV1\n"
             << "onnx_" << FingerprintText(onnx_fingerprint) << '\n'
             << "tensorrt=" << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR
             << '.' << NV_TENSORRT_PATCH << '.' << NV_TENSORRT_BUILD << '\n'
#if defined(CUDART_VERSION)
             << "cudart=" << CUDART_VERSION << '\n'
#else
             << "cudart=unknown\n"
#endif
             << "compute_capability=" << device_compute_major_ << '.'
             << device_compute_minor_ << '\n'
             << "gpu_name=" << device_name_ << '\n'
             << "gpu_multiprocessors=" << device_multiprocessor_count_ << '\n'
             << "gpu_global_memory=" << device_total_global_memory_ << '\n'
             << "fp16=" << (config_.enable_fp16 ? 1 : 0) << '\n'
             << "workspace=" << config_.workspace_size << '\n'
             << "profile_min=" << ShapeText(config_.min_shape) << '\n'
             << "profile_opt=" << ShapeText(config_.opt_shape) << '\n'
             << "profile_max=" << ShapeText(config_.max_shape) << '\n';
    prefix = metadata.str();
    return true;
  }

  std::string CacheMetadata(const std::string &prefix,
                            const FileFingerprint &engine_fingerprint) const {
    return prefix + "engine_" + FingerprintText(engine_fingerprint) + "\n";
  }

  bool EngineCacheMatches() const {
    if (!HasOnnxSource() || !FileExists(config_.engine_path)) {
      return false;
    }
    std::string prefix;
    std::string saved_metadata;
    if (!BuildCachePrefix(prefix) ||
        !ReadTextFile(config_.engine_path + ".meta", saved_metadata) ||
        saved_metadata.compare(0U, prefix.size(), prefix) != 0) {
      return false;
    }
    FileFingerprint engine_fingerprint;
    if (!FingerprintFile(config_.engine_path, engine_fingerprint)) {
      return false;
    }
    return saved_metadata == CacheMetadata(prefix, engine_fingerprint);
  }

  bool PublishEngineCache(const void *engine_data, std::size_t engine_size,
                          const std::string &metadata) {
    const std::string metadata_path = config_.engine_path + ".meta";
    const std::string engine_temporary = TemporaryPath(config_.engine_path);
    const std::string metadata_temporary = TemporaryPath(metadata_path);

    bool engine_written = false;
    {
      std::ofstream output(engine_temporary.c_str(),
                           std::ios::binary | std::ios::trunc);
      if (output.good()) {
        output.write(static_cast<const char *>(engine_data),
                     static_cast<std::streamsize>(engine_size));
        engine_written = output.good();
      }
    }
    if (!engine_written) {
      std::remove(engine_temporary.c_str());
      return Fail("Could not write temporary TensorRT engine: " +
                  engine_temporary);
    }

    bool metadata_written = false;
    {
      std::ofstream output(metadata_temporary.c_str(),
                           std::ios::binary | std::ios::trunc);
      if (output.good()) {
        output.write(metadata.data(),
                     static_cast<std::streamsize>(metadata.size()));
        metadata_written = output.good();
      }
    }
    if (!metadata_written) {
      std::remove(engine_temporary.c_str());
      std::remove(metadata_temporary.c_str());
      return Fail("Could not write TensorRT cache metadata: " +
                  metadata_temporary);
    }

    if (!ReplaceFileAtomically(engine_temporary, config_.engine_path)) {
      std::remove(engine_temporary.c_str());
      std::remove(metadata_temporary.c_str());
      return Fail("Could not publish TensorRT engine atomically: " +
                  config_.engine_path);
    }
    if (!ReplaceFileAtomically(metadata_temporary, metadata_path)) {
      std::remove(metadata_temporary.c_str());
      return Fail("Could not publish TensorRT cache metadata atomically: " +
                  metadata_path);
    }
    return true;
  }

  bool BuildEngine() {
    if (!HasOnnxSource()) {
      return Fail("ONNX model does not exist: " + config_.onnx_path);
    }
    std::string cache_prefix_before;
    if (!BuildCachePrefix(cache_prefix_before)) {
      return Fail("Could not fingerprint ONNX model: " + config_.onnx_path);
    }

    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger_);
    if (builder == nullptr) {
      return FailWithLogger("createInferBuilder failed");
    }

    const std::uint32_t explicit_batch =
        1U << static_cast<std::uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition *network =
        builder->createNetworkV2(explicit_batch);
    nvinfer1::IBuilderConfig *builder_config = builder->createBuilderConfig();
    nvonnxparser::IParser *parser =
        network == nullptr ? nullptr
                           : nvonnxparser::createParser(*network, logger_);

    if (network == nullptr || builder_config == nullptr || parser == nullptr) {
      DestroyTensorRTObject(parser);
      DestroyTensorRTObject(builder_config);
      DestroyTensorRTObject(network);
      DestroyTensorRTObject(builder);
      return FailWithLogger("Failed to create TensorRT builder objects");
    }

    const bool parsed = parser->parseFromFile(
        config_.onnx_path.c_str(),
        static_cast<int>(nvinfer1::ILogger::Severity::kERROR));
    if (!parsed || network->getNbInputs() != 1 ||
        network->getNbOutputs() != 1) {
      std::string error = ParserErrors(*parser);
      if (error.empty()) {
        error = "ONNX model must parse with exactly one input and one output";
      }
      DestroyTensorRTObject(parser);
      DestroyTensorRTObject(builder_config);
      DestroyTensorRTObject(network);
      DestroyTensorRTObject(builder);
      return Fail(error);
    }

    builder_config->setMaxWorkspaceSize(config_.workspace_size);
    if (config_.enable_fp16) {
      if (!builder->platformHasFastFp16()) {
        DestroyTensorRTObject(parser);
        DestroyTensorRTObject(builder_config);
        DestroyTensorRTObject(network);
        DestroyTensorRTObject(builder);
        return Fail("FP16 was requested but is not supported by this platform");
      }
      builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    nvinfer1::IOptimizationProfile *profile = nullptr;
    nvinfer1::ITensor *input = network->getInput(0);
    const nvinfer1::Dims network_dims = input->getDimensions();
    if (HasDynamicDimension(network_dims)) {
      const nvinfer1::Dims min_dims = ToDims(config_.min_shape);
      const nvinfer1::Dims opt_dims = ToDims(config_.opt_shape);
      const nvinfer1::Dims max_dims = ToDims(config_.max_shape);
      if (!ShapeMatchesNetwork(network_dims, min_dims) ||
          !ShapeMatchesNetwork(network_dims, opt_dims) ||
          !ShapeMatchesNetwork(network_dims, max_dims) ||
          !IsOrderedProfile(min_dims, opt_dims, max_dims)) {
        DestroyTensorRTObject(parser);
        DestroyTensorRTObject(builder_config);
        DestroyTensorRTObject(network);
        DestroyTensorRTObject(builder);
        return Fail("TensorRT optimization profile is invalid");
      }

      profile = builder->createOptimizationProfile();
      if (profile == nullptr ||
          !profile->setDimensions(
              input->getName(), nvinfer1::OptProfileSelector::kMIN, min_dims) ||
          !profile->setDimensions(
              input->getName(), nvinfer1::OptProfileSelector::kOPT, opt_dims) ||
          !profile->setDimensions(
              input->getName(), nvinfer1::OptProfileSelector::kMAX, max_dims) ||
          !profile->isValid() ||
          builder_config->addOptimizationProfile(profile) < 0) {
        DestroyTensorRTObject(parser);
        DestroyTensorRTObject(builder_config);
        DestroyTensorRTObject(network);
        DestroyTensorRTObject(builder);
        return Fail("Failed to add TensorRT optimization profile");
      }
    }

    nvinfer1::IHostMemory *serialized =
        builder->buildSerializedNetwork(*network, *builder_config);
    if (serialized == nullptr) {
      DestroyTensorRTObject(parser);
      DestroyTensorRTObject(builder_config);
      DestroyTensorRTObject(network);
      DestroyTensorRTObject(builder);
      return FailWithLogger("TensorRT engine build failed");
    }

    std::string cache_prefix_after;
    const bool source_unchanged = BuildCachePrefix(cache_prefix_after) &&
                                  cache_prefix_after == cache_prefix_before;
    FileFingerprint engine_fingerprint;
    UpdateFingerprint(engine_fingerprint, serialized->data(),
                      serialized->size());
    const bool cache_published =
        source_unchanged &&
        PublishEngineCache(
            serialized->data(), serialized->size(),
            CacheMetadata(cache_prefix_after, engine_fingerprint));

    DestroyTensorRTObject(serialized);
    DestroyTensorRTObject(parser);
    DestroyTensorRTObject(builder_config);
    DestroyTensorRTObject(network);
    DestroyTensorRTObject(builder);

    if (!source_unchanged) {
      return Fail("ONNX model changed while TensorRT was building the engine");
    }
    if (!cache_published) {
      return false;
    }
    return true;
  }

  bool LoadEngine() {
    ReleaseRuntimeObjects();
    input_index_ = -1;
    output_index_ = -1;
    input_name_.clear();
    output_name_.clear();
    bindings_.clear();
    binding_capacity_.clear();

    std::ifstream input(config_.engine_path.c_str(),
                        std::ios::binary | std::ios::ate);
    if (!input.good()) {
      return Fail("TensorRT engine does not exist: " + config_.engine_path);
    }
    const std::streamoff size = input.tellg();
    if (size <= 0) {
      return Fail("TensorRT engine is empty: " + config_.engine_path);
    }
    input.seekg(0, std::ios::beg);
    std::vector<char> bytes(static_cast<std::size_t>(size));
    input.read(bytes.data(), static_cast<std::streamsize>(size));
    if (!input.good()) {
      return Fail("Failed to read TensorRT engine: " + config_.engine_path);
    }

    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (runtime_ == nullptr) {
      return FailWithLogger("createInferRuntime failed");
    }
    engine_ = runtime_->deserializeCudaEngine(bytes.data(), bytes.size());
    if (engine_ == nullptr) {
      return FailWithLogger("TensorRT engine deserialization failed");
    }
    context_ = engine_->createExecutionContext();
    if (context_ == nullptr) {
      return FailWithLogger("TensorRT execution context creation failed");
    }
    if (engine_->getNbBindings() != 2) {
      return Fail("TensorRT recognizer requires exactly two bindings");
    }

    for (int index = 0; index < engine_->getNbBindings(); ++index) {
      if (engine_->bindingIsInput(index)) {
        if (input_index_ >= 0) {
          return Fail("TensorRT recognizer has more than one input");
        }
        input_index_ = index;
        input_name_ = engine_->getBindingName(index);
      } else {
        if (output_index_ >= 0) {
          return Fail("TensorRT recognizer has more than one output");
        }
        output_index_ = index;
        output_name_ = engine_->getBindingName(index);
      }
      if (engine_->getBindingDataType(index) != nvinfer1::DataType::kFLOAT) {
        return Fail("TensorRT recognizer supports only FP32 I/O tensors");
      }
    }
    if (input_index_ < 0 || output_index_ < 0) {
      return Fail("TensorRT recognizer input/output bindings were not found");
    }

    bindings_.assign(static_cast<std::size_t>(engine_->getNbBindings()),
                     nullptr);
    binding_capacity_.assign(bindings_.size(), 0U);
    return true;
  }

  bool EnsureBuffer(int binding_index, std::size_t bytes) {
    const std::size_t index = static_cast<std::size_t>(binding_index);
    if (index >= bindings_.size() || bytes == 0U) {
      return Fail("Invalid TensorRT binding buffer request");
    }
    if (binding_capacity_[index] >= bytes && bindings_[index] != nullptr) {
      return true;
    }
    if (bindings_[index] != nullptr) {
      cudaFree(bindings_[index]);
      bindings_[index] = nullptr;
      binding_capacity_[index] = 0U;
    }
    cudaError_t status = cudaMalloc(&bindings_[index], bytes);
    if (status != cudaSuccess) {
      return FailCuda("cudaMalloc failed", status);
    }
    binding_capacity_[index] = bytes;
    return true;
  }

  void ReleaseRuntimeObjects() {
    DestroyTensorRTObject(context_);
    DestroyTensorRTObject(engine_);
    DestroyTensorRTObject(runtime_);
  }

  bool Fail(const std::string &message) {
    last_error_ = message;
    return false;
  }

  bool FailWithLogger(const std::string &prefix) {
    const std::string logger_message = logger_.LastMessage();
    if (logger_message.empty()) {
      return Fail(prefix);
    }
    return Fail(prefix + ": " + logger_message);
  }

  bool FailCuda(const std::string &prefix, cudaError_t status) {
    return Fail(prefix + ": " + cudaGetErrorString(status));
  }

private:
  TensorRTRecognizerConfig config_;
  ErrorOnlyLogger logger_;
  nvinfer1::IRuntime *runtime_ = nullptr;
  nvinfer1::ICudaEngine *engine_ = nullptr;
  nvinfer1::IExecutionContext *context_ = nullptr;
  cudaStream_t stream_ = nullptr;
  std::vector<void *> bindings_;
  std::vector<std::size_t> binding_capacity_;
  int input_index_ = -1;
  int output_index_ = -1;
  std::string input_name_;
  std::string output_name_;
  std::string last_error_;
  int device_compute_major_ = 0;
  int device_compute_minor_ = 0;
  std::string device_name_;
  int device_multiprocessor_count_ = 0;
  std::uint64_t device_total_global_memory_ = 0U;
  bool initialized_ = false;
};

TensorRTRecognizer::TensorRTRecognizer() : impl_(new Impl()) {}

TensorRTRecognizer::~TensorRTRecognizer() {
  delete impl_;
  impl_ = nullptr;
}

bool TensorRTRecognizer::Init(const TensorRTRecognizerConfig &config) {
  return impl_->Init(config);
}

bool TensorRTRecognizer::Infer(const std::vector<float> &input,
                               const TensorRTShape &input_shape,
                               std::vector<float> &output,
                               std::vector<int> &output_shape) {
  return impl_->Infer(input, input_shape, output, output_shape);
}

void TensorRTRecognizer::Release() { impl_->Release(); }

bool TensorRTRecognizer::IsInitialized() const {
  return impl_->IsInitialized();
}

const std::string &TensorRTRecognizer::LastError() const {
  return impl_->LastError();
}

const std::string &TensorRTRecognizer::InputName() const {
  return impl_->InputName();
}

const std::string &TensorRTRecognizer::OutputName() const {
  return impl_->OutputName();
}

bool TensorRTRecognizer::GetInputShape(TensorRTShape &shape) const {
  return impl_->GetInputShape(shape);
}

} // namespace plate_recognition
