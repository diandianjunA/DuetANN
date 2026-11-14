#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include <faiss/impl/ProductQuantizer.h>
#include <faiss/VectorTransform.h>

#include "common/vector_dtype.hh"
#include "gpu_search/pq_index.hh"

namespace {

std::vector<float> read_vectors(const std::filesystem::path& path, u32& dim,
                                size_t& count) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("failed to open vector file");
  u32 rows = 0;
  input.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  if (!input || rows == 0 || dim == 0) {
    throw std::runtime_error("invalid vector header");
  }
  count = std::min<size_t>(rows, 64);
  std::vector<float> values(count * dim);
  if (path.extension() == ".u8bin") {
    std::vector<u8> encoded(count * dim);
    input.read(reinterpret_cast<char*>(encoded.data()),
               static_cast<std::streamsize>(encoded.size()));
    if (!input) throw std::runtime_error("truncated vector payload");
    for (size_t index = 0; index < count; ++index) {
      decode_storage_vector_to_float(
        encoded.data() + index * dim, VectorDType::uint8, dim,
        values.data() + index * dim);
    }
    return values;
  }
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!input) throw std::runtime_error("truncated vector payload");
  return values;
}

size_t audit(const gpu_search::pq::Model& model,
             const std::vector<float>& vectors, size_t count) {
  faiss::LinearTransform transform(model.dim, model.dim, false);
  transform.A = model.rotation;
  transform.is_trained = true;
  transform.is_orthonormal = true;
  faiss::ProductQuantizer product(model.dim, model.subquantizers, 8);
  product.centroids = model.centroids;

  std::vector<float> transformed(vectors.size());
  if (model.has_rotation()) {
    transform.apply_noalloc(count, vectors.data(), transformed.data());
  } else {
    transformed = vectors;
  }
  std::vector<u8> expected(count * model.code_bytes());
  product.compute_codes(transformed.data(), expected.data(), count);

  size_t mismatches = 0;
  size_t printed = 0;
  std::vector<float> scratch(model.dim);
  std::vector<u8> actual(model.code_bytes());
  for (size_t index = 0; index < count; ++index) {
    gpu_search::pq::encode(
      model,
      std::span<const float>{vectors.data() + index * model.dim, model.dim},
      actual, scratch);
    for (u32 component = 0; component < model.code_bytes(); ++component) {
      const u8 expected_code = expected[index * model.code_bytes() + component];
      if (actual[component] == expected_code) continue;
      ++mismatches;
      if (printed < 16) {
        std::cerr << "mismatch vector=" << index
                  << " component=" << component
                  << " runtime=" << static_cast<u32>(actual[component])
                  << " faiss=" << static_cast<u32>(expected_code) << '\n';
        ++printed;
      }
    }
  }
  std::cerr << "PQ encoder mismatch components=" << mismatches << "/"
            << (count * model.code_bytes()) << '\n';
  return mismatches;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 3) {
    gpu_search::pq::Model model;
    std::string error;
    if (!gpu_search::pq::read_model(argv[1], model, &error)) {
      std::cerr << error << '\n';
      return 2;
    }
    u32 dim = 0;
    size_t count = 0;
    const auto values = read_vectors(argv[2], dim, count);
    if (dim != model.dim) {
      std::cerr << "model/vector dimension mismatch\n";
      return 2;
    }
    return audit(model, values, count) == 0 ? 0 : 1;
  }

  constexpr u32 kDim = 8;
  constexpr u32 kSubquantizers = 4;
  constexpr size_t kCount = 512;
  gpu_search::pq::Model model;
  model.dim = kDim;
  model.subquantizers = kSubquantizers;
  model.centroids.resize(
    static_cast<size_t>(model.subquantizers) *
    gpu_search::pq::kCentroidsPerSubquantizer * model.subvector_dim());
  for (u32 subquantizer = 0; subquantizer < model.subquantizers;
       ++subquantizer) {
    for (u32 centroid = 0; centroid < gpu_search::pq::kCentroidsPerSubquantizer;
         ++centroid) {
      const size_t base = (static_cast<size_t>(subquantizer) *
                             gpu_search::pq::kCentroidsPerSubquantizer +
                           centroid) * model.subvector_dim();
      model.centroids[base] =
        static_cast<float>(centroid) * 0.61803398875F +
        static_cast<float>(subquantizer) * 0.123F;
      model.centroids[base + 1] =
        static_cast<float>(centroid) * 0.38196601125F +
        static_cast<float>(subquantizer) * 0.456F;
    }
  }
  std::vector<float> values(kCount * kDim);
  for (size_t node = 0; node < kCount; ++node) {
    for (u32 dimension = 0; dimension < kDim; ++dimension) {
      values[node * kDim + dimension] =
        static_cast<float>((node * 17 + dimension * 29) % 997) / 31.0F;
    }
  }
  return audit(model, values, kCount) == 0 ? 0 : 1;
}
