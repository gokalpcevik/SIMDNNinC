#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#define STBI_FAILURE_USERMSG
#define STB_IMAGE_IMPLEMENTATION
#include "vendor/stb_image.h"

static constexpr size_t INPUT_LEN = 28ULL * 28ULL; // 28x28 pixel image
static constexpr size_t L1_NEURON_COUNT = 512ULL;
static constexpr size_t L2_NEURON_COUNT = 512ULL;
static constexpr size_t L3_NEURON_COUNT = 10ULL; // Output layer
static constexpr size_t L1_PARAM_COUNT = L1_NEURON_COUNT * INPUT_LEN + L1_NEURON_COUNT;
static constexpr size_t L2_PARAM_COUNT = L2_NEURON_COUNT * L1_NEURON_COUNT + L2_NEURON_COUNT;
static constexpr size_t L3_PARAM_COUNT = L3_NEURON_COUNT * L2_NEURON_COUNT + L3_NEURON_COUNT;
static constexpr size_t LAYER_COUNT = 3;

char const* ITEM_LABELS[] = {"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                             "Sandal",      "Shirt",   "Sneaker",  "Bag",   "Ankle boot"};

struct time_logger {
  time_logger() { start = std::chrono::high_resolution_clock::now(); }
  void stop() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    printf("Took %ld us\n", duration);
  }
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

constexpr size_t get_parameter_count() {
  size_t total_param_count = 0;
  total_param_count += L1_PARAM_COUNT;
  total_param_count += L2_PARAM_COUNT;
  total_param_count += L3_PARAM_COUNT;
  return total_param_count;
}

int classify_item_simd(float* params, float* data);
int classify_item_no_simd(float* params, float* data);
void convert_8bpc_to_32bpc_simd(unsigned char* u8, float* f32, size_t len);
float fast_dot_product(float const* a, float const* b, size_t len);
void soft_max(float* in, float* out, size_t len);

// NOTE: I know I am leaking memory here(and other places), but program only runs once and the allocated memory will be
// reclaimed by the OS at program exit.
int main(int argc, char** argv) {
  if (argc < 2) {
    printf("Image file not provided.\n");
    return EXIT_FAILURE;
  }
  int x, y, n; // width, height, bpc
  unsigned char* data8bpc = stbi_load(argv[1], &x, &y, &n, 1);
  if (data8bpc == NULL) {
    printf("%s\n", stbi_failure_reason());
    return EXIT_FAILURE;
  }
  if (n != 1) {
    printf("Expected grayscale image, received something else.\n");
    return EXIT_FAILURE;
  }
  printf("x:%d,y:%d,n:%d\n", x, y, n);
  size_t res = (size_t)x * (size_t)y; // texture resolution (28x28)
  float* data32bpc = (float*)aligned_alloc(32, res * sizeof(float));
  convert_8bpc_to_32bpc_simd(data8bpc, data32bpc, res); // Neural network expects grayscale to be in range [0.0, 1.0]
  int fd = open("./parameters_raw", O_RDONLY);
  if (fd == -1) {
    printf("Error while opening the parameters file: %d\n", errno);
    return EXIT_FAILURE;
  }
  size_t param_count = get_parameter_count();
  size_t buf_size = param_count * sizeof(float);
  float* parameters = (float*)aligned_alloc(32, buf_size);
  if (parameters == NULL) {
    printf("OOM while allocating a buffer for model parameters.\n");
    return EXIT_FAILURE;
  }
  ssize_t rd_bytes = read(fd, parameters, buf_size);
  if (rd_bytes == -1) {
    printf("Error while reading parameters: %d\n", errno);
    return EXIT_FAILURE;
  }
  if (rd_bytes != buf_size) {
    printf("Model parameter file has incorrect size. Read bytes:%zd, expected:%zu. Exiting.\n", rd_bytes, buf_size);
    return EXIT_FAILURE;
  }
  int option = 0;
  if (argc > 2) {
    option = atoi(argv[2]);
  }
  if (option == 0) {
    printf("Classifying item...[NO SIMD]\n");
  	time_logger time;
    int fashion_class = classify_item_no_simd(parameters, data32bpc);
  	time.stop();
    printf("Item classified: %s (%d)\n", ITEM_LABELS[fashion_class], fashion_class);
  } else if (option == 1) {
    printf("Classifying item...[SIMD]\n");
  	time_logger time;
    int fashion_class = classify_item_simd(parameters, data32bpc);
  	time.stop();
    printf("Item classified: %s (%d)\n", ITEM_LABELS[fashion_class], fashion_class);
  }
  return EXIT_SUCCESS;
}

int classify_item_simd(float* params, float* data) {
  // NOTE: I know I am leaking memory here, but program only runs once and the allocated memory will be reclaimed by the
  // OS.
  // Layer 1
  float* outL1 = (float*)aligned_alloc(32, L1_NEURON_COUNT * sizeof(float));
  size_t bias_offsetL1 = L1_NEURON_COUNT * INPUT_LEN;
  // Layer 2
  float* outL2 = (float*)aligned_alloc(32, L2_NEURON_COUNT * sizeof(float));
  size_t weight_offsetL2 = L1_PARAM_COUNT;
  size_t bias_offsetL2 = weight_offsetL2 + L2_NEURON_COUNT * L1_NEURON_COUNT;
  // Layer 3
  float* outL3 = (float*)aligned_alloc(32, L3_NEURON_COUNT * sizeof(float));
  size_t weight_offsetL3 = L1_PARAM_COUNT + L2_PARAM_COUNT;
  size_t bias_offsetL3 = weight_offsetL3 + L3_NEURON_COUNT * L2_NEURON_COUNT;
  for (size_t row_idx = 0; row_idx < L1_NEURON_COUNT; ++row_idx) {
    float* a = &params[row_idx * INPUT_LEN];
    float* b = data;
    float z = fast_dot_product(a, b, INPUT_LEN) + params[bias_offsetL1 + row_idx];
    outL1[row_idx] = fmax(0.0f, z);
  }
  for (size_t row_idx = 0; row_idx < L2_NEURON_COUNT; ++row_idx) {
    float* a = &params[weight_offsetL2 + row_idx * L1_NEURON_COUNT];
    float* b = outL1;
    float z = fast_dot_product(a, b, L1_NEURON_COUNT) + params[bias_offsetL2 + row_idx];
    outL2[row_idx] = fmax(0.0f, z);
  }
  for (size_t row_idx = 0; row_idx < L3_NEURON_COUNT; ++row_idx) {
    float* a = &params[weight_offsetL3 + row_idx * L2_NEURON_COUNT];
    float* b = outL2;
    outL3[row_idx] = fast_dot_product(a, b, L2_NEURON_COUNT) + params[bias_offsetL3 + row_idx];
  }
  float soft_out[L3_NEURON_COUNT];
  soft_max(outL3, soft_out, L3_NEURON_COUNT);
  float max = 0.0f;
  int item_class = -1;
  for (size_t i = 0; i < L3_NEURON_COUNT; ++i) {
    if (soft_out[i] > max) {
      max = outL3[i];
      item_class = i;
    }
  }
  return item_class;
}

int classify_item_no_simd(float* params, float* data) {
  // Layer 1
  float outL1[L1_NEURON_COUNT];
  size_t bias_offsetL1 = L1_NEURON_COUNT * INPUT_LEN; // Layer 1 bias offset
                                                      // Layer 2
  float outL2[L2_NEURON_COUNT];
  size_t weight_offsetL2 = L1_PARAM_COUNT; // Layer 2 - weight offset from the params buffer
  size_t bias_offsetL2 = weight_offsetL2 + L2_NEURON_COUNT * L1_NEURON_COUNT;
  // Layer 3
  float outL3[L3_NEURON_COUNT];
  size_t weight_offsetL3 = L1_PARAM_COUNT + L2_PARAM_COUNT;
  size_t bias_offsetL3 = weight_offsetL3 + L3_NEURON_COUNT * L2_NEURON_COUNT;

  // NOTE: L1
  for (size_t row_idx = 0; row_idx < L1_NEURON_COUNT; ++row_idx) {
    float accumulator = 0.0f;
    for (size_t input_idx = 0; input_idx < INPUT_LEN; ++input_idx) {
      accumulator += params[row_idx * INPUT_LEN + input_idx] * data[input_idx];
    }
    outL1[row_idx] = accumulator;
  }
  for (size_t row_idx = 0; row_idx < L1_NEURON_COUNT; ++row_idx) {
    outL1[row_idx] += params[bias_offsetL1 + row_idx]; // Add the bias
    outL1[row_idx] = fmax(0.0f, outL1[row_idx]);       // Apply ReLU
  }
  // Layer 2
  for (size_t row_idx = 0; row_idx < L2_NEURON_COUNT; ++row_idx) {
    float accumulator = 0.0f;
    for (size_t outL2_idx = 0; outL2_idx < L1_NEURON_COUNT; ++outL2_idx) {
      accumulator += params[weight_offsetL2 + row_idx * L1_NEURON_COUNT + outL2_idx] * outL1[outL2_idx];
    }
    outL2[row_idx] = accumulator;
  }
  for (size_t row_idx = 0; row_idx < L2_NEURON_COUNT; ++row_idx) {
    outL2[row_idx] += params[bias_offsetL2 + row_idx]; // Add the bias
    outL2[row_idx] = fmax(0.0f, outL2[row_idx]);       // Apply ReLU
  }
  for (size_t row_idx = 0; row_idx < L3_NEURON_COUNT; ++row_idx) {
    float acc = 0.0f;
    for (size_t outL3_idx = 0; outL3_idx < L2_NEURON_COUNT; ++outL3_idx) {
      acc += params[weight_offsetL3 + row_idx * L2_NEURON_COUNT + outL3_idx] * outL2[outL3_idx];
    }
    outL3[row_idx] = acc;
  }
  for (size_t row_idx = 0; row_idx < L3_NEURON_COUNT; ++row_idx) {
    outL3[row_idx] += params[bias_offsetL3 + row_idx]; // Add the bias to get the logit
  }
  float soft_out[L3_NEURON_COUNT];
  soft_max(outL3, soft_out, L3_NEURON_COUNT);
  float max = 0.0f;
  int item_class = -1;
  for (size_t i = 0; i < L3_NEURON_COUNT; ++i) {
    if (soft_out[i] > max) {
      max = outL3[i];
      item_class = i;
    }
  }
  return item_class;
}
// Probably there are faster implementations, but I'm too lazy to do it. Besides, it probably won't make much difference
// in performance.
void soft_max(float* in, float* out, size_t len) {
  float max = in[0];
  // Find max
  for (size_t i = 0; i < len; ++i) {
    if (in[i] > max) {
      max = in[i];
    }
  }
  // To avoid numerical instability, we'll use the identity softmax(x)=softmax(x+c)
  // Numerical Computation Chapter in https://www.deeplearningbook.org/
  for (size_t i = 0; i < len; ++i) {
    out[i] = in[i] - max;
  }
  // Find sum (softmax denom.)
  float sum = 0.0f;
  for (size_t i = 0; i < len; ++i) {
    sum += out[i];
  }
  // Softmax
  for (size_t i = 0; i < len; ++i) {
    out[i] = out[i] / sum;
  }
}

// WARNING: Make sure pointers a and b are aligned to 32-byte boundary. (Allocate those buffers via aligned_alloc)
// WARNING: Length needs to be a multiple of 16.
float fast_dot_product(float const* a, float const* b, size_t len) {
  __m256 dot0 = _mm256_set1_ps(0.0f);
  __m256 dot1 = _mm256_set1_ps(0.0f);
  for (size_t i = 0; i < len - 15; i += 16) {
    __m256 v0 = _mm256_load_ps(&a[i + 0]);
    __m256 v1 = _mm256_load_ps(&b[i + 0]);
    __m256 v2 = _mm256_load_ps(&a[i + 8]);
    __m256 v3 = _mm256_load_ps(&b[i + 8]);
    dot0 = _mm256_fmadd_ps(v0, v1, dot0);
    dot1 = _mm256_fmadd_ps(v2, v3, dot1);
  }
  __m256 dot01 = _mm256_add_ps(dot0, dot1);
  __m128 vlow = _mm256_castps256_ps128(dot01);
  __m128 vhigh = _mm256_extractf128_ps(dot01, 1);
  vlow = _mm_add_ps(vlow, vhigh);
  __m128 shuf = _mm_movehdup_ps(vlow);
  __m128 sums = _mm_add_ps(vlow, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}

void convert_8bpc_to_32bpc_simd(unsigned char* u8, float* f32, size_t len) {
  size_t i;
  __m256 const255f = _mm256_set1_ps(255.0f);
  for (i = 0; i < len - 7; i += 8) {
    __m128i u8ymm = _mm_loadu_si128((__m128i const*)&u8[i]); // Load 8 unsigned-integers
    __m256i i32ymm = _mm256_cvtepu8_epi32(u8ymm);            // Convert 8 unsigned-integers to 8 32-bit integers
    __m256 f32ymm = _mm256_cvtepi32_ps(i32ymm);              // Convert 8 ints to 8 single-precision floats
    f32ymm = _mm256_div_ps(f32ymm, const255f);               // Divide by 255.0f to scale to [0.0,1.0]
    _mm256_store_ps(&f32[i], f32ymm);                        // Store
  }
  // Convert the remaining elements
  for (; i < len; i++) {
    f32[i] = (float)u8[i] / 255.0f;
  }
}
