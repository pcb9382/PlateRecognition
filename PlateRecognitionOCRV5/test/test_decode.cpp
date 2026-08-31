#include "OCRV5Text.h"

#include "PlateRecognitionCharset.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool Expect(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << std::endl;
    return false;
  }
  return true;
}

bool TestVariant(
    plate_recognition_ocr_v5_internal::EmbeddedModelVariant variant) {
  using namespace plate_recognition_ocr_v5_internal;

  bool ok = true;
  std::string error;
  ok = Expect(ValidateEmbeddedMapping(variant, error),
              "embedded PP-OCRv5 mapping validation failed") &&
       ok;
  ok = Expect(EmbeddedClassIndex(variant, 0U) == 1206U,
              "unexpected first embedded PP-OCRv5 class index") &&
       ok;
  ok = Expect(EmbeddedClassIndex(variant, 76U) == 1503U,
              "unexpected last embedded PP-OCRv5 class index") &&
       ok;
  ok = Expect(EmbeddedClassIndex(variant, 77U) == 0U,
              "out-of-range embedded mapping lookup must return blank") &&
       ok;

  const std::size_t province = EmbeddedClassIndex(variant, 0U);
  const std::size_t latin_a = EmbeddedClassIndex(variant, 51U);
  const std::size_t ignored_native_class = 1U;
  const std::size_t native_indices[] = {
      province, province, 0U, province, ignored_native_class,
      province, latin_a,  latin_a, ignored_native_class, latin_a};
  const std::size_t time_steps =
      sizeof(native_indices) / sizeof(native_indices[0]);
  std::vector<float> logits(time_steps * kOCRV5OutputClassCount, -10.0F);
  for (std::size_t time = 0U; time < time_steps; ++time) {
    logits[time * kOCRV5OutputClassCount + native_indices[time]] = 10.0F;
  }

  std::string decoded;
  ok = Expect(DecodeCtc(logits.data(), time_steps, kOCRV5OutputClassCount,
                        variant, decoded, error),
              "embedded PP-OCRv5 CTC decoder returned an error") &&
       ok;
  const std::vector<std::string> &characters =
      plate_recognition::PlateRecognitionCharacters();
  const std::string expected =
      characters[0] + characters[0] + characters[0] + "AA";
  ok = Expect(decoded == expected,
              "native-space CTC collapse or sparse mapping failed") &&
       ok;

  ok = Expect(!DecodeCtc(logits.data(), time_steps,
                         kOCRV5OutputClassCount - 1U, variant, decoded, error),
              "PP-OCRv5 decoder must reject a mismatched class count") &&
       ok;
  return ok;
}

} // namespace

int main() {
  using namespace plate_recognition_ocr_v5_internal;

  bool ok = true;
  ok = Expect(AllowedPlateCharacterCount() == 77U,
              "PlateRecognition whitelist must contain 77 visible characters") &&
       ok;

  const std::vector<std::string> &characters =
      plate_recognition::PlateRecognitionCharacters();
  const std::string noisy = characters[0] + "I" + characters[1] +
                            "O01-" + characters[75] + characters[76];
  const std::string filtered = characters[0] + characters[1] + "01" +
                               characters[75] + characters[76];
  ok = Expect(FilterPlateCharacters(noisy) == filtered,
              "UTF-8 PlateRecognition whitelist filtering failed") &&
       ok;
  ok = Expect(ComputeInputWidth(48, 168, 48, 20, 1280) == 180,
              "dynamic width must round up to a multiple of 20") &&
       ok;
  ok = Expect(ComputeInputWidth(100, 1, 48, 20, 1280) == 20,
              "dynamic width minimum clamp failed") &&
       ok;
  ok = Expect(ComputeInputWidth(1, 10000, 48, 20, 1280) == 1280,
              "dynamic width maximum clamp failed") &&
       ok;

  // Mobile and server intentionally exercise separate embedded tables. Their
  // current values match because both upstream models share class ordering.
  ok = TestVariant(EmbeddedModelVariant::kMobile) && ok;
  ok = TestVariant(EmbeddedModelVariant::kServer) && ok;
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
