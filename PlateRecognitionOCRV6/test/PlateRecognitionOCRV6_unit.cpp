#include "OCRV6Text.h"

#include "PlateRecognitionCharset.h"

#include <cstddef>
#include <string>
#include <vector>

namespace detail = plate_recognition_ocr_v6::detail;

namespace {

void SetBest(std::vector<float> &logits, std::size_t time, std::size_t classes,
             std::size_t best) {
  logits[time * classes + best] = 1.0f;
}

bool VerifyTier(std::size_t class_count) {
  const std::vector<std::string> &characters =
      plate_recognition::PlateRecognitionCharacters();
  if (!detail::IsSupportedClassCount(class_count) ||
      detail::EmbeddedMappingSize(class_count) != characters.size() ||
      characters.size() != detail::kEmbeddedPlateCharacterCount) {
    return false;
  }

  std::vector<std::size_t> model_classes(characters.size(), class_count);
  std::size_t mapped_count = 0;
  for (std::size_t model_class = 1; model_class < class_count; ++model_class) {
    std::size_t plate_index = characters.size();
    if (!detail::LookupPlateCharacterIndex(class_count, model_class,
                                           plate_index)) {
      continue;
    }
    if (plate_index >= characters.size() ||
        model_classes[plate_index] != class_count) {
      return false;
    }
    model_classes[plate_index] = model_class;
    ++mapped_count;
  }
  if (mapped_count != characters.size()) {
    return false;
  }

  std::string expected;
  std::vector<float> all_logits(characters.size() * class_count, -1.0f);
  for (std::size_t index = 0; index < characters.size(); ++index) {
    if (model_classes[index] == class_count) {
      return false;
    }
    SetBest(all_logits, index, class_count, model_classes[index]);
    expected += characters[index];
  }
  std::string decoded;
  if (!detail::DecodeAndFilterCtc(all_logits.data(), characters.size(),
                                  class_count, decoded) ||
      decoded != expected) {
    return false;
  }

  std::size_t unmapped_class = 1;
  std::size_t ignored_plate_index = 0;
  while (unmapped_class < class_count &&
         detail::LookupPlateCharacterIndex(class_count, unmapped_class,
                                           ignored_plate_index)) {
    ++unmapped_class;
  }
  if (unmapped_class == class_count ||
      detail::LookupPlateCharacterIndex(class_count, 0, ignored_plate_index) ||
      detail::LookupPlateCharacterIndex(class_count, class_count,
                                        ignored_plate_index)) {
    return false;
  }

  // CTC deduplication is based on raw model indices. An unmapped class and a
  // blank both interrupt repetition even though neither is emitted.
  const std::size_t a_plate_index = 51U;
  const std::size_t a_model_class = model_classes[a_plate_index];
  std::vector<float> ctc_logits(6U * class_count, -1.0f);
  SetBest(ctc_logits, 0, class_count, a_model_class);
  SetBest(ctc_logits, 1, class_count, a_model_class);
  SetBest(ctc_logits, 2, class_count, unmapped_class);
  SetBest(ctc_logits, 3, class_count, a_model_class);
  SetBest(ctc_logits, 4, class_count, 0);
  SetBest(ctc_logits, 5, class_count, a_model_class);
  if (!detail::DecodeAndFilterCtc(ctc_logits.data(), 6U, class_count,
                                  decoded) ||
      decoded != characters[a_plate_index] + characters[a_plate_index] +
                     characters[a_plate_index]) {
    return false;
  }
  return true;
}

} // namespace

int main() {
  if (!VerifyTier(detail::kTinyClassCount)) {
    return 1;
  }
  if (!VerifyTier(detail::kFullClassCount)) {
    return 2;
  }
  if (detail::IsSupportedClassCount(100U) ||
      detail::EmbeddedMappingSize(100U) != 0U) {
    return 3;
  }
  float unsupported_logits[100] = {};
  std::string decoded = "stale";
  if (detail::DecodeAndFilterCtc(unsupported_logits, 1U, 100U, decoded) ||
      !decoded.empty()) {
    return 4;
  }
  return 0;
}
