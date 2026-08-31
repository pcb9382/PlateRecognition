#include "OCRV5Text.h"

#include "PlateRecognitionCharset.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <vector>

namespace plate_recognition_ocr_v5_internal {
namespace {

// PP-OCRv5 mobile/light recognition mapping. The indices are native model
// output class indices (0 is CTC blank), ordered exactly like
// PlateRecognitionCharacters().
const std::size_t kMobileClassIndices[] = {
    1206, 870,  1683, 2763, 3340, 1828, 2859, 332,  357,  2655, 678,
    2066, 2699, 1637, 3495, 2717, 3313, 2283, 2752, 2700, 1835, 2504,
    44,   1495, 83,   3392, 1321, 206,  948,  313,  2994, 1274, 3453,
    2749, 3295, 1365, 1140, 2354, 328,  1981, 500,  16163, 16164, 16165,
    16166, 16167, 16168, 16169, 16170, 16171, 16172, 16179, 16180, 16181,
    16182, 16183, 16184, 16185, 16186, 16188, 16189, 16190, 16191, 16192,
    16194, 16195, 16196, 16197, 16198, 16199, 16200, 16201, 16202, 16203,
    16204, 1733, 1503};

// PP-OCRv5 server recognition mapping. It is deliberately stored separately
// from the mobile table so either bundled model can evolve without silently
// changing the other model's contract. The current upstream models share the
// same 18,385-class dictionary ordering.
const std::size_t kServerClassIndices[] = {
    1206, 870,  1683, 2763, 3340, 1828, 2859, 332,  357,  2655, 678,
    2066, 2699, 1637, 3495, 2717, 3313, 2283, 2752, 2700, 1835, 2504,
    44,   1495, 83,   3392, 1321, 206,  948,  313,  2994, 1274, 3453,
    2749, 3295, 1365, 1140, 2354, 328,  1981, 500,  16163, 16164, 16165,
    16166, 16167, 16168, 16169, 16170, 16171, 16172, 16179, 16180, 16181,
    16182, 16183, 16184, 16185, 16186, 16188, 16189, 16190, 16191, 16192,
    16194, 16195, 16196, 16197, 16198, 16199, 16200, 16201, 16202, 16203,
    16204, 1733, 1503};

const std::size_t kEmbeddedCharacterCount =
    sizeof(kMobileClassIndices) / sizeof(kMobileClassIndices[0]);

static_assert(kEmbeddedCharacterCount == 77U,
              "PP-OCRv5 embedded mapping must contain 77 characters");
static_assert(sizeof(kServerClassIndices) / sizeof(kServerClassIndices[0]) ==
                  kEmbeddedCharacterCount,
              "PP-OCRv5 mobile/server mappings must have equal size");

const std::size_t *ClassIndices(EmbeddedModelVariant variant) {
  return variant == EmbeddedModelVariant::kMobile ? kMobileClassIndices
                                                   : kServerClassIndices;
}

bool LookupPlateCharacter(EmbeddedModelVariant variant,
                          std::size_t model_class_index,
                          std::size_t &plate_character_index) {
  const std::size_t *indices = ClassIndices(variant);
  for (std::size_t index = 0; index < kEmbeddedCharacterCount; ++index) {
    if (indices[index] == model_class_index) {
      plate_character_index = index;
      return true;
    }
  }
  return false;
}

} // namespace

int ComputeInputWidth(int image_height, int image_width, int target_height,
                      int min_width, int max_width) {
  if (image_height <= 0 || image_width <= 0 || target_height <= 0 ||
      min_width <= 0 || max_width < min_width) {
    return 0;
  }
  const double scaled = static_cast<double>(image_width) *
                        static_cast<double>(target_height) /
                        static_cast<double>(image_height);
  if (!std::isfinite(scaled)) {
    return 0;
  }
  const double bounded = std::min(
      static_cast<double>(std::numeric_limits<int>::max() - 19),
      std::max(1.0, std::floor(scaled + 0.5)));
  const int rounded = static_cast<int>(bounded);
  const int multiple_of_twenty = ((rounded + 19) / 20) * 20;
  return std::max(min_width, std::min(max_width, multiple_of_twenty));
}

std::size_t AllowedPlateCharacterCount() {
  return plate_recognition::PlateRecognitionCharacters().size();
}

std::string FilterPlateCharacters(const std::string &text) {
  return plate_recognition::FilterPlateRecognitionText(text);
}

bool ValidateEmbeddedMapping(EmbeddedModelVariant variant,
                             std::string &error) {
  error.clear();
  if (AllowedPlateCharacterCount() != kEmbeddedCharacterCount) {
    error = "embedded PP-OCRv5 mapping does not match PlateRecognition";
    return false;
  }

  const std::size_t *indices = ClassIndices(variant);
  std::set<std::size_t> unique_indices;
  for (std::size_t index = 0; index < kEmbeddedCharacterCount; ++index) {
    if (indices[index] == 0U || indices[index] >= kOCRV5OutputClassCount ||
        !unique_indices.insert(indices[index]).second) {
      error = "embedded PP-OCRv5 mapping contains an invalid class index";
      return false;
    }
  }
  return true;
}

std::size_t EmbeddedClassIndex(EmbeddedModelVariant variant,
                               std::size_t plate_character_index) {
  if (plate_character_index >= kEmbeddedCharacterCount) {
    return 0U;
  }
  return ClassIndices(variant)[plate_character_index];
}

bool DecodeCtc(const float *logits, std::size_t time_steps,
               std::size_t class_count, EmbeddedModelVariant variant,
               std::string &plate_text, std::string &error) {
  plate_text.clear();
  error.clear();
  if (logits == NULL || time_steps == 0U ||
      class_count != kOCRV5OutputClassCount) {
    error = "PP-OCRv5 output must contain exactly 18,385 classes";
    return false;
  }
  if (time_steps > std::numeric_limits<std::size_t>::max() / class_count) {
    error = "CTC tensor size overflows size_t";
    return false;
  }
  if (!ValidateEmbeddedMapping(variant, error)) {
    return false;
  }

  const std::vector<std::string> &characters =
      plate_recognition::PlateRecognitionCharacters();
  std::size_t previous = 0U;
  for (std::size_t time = 0; time < time_steps; ++time) {
    const float *row = logits + time * class_count;
    std::size_t best = 0U;
    float best_value = row[0];
    for (std::size_t model_class = 1U; model_class < class_count;
         ++model_class) {
      if (row[model_class] > best_value) {
        best = model_class;
        best_value = row[model_class];
      }
    }

    if (best != 0U && best != previous) {
      std::size_t plate_character_index = 0U;
      if (LookupPlateCharacter(variant, best, plate_character_index)) {
        plate_text += characters[plate_character_index];
      }
    }
    // CTC repetition is tracked in the native 18,385-class space. An ignored
    // OCR class still separates two otherwise equal plate characters.
    previous = best;
  }
  return true;
}

} // namespace plate_recognition_ocr_v5_internal
