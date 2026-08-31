#pragma once

#include <cstddef>
#include <string>

namespace plate_recognition_ocr_v5_internal {

enum class EmbeddedModelVariant {
  kMobile = 0,
  kServer = 1,
};

const std::size_t kOCRV5OutputClassCount = 18385U;

int ComputeInputWidth(int image_height, int image_width, int target_height,
                      int min_width, int max_width);

std::size_t AllowedPlateCharacterCount();

std::string FilterPlateCharacters(const std::string &text);

bool ValidateEmbeddedMapping(EmbeddedModelVariant variant, std::string &error);

std::size_t EmbeddedClassIndex(EmbeddedModelVariant variant,
                               std::size_t plate_character_index);

bool DecodeCtc(const float *logits, std::size_t time_steps,
               std::size_t class_count, EmbeddedModelVariant variant,
               std::string &plate_text, std::string &error);

} // namespace plate_recognition_ocr_v5_internal
