#pragma once

#include <cstddef>
#include <string>

namespace plate_recognition_ocr_v6 {
namespace detail {

// PP-OCRv6 tiny uses one recognition vocabulary; small and medium share the
// full vocabulary. Only the 77 PlateRecognition-visible entries are embedded.
static const std::size_t kTinyClassCount = 6906U;
static const std::size_t kFullClassCount = 18710U;
static const std::size_t kEmbeddedPlateCharacterCount = 77U;

bool IsSupportedClassCount(std::size_t class_count);

std::size_t EmbeddedMappingSize(std::size_t class_count);

// Returns the index in plate_recognition::PlateRecognitionCharacters() for a
// mapped raw model class. Blank and non-plate OCR classes return false.
bool LookupPlateCharacterIndex(std::size_t class_count,
                               std::size_t model_class_index,
                               std::size_t &plate_character_index);

bool DecodeAndFilterCtc(const float *logits, std::size_t time_steps,
                        std::size_t class_count, std::string &plate_text);

} // namespace detail
} // namespace plate_recognition_ocr_v6
