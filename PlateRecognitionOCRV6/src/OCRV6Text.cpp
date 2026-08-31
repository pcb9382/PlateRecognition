#include "OCRV6Text.h"

#include "PlateRecognitionCharset.h"

#include <cstdint>
#include <limits>
#include <vector>

namespace plate_recognition_ocr_v6 {
namespace detail {
namespace {

// Each array is ordered exactly like PlateRecognitionCharacters(). Values are
// raw PP-OCRv6 output class indices (class 0 is the CTC blank).
const std::uint16_t kTinyClassIndexByPlateCharacter[] = {
    745,  3342, 3396, 3506, 1010, 2876, 5177, 5875, 1294, 6785, 5014,
    3424, 4107, 6303, 5732, 6676, 5647, 6025, 3525, 4505, 3052, 3923,
    2133, 5699, 725,  5243, 6350, 3971, 6434, 1930, 2819, 1919, 5516,
    3515, 3617, 2617, 848,  6494, 3269, 4954, 1219, 33,   34,   35,
    36,   37,   38,   39,   40,   41,   42,   43,   44,   45,   46,
    47,   48,   49,   50,   52,   53,   54,   55,   56,   58,   59,
    60,   61,   62,   63,   64,   65,   66,   67,   68,   6357, 1393};

const std::uint16_t kFullClassIndexByPlateCharacter[] = {
    2208,  7820,  7911,  8106,  2753,  6668,  12308, 14562, 3223,  17483,
    11961, 7953,  9652,  15746, 14095, 16993, 13892, 14823, 8151,  10722,
    7043,  9208,  5031,  14049, 2187,  12566, 15822, 9360,  16003, 4582,
    6550,  4564,  13672, 8121,  8413,  6047,  2370,  16238, 7702,  11833,
    3113,  33,    34,    35,    36,    37,    38,    39,    40,    41,
    42,    43,    44,    45,    46,    47,    48,    49,    50,    52,
    53,    54,    55,    56,    58,    59,    60,    61,    62,    63,
    64,    65,    66,    67,    68,    15837, 3376};

static_assert(sizeof(kTinyClassIndexByPlateCharacter) /
                      sizeof(kTinyClassIndexByPlateCharacter[0]) ==
                  kEmbeddedPlateCharacterCount,
              "PP-OCRv6 tiny mapping must contain 77 entries");
static_assert(sizeof(kFullClassIndexByPlateCharacter) /
                      sizeof(kFullClassIndexByPlateCharacter[0]) ==
                  kEmbeddedPlateCharacterCount,
              "PP-OCRv6 full mapping must contain 77 entries");

const std::uint16_t *MappingForClassCount(std::size_t class_count) {
  if (class_count == kTinyClassCount) {
    return kTinyClassIndexByPlateCharacter;
  }
  if (class_count == kFullClassCount) {
    return kFullClassIndexByPlateCharacter;
  }
  return NULL;
}

} // namespace

bool IsSupportedClassCount(std::size_t class_count) {
  return MappingForClassCount(class_count) != NULL;
}

std::size_t EmbeddedMappingSize(std::size_t class_count) {
  return IsSupportedClassCount(class_count) ? kEmbeddedPlateCharacterCount : 0U;
}

bool LookupPlateCharacterIndex(std::size_t class_count,
                               std::size_t model_class_index,
                               std::size_t &plate_character_index) {
  const std::uint16_t *mapping = MappingForClassCount(class_count);
  if (mapping == NULL || model_class_index >= class_count) {
    return false;
  }
  for (std::size_t index = 0; index < kEmbeddedPlateCharacterCount; ++index) {
    if (mapping[index] == model_class_index) {
      plate_character_index = index;
      return true;
    }
  }
  return false;
}

bool DecodeAndFilterCtc(const float *logits, std::size_t time_steps,
                        std::size_t class_count, std::string &plate_text) {
  plate_text.clear();
  const std::vector<std::string> &plate_characters =
      plate_recognition::PlateRecognitionCharacters();
  if (logits == NULL || time_steps == 0 ||
      !IsSupportedClassCount(class_count) ||
      plate_characters.size() != kEmbeddedPlateCharacterCount ||
      time_steps > std::numeric_limits<std::size_t>::max() / class_count) {
    return false;
  }

  std::size_t previous = class_count;
  for (std::size_t time = 0; time < time_steps; ++time) {
    const float *row = logits + time * class_count;
    std::size_t best = 0;
    for (std::size_t index = 1; index < class_count; ++index) {
      if (row[index] > row[best]) {
        best = index;
      }
    }

    if (best == 0) {
      previous = best;
      continue;
    }
    if (best == previous) {
      continue;
    }

    // Update the raw model index before the sparse lookup. Therefore an
    // unmapped OCR class still separates equal mapped classes for CTC.
    previous = best;
    std::size_t plate_character_index = 0;
    if (LookupPlateCharacterIndex(class_count, best, plate_character_index)) {
      plate_text += plate_characters[plate_character_index];
    }
  }
  return true;
}

} // namespace detail
} // namespace plate_recognition_ocr_v6
