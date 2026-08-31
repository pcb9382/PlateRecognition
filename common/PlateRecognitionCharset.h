#pragma once

#include <cstddef>
#include <string>
#include <unordered_set>
#include <vector>

namespace plate_recognition {

inline const std::vector<std::string> &PlateRecognitionCharacters() {
  static const std::vector<std::string> characters = {
      "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏",
      "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼",
      "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "学", "警",
      "港", "澳", "挂", "使", "领", "民", "航", "危", "0",  "1",  "2",
      "3",  "4",  "5",  "6",  "7",  "8",  "9",  "A",  "B",  "C",  "D",
      "E",  "F",  "G",  "H",  "J",  "K",  "L",  "M",  "N",  "P",  "Q",
      "R",  "S",  "T",  "U",  "V",  "W",  "X",  "Y",  "Z",  "险", "品"};
  return characters;
}

inline const std::unordered_set<std::string> &PlateRecognitionCharacterSet() {
  static const std::unordered_set<std::string> characters(
      PlateRecognitionCharacters().begin(), PlateRecognitionCharacters().end());
  return characters;
}

inline bool IsPlateRecognitionCharacter(const std::string &character) {
  return PlateRecognitionCharacterSet().count(character) != 0U;
}

inline std::size_t Utf8CodePointLength(unsigned char lead) {
  if ((lead & 0x80U) == 0U) {
    return 1U;
  }
  if ((lead & 0xE0U) == 0xC0U) {
    return 2U;
  }
  if ((lead & 0xF0U) == 0xE0U) {
    return 3U;
  }
  if ((lead & 0xF8U) == 0xF0U) {
    return 4U;
  }
  return 1U;
}

inline std::string FilterPlateRecognitionText(const std::string &text) {
  std::string filtered;
  for (std::size_t offset = 0U; offset < text.size();) {
    std::size_t length =
        Utf8CodePointLength(static_cast<unsigned char>(text[offset]));
    if (offset + length > text.size()) {
      length = 1U;
    }

    const std::string character = text.substr(offset, length);
    if (IsPlateRecognitionCharacter(character)) {
      filtered += character;
    }
    offset += length;
  }
  return filtered;
}

} // namespace plate_recognition
