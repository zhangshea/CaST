// Minimal reader for NumPy .npy files (format version 1.x/2.x, little-endian, C order), used by the command-line
// tools of Sec. VI-A (graph construction and index updates); the Python module receives arrays directly.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace cast {

struct NpyArray {
  std::string dtype;            // "<f4", "<i4", "<i8" or "<f8"
  std::vector<size_t> shape;
  std::vector<char> bytes;
  size_t rows() const { return shape.empty() ? 0 : shape[0]; }
  size_t cols() const { return shape.size() < 2 ? 1 : shape[1]; }
  template <class T> const T* data() const { return reinterpret_cast<const T*>(bytes.data()); }
};

inline NpyArray read_npy(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  char magic[6];
  if (std::fread(magic, 1, 6, f) != 6 || std::memcmp(magic, "\x93NUMPY", 6) != 0)
    throw std::runtime_error("not a .npy file: " + path);
  unsigned char ver[2];
  if (std::fread(ver, 1, 2, f) != 2) throw std::runtime_error("bad header");
  uint32_t hlen = 0;
  if (ver[0] == 1) {
    uint16_t h16; if (std::fread(&h16, 2, 1, f) != 1) throw std::runtime_error("bad header"); hlen = h16;
  } else {
    if (std::fread(&hlen, 4, 1, f) != 1) throw std::runtime_error("bad header");
  }
  std::string hdr(hlen, '\0');
  if (std::fread(&hdr[0], 1, hlen, f) != hlen) throw std::runtime_error("bad header");
  NpyArray a;
  auto p = hdr.find("'descr':");
  auto q1 = hdr.find('\'', p + 8), q2 = hdr.find('\'', q1 + 1);
  a.dtype = hdr.substr(q1 + 1, q2 - q1 - 1);
  if (hdr.find("'fortran_order': True") != std::string::npos) throw std::runtime_error("fortran order unsupported");
  auto s1 = hdr.find('(', hdr.find("'shape':")), s2 = hdr.find(')', s1);
  std::string sh = hdr.substr(s1 + 1, s2 - s1 - 1);
  size_t pos = 0;
  while (pos < sh.size()) {
    while (pos < sh.size() && (sh[pos] == ' ' || sh[pos] == ',')) pos++;
    if (pos >= sh.size()) break;
    size_t e = pos; while (e < sh.size() && sh[e] >= '0' && sh[e] <= '9') e++;
    a.shape.push_back(std::stoull(sh.substr(pos, e - pos)));
    pos = e;
  }
  size_t item = (a.dtype == "<f4" || a.dtype == "<i4") ? 4 : (a.dtype == "<i8" || a.dtype == "<f8") ? 8 : 0;
  if (!item) throw std::runtime_error("unsupported dtype " + a.dtype);
  size_t cnt = 1; for (auto s : a.shape) cnt *= s;
  a.bytes.resize(cnt * item);
  if (std::fread(a.bytes.data(), 1, a.bytes.size(), f) != a.bytes.size()) throw std::runtime_error("short read " + path);
  std::fclose(f);
  return a;
}

}  // namespace cast
