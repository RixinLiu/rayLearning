#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "ray/util/compat.h"
#include "ray/util/macros.h"

namespace plasma {

class ClientMmapTableEntry {
 public:
  ClientMmapTableEntry(MEMFD_TYPE fd, int64_t map_size);

  ~ClientMmapTableEntry();

  uint8_t *pointer() const { return reinterpret_cast<uint8_t *>(pointer_); }

  MEMFD_TYPE fd() const { return fd_; }

 private:
  /// The associated file descriptor on the client.
  MEMFD_TYPE fd_;
  /// The result of mmap for this file descriptor.
  void *pointer_;
  /// The length of the memory-mapped file.
  size_t length_;

  void MaybeMadviseDontdump();

  RAY_DISALLOW_COPY_AND_ASSIGN(ClientMmapTableEntry);
};

}  // namespace plasma
