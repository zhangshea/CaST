#!/usr/bin/env bash
# Build the tools, the test of the answer queue and the Python module of the search engine (paper Sections III-A and VI-A).
#   ./build.sh [HNSWLIB_DIR]          (or HNSWLIB_DIR=/path/to/hnswlib ./build.sh)
# HNSWLIB_DIR is the directory that holds hnswlib.h (upstream hnswlib v0.8.0, see README.md), or the root of an
# hnswlib checkout. Requires a C++17 compiler and, for the module, pybind11 importable by $PYTHON.
# Optional variables: PYTHON (default python3), CXX (default g++), FLAGS, BUILD_DIR (default: build/ in this
# directory; binaries go to $BUILD_DIR/bin, the module to $BUILD_DIR). Every binary uses the same flags.
set -euo pipefail
cd "$(dirname "$0")"
HNSW=${1:-${HNSWLIB_DIR:-}}
if [ -z "$HNSW" ]; then echo "usage: ./build.sh HNSWLIB_DIR   (directory with hnswlib.h; see README.md)" >&2; exit 2; fi
if [ ! -f "$HNSW/hnswlib.h" ] && [ -f "$HNSW/hnswlib/hnswlib.h" ]; then HNSW="$HNSW/hnswlib"; fi
if [ ! -f "$HNSW/hnswlib.h" ]; then echo "hnswlib.h not found in $HNSW" >&2; exit 2; fi
PY=${PYTHON:-python3}
CXX=${CXX:-g++}
FLAGS=${FLAGS:-"-std=c++17 -O3 -march=native -DNDEBUG -pthread"}
OUT=${BUILD_DIR:-build}
mkdir -p "$OUT/bin"
$CXX $FLAGS -I"$HNSW" -o "$OUT/bin/build_hnsw.tmp" tools/build_hnsw.cpp && mv -f "$OUT/bin/build_hnsw.tmp" "$OUT/bin/build_hnsw"
$CXX $FLAGS -I"$HNSW" -o "$OUT/bin/update_hnsw.tmp" tools/update_hnsw.cpp && mv -f "$OUT/bin/update_hnsw.tmp" "$OUT/bin/update_hnsw"
$CXX $FLAGS -o "$OUT/bin/test_answer_queue.tmp" tests/test_answer_queue.cpp && mv -f "$OUT/bin/test_answer_queue.tmp" "$OUT/bin/test_answer_queue"
EXT=$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
INC=$($PY -c "import pybind11, sysconfig; print('-I' + pybind11.get_include() + ' -I' + sysconfig.get_paths()['include'])")
SHARED="-shared -fPIC"
if [ "$(uname -s)" = Darwin ]; then SHARED="$SHARED -undefined dynamic_lookup"; fi
$CXX $FLAGS $SHARED $INC -I"$HNSW" bindings.cpp -o "$OUT/cast_engine.tmp$EXT"
mv -f "$OUT/cast_engine.tmp$EXT" "$OUT/cast_engine$EXT"
echo "built: $OUT/bin/build_hnsw $OUT/bin/update_hnsw $OUT/bin/test_answer_queue $OUT/cast_engine$EXT ($FLAGS)"
