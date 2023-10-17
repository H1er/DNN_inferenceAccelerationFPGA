#ifndef LIBRARY_CONSTANTS_H
#define LIBRARY_CONSTANTS_H

#include <iostream>
#include <string>
#include <cstring>
#include <ap_fixed.h>
#include <hls_stream.h>

const std::string PYTHON_COMMAND = "python3";

const std::string LIB_PATH = "/home/h1er/Vitis_TFG_workspace/matrix_product/src/Vitis_Library";

const int TILE_SIZE=400;

typedef /*ap_fixed<32,16,AP_RND_ZERO,AP_WRAP_SM>*/ float bitsx;

typedef /*ap_fixed<32,16,AP_RND_ZERO,AP_WRAP_SM>*/ float data_t;

typedef hls::stream<bitsx> stream_t;

#endif


