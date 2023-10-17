#ifndef inference_H
#define inference_H

#include <iostream>
#include <vector>
#include <string>
#include "xcl2.hpp"

#include "../Vitis_Library/Import_engine/import_engine.h"

using namespace std;



//model inference related
vector<vector<vector<bitsx>>> inferenceConv(vector<vector<bitsx>> input,vector<vector<vector<vector<bitsx>>>> weights);

vector<vector<bitsx>> inferenceFC(vector<vector<bitsx>> input, vector<vector<bitsx>> weights);

vector<vector<vector<vector<bitsx>>>> model_inference(Model m, vector<vector<vector<bitsx>>> input, cl::Kernel krnl);

void test_inference(Model m, string inputs, string results, cl::Context context,cl::Kernel krnl,cl::CommandQueue q, int ninputs);



//device related
void setKernelArgs(cl_int err, cl::Kernel &krnl, cl::Buffer in_buf, cl::Buffer weights_buf, cl::Buffer out_buf, int input_f,int input_c,int weights_c);

cl::Kernel  programDevice(int argc, char** argv, cl::CommandQueue &q, cl::Context &context, int &err);

void test_kernel(int n, cl::CommandQueue &q,cl::Kernel &krnl, cl::Context ctx);

string getWD();

#endif
