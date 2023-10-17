#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <algorithm>
#include "hls_stream.h"
#include <stdio.h>
#include <vector>
#include <cmath>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <hls_math.h>

#include "../../matrix_product/src/Vitis_Library/library_constants.h"

typedef hls::stream<double> stream_f;

int read_inp_count=0,write_inp_count=0,
	read_ws_count=0,write_ws_count=0,
	read_res_count=0,write_res_count=0;


void read_input(stream_t (&inp),const bitsx *input)
{
#ifndef SYNTHESIS
			//std::cout<<std::endl<<"read input "<<std::endl;
#endif
read_input_outer:for(int i=0;i<TILE_SIZE;i++)
	{
read_input_inner:for(int j=0;j<TILE_SIZE;j++)
		{
#pragma HLS PIPELINE II=1
			write_inp_count++;

			inp.write(input[(i+(TILE_SIZE-j-1)*TILE_SIZE)]);//j*TS+i?	//input[(j+1)*TILE_SIZE-1-i]
#ifndef SYNTHESIS
			//std::cout<<"se escribe en el stream de entrada de input el valor "<<input[(i+(TILE_SIZE-j-1)*TILE_SIZE)]<<std::endl;

		}
	}
	std::cout<<std::endl;//<<"read input done"<<std::endl;
#endif
}

void read_ws(stream_t (&weights),const bitsx *ws)
{
#ifndef SYNTHESIS
			//std::cout<<std::endl<<"read weights "<<std::endl;
#endif
read_ws_outer:for(int i=0;i<TILE_SIZE;i++)
	{
read_ws_inner:for(int j=0;j<TILE_SIZE;j++)
		{
#pragma HLS PIPELINE II=1

			weights.write(ws[TILE_SIZE*i+j]);  //ws[(i+1)*TILE_SIZE-1-j] reverse por bloques
			write_ws_count++;				   //ws[(TILE_SIZE*TILE_SIZE)-(i*TILE_SIZE+j)-1] reverse completo
											   //ws[(TILE_SIZE*TILE_SIZE)-1-((i+1)*TILE_SIZE-1-j)] reverse por bloques, bloques reversed
#ifndef SYNTHESIS
			//std::cout<<"se escribe en el stream de weights de entrada el valor "<<ws[TILE_SIZE*i+j]<<std::endl;
		}
	}
	std::cout<<std::endl;//<<"read weights done"<<std::endl;
	#endif
}


void processingElement(stream_t (&inp_in),
		               stream_t (&inp_out),
					   stream_t (&weights_in),
					   stream_t (&weights_out),
					   stream_t (&out_stream),
					   int i
					   )
{
	bitsx result_buffer[TILE_SIZE]={};


#ifndef SYNTHESIS
	//std::cout<<std::endl<<"Lanzado fila PE"<<i<<std::endl;
#endif

launch_PEs:for(int l=0;l<TILE_SIZE;l++)
			{
#ifndef SYNTHESIS
	//std::cout<<std::endl<<"Lanzado PE"<<i<<l<<std::endl;
#endif
//--------------------------------------------------------

				  bitsx inp_val;

	forward_input:for(int k=0;k<TILE_SIZE-i;k++)
				  {
#pragma HLS PIPELINE II=1
					inp_val = inp_in.read();
					#pragma HLS PIPELINE II=1			//se lee el valor del PE y se hace forward del resto DONE

					if(k<TILE_SIZE-i-1)
					{
						inp_out.write(inp_val);
#ifndef SYNTHESIS
		//std::cout<<"valor "<<inp_val<<" escrito en input stream "<<i+1<<std::endl;
#endif
					}
				  }
//--------------------------------------------------------



#ifndef SYNTHESIS
		//std::cout<<"own value: "<<inp_val<<std::endl<<std::endl;
#endif
		bitsx acc=0;
		 //se se lee de weights
calculate_results:for(int x=0;x<TILE_SIZE;x++)
		{
#ifndef SYNTHESIS
			//std::cout<<"weight leido: "<<weights_val<<std::endl;
			//std::cout<<acc<<" acc += "<<inp_val<<" * "<<weights_val<<std::endl;
#endif
#pragma HLS PIPELINE II=1

			bitsx weights_val=weights_in.read();
			acc=result_buffer[x];

			acc+= inp_val * weights_val;

			result_buffer[x]=acc;

#ifndef SYNTHESIS
			//std::cout<<"result buffer["<<x<<"] = "<<acc<<std::endl;
#endif
			if(i<TILE_SIZE-1)
			{
#ifndef SYNTHESIS
				//std::cout<<"forward weight "<<weights_val<<std::endl;
#endif
				weights_out.write(weights_val);
			}

		}






		if(l==TILE_SIZE-1)
		{
#ifndef SYNTHESIS
			//std::cout<<"Escribir resultados de la fila "<<i<<std::endl;
#endif
save_outputs:for(int y=0;y<TILE_SIZE;y++)
			{
#pragma HLS PIPELINE II=1
				out_stream.write(result_buffer[y]);
			}

		}


}


}


//
//					         	 	  (Weights stream 0)
//   					     	   |          |          |
//			   		          	   v          v          v
//  					      	  ___        ___        ___
// 					 	 /  	 |   |      |   |      |   |
// processingElement0 --|	A0_->|i00| ---> |i01| ---> |i02| --->
//					  	 \       |___|      |___|      |___|
//        					   	   |		  |			 |
//         					   	   v		  v		     v
//									  (Weights stream 1)
//    							  
//								  __        ___        ___
//					     /       |   |      |   |      |   |
// processingElement1 --|	A1_->|i10| ---> |i11| ---> |i12| --->
//       			     \		 |___|      |___|      |___|
//      					   	   |		  |			 |
//        					   	   v		  v		     v
//									  (Weights stream 2)
//  					     	  ___        ___        ___
//      			     /	 	 |   |      |   |      |   |
// processingElement2 --|	A2_->|i20| ---> |i21| ---> |i21| --->
//     				     \     	 |___|      |___|      |___|
//       					   	   |		  |			 |
//      					   	   v		  v		     v


void write_result(stream_t (&otp)[TILE_SIZE], bitsx *result)
{
	bitsx val;

write_result_outer:for(int i=0;i<TILE_SIZE;i++)
	{
write_result_inner:for(int j=0;j<TILE_SIZE;j++)
		{
#pragma HLS PIPELINE II=1
			val = otp[i].read();
			read_res_count++;
			result[i*TILE_SIZE+j]=val;
		}
	}
}

extern "C" {

void top(const bitsx *input,
		 const bitsx *ws,
		 bitsx *result
		 )
{
#ifndef SYNTHESIS
	//std::cout<<"kernel start "<<std::endl;
#endif
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=ws offset=slave bundle=gmem1 max_widen_bitwidth=1024
#pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem3

#pragma HLS DATAFLOW

    stream_t inp[TILE_SIZE+1];
#pragma HLS ARRAY_PARTITION dim=1

    stream_t weights[TILE_SIZE+1];
#pragma HLS ARRAY_PARTITION dim=1

    stream_t otp[TILE_SIZE];
#pragma HLS ARRAY_PARTITION dim=1

#ifndef SYNTHESIS
//std::cout<<"reading inputs "<<std::endl;
#endif
read_inputs:read_input(inp[0],input);
#ifndef SYNTHESIS
//std::cout<<"reading inputs done"<<std::endl;

//std::cout<<"reading weights"<<std::endl;
#endif
read_weights:read_ws(weights[0],ws);
#ifndef SYNTHESIS
//std::cout<<"reading weights done"<<std::endl;
//std::cout<<std::endl<<std::endl<<"streams_info after reading parameters: "<<read_count<<" read, "<<write_count<<" written "<<std::endl;

#endif
pe_fils:for(int i=0;i<TILE_SIZE;i++)
	{
#pragma HLS UNROLL

			processingElement(inp[i],
							  inp[i+1],
							  weights[i],				// 15 18 21
							  weights[i+1],				// 42 54 66
							  otp[i],					// 69 90 111
							  i
							  );

	}

#ifndef SYNTHESIS
	//std::cout<<"PE's done "<<std::endl;
	//std::cout<<"writing results "<<std::endl;
#endif

write_back:write_result(otp,result);
#ifndef SYNTHESIS
	//std::cout<<"writing results done"<<std::endl;
#endif

//std::cout<<std::endl<<"streams_info at kernel end:\n\t inputs: "<<read_inp_count<<" read, "<<write_inp_count<<" written "
//									<<"\n\t weights: "<<read_ws_count<<" read, "<<write_ws_count<<" written "
//									<<"\n\t res: "<<read_res_count<<" read, "<<write_res_count<<" written "<<std::endl;


#ifndef SYNTHESIS
/*
std::cout<<"kernel result: "<<std::endl;
for(int i=0;i<TILE_SIZE;i++)
{
	for(int j=0;j<TILE_SIZE;j++)
	{
		std::cout<<result[i*TILE_SIZE+j]<<" ";
	}
	std::cout<<std::endl;
}

std::cout<<"fin kernel";
*/
#endif



}


}
