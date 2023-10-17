#include <iostream>
#include <string>
#include <cstring>
#include <cmath>
#include <fstream>
#include <tuple>
#include <ap_fixed.h>
#include <ctime>
#include "xcl2.hpp"


#include "../Vitis_Library/inference.h"
#include "Import_engine/import_engine.h"
#include "../Vitis_Library/library_constants.h"

int contx=0;



using namespace std;

int mcd(int a, int b)
{
    if (b == 0)
    {
        return a;
    }
    return mcd(b, a % b);
}

vector<vector<vector<vector<bitsx>>>> sliceMatrix(vector<vector<bitsx>> matrix)
{
	vector<vector<vector<vector<bitsx>>>> sliced;
	/* dividir filter e input_img en tile_size x tile_size e ir pasandolos al kernel */
	int fils=matrix.size();
	int cols=matrix[0].size();
	int tiles_fils=ceil(((double)fils/TILE_SIZE));
	int tiles_cols=ceil(((double)cols/TILE_SIZE));

	/*cout<<"slicemat"<<endl;
	for(int i=0;i<fils;i++)
		{
			for(int j=0;j<cols;j++)
			{
				cout<<matrix[i][j]<<" ";
			}
			cout<<endl;
		}*/

	vector<vector<vector<bitsx>>> ftiles;
	vector<vector<bitsx>> tile;
	vector<bitsx> vect;

	for(int i=0;i<tiles_fils;i++)
	{
		for(int j=0;j<tiles_cols;j++)
		{
			//cout<<"tile "<<i<<" "<<j<<endl;

			int tfil=0;

			for(int k=i*TILE_SIZE;k<i*TILE_SIZE+TILE_SIZE;k++)
			{
				for(int l=j*TILE_SIZE;l<j*TILE_SIZE+TILE_SIZE;l++)
				{
					//cout<<i<<" "<<j<<" "<<k<<" "<<l<<endl;
					if(k>=fils||l>=cols)
					{
						//cout<<"out of bounds	"<<k<<" "<<l<<" -> "<<0<<"		"<<endl;
						vect.push_back( 0 );
					}
					else
					{
						//cout<<"		"<<k<<" "<<l<<" -> "<<matrix[k][l]<<"		";
						vect.push_back( matrix[k][l] );
						/*if(matrix[k][l]!=0)
						{
							cout<<"pushing non-zero value: "<<matrix[k][l]<<endl;
						}*/

					}
				}
				tile.push_back(vect);
				vect.clear();
				tfil++;
				//cout<<endl;
			}
			ftiles.push_back(tile);
			tile.clear();
		}
		sliced.push_back(ftiles);
		ftiles.clear();

	}
	return sliced;
}

vector<vector<vector<vector<bitsx>>>> sliceMatrixInputDense(vector<vector<vector<vector<bitsx>>>> matrix)
{
	int fils=matrix.size();
	int x   =matrix[0].size();
	int y   =matrix[0][0].size();
	int cols=matrix[0][0][0].size();

	/* dividir filter e input_img en tile_size x tile_size e ir pasandolos al kernel */


	vector<vector<bitsx>> mat(fils,vector<bitsx>(cols,0));

	cout<<"matrix dims: "<<fils<<"x"<<x<<"x"<<y<<"x"<<cols<<endl;

	cout<<"idense fils: "<<fils<<endl;
	cout<<"idense cols: "<<cols<<endl;

	cout<<"input_mat "<<endl;
	for(int i=0;i<fils;i++)
	{
		for(int j=0;j<cols;j++)
		{
			mat[i][j]=matrix[i][0][0][j];
		}
	}

	return sliceMatrix(mat);
}

//writes the matrix in the buffer matrix
void load_buffer_mat(vector<bitsx, aligned_allocator<bitsx>> &vector_fpga, vector<vector<bitsx>> matrix)
{
	//cout<<endl<<"loading data to buffer... "<<endl;
	int cont=0;
	for(int i=0;i<TILE_SIZE;i++)
	{
		for(int j=0;j<TILE_SIZE;j++)
		{
			vector_fpga[i*TILE_SIZE+j]=matrix[i][j];
			cont++;
		}
	}
}

void get_buffer_mat(vector<vector<bitsx>> &matrix, vector<bitsx, aligned_allocator<bitsx>> vector_fpga)
{
	matrix.clear();
	vector<bitsx> aux;

	//cout<<"Getting result from kernel "<<endl;
	for(int i=0;i<TILE_SIZE;i++)
	{
		for(int j=0;j<TILE_SIZE;j++)
		{
			//std::cout<<vector_fpga[i*TILE_SIZE+j]<<" ";
			aux.push_back(vector_fpga[i*TILE_SIZE+j]);

		}
		//cout<<endl;
		matrix.push_back(aux);
		aux.clear();
		//std::cout<<std::endl;
	}
	//cout<<"Buffer results loaded back to host"<<endl;
}

void launch_MM_Kernel(cl::Buffer input_buffer,cl::Buffer weights_buffer,cl::Buffer output_buffer, cl::CommandQueue &q,cl::Kernel &krnl)
{
	cl_int err;


	//std::cout << "Moving inputs and filters from host to device" << std::endl;
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({input_buffer, weights_buffer}, 0));
	OCL_CHECK(err, err = q.finish());


	//std::cout << "Launching kernel\n";
	OCL_CHECK(err, err = q.enqueueTask(krnl));
	OCL_CHECK(err, err = q.finish());


	//std::cout << "Moving output from device to host\n";
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({output_buffer}, CL_MIGRATE_MEM_OBJECT_HOST));
	OCL_CHECK(err, err = q.finish());
}

cl::Kernel  programDevice(int argc, char** argv, cl::CommandQueue &q, cl::Context &context, int &err)
{
	if (argc != 2)
	{
		std::cout << "Usage: " << argv[0] << "<XCLBIN File>" << std::endl;
		exit(EXIT_FAILURE);
	}

	std::string binaryFile = argv[1];
	cl::Program program;
	cl::Kernel krnl;

	auto devices = xcl::get_xil_devices();
	auto fileBuf = xcl::read_binary_file(binaryFile);
	cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
	bool valid_device = false;
	for (unsigned int i = 0; i < devices.size(); i++)
	{
		auto device = devices[i];
		// Creating Context and Command Queue for selected Device
		OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
		OCL_CHECK(err, q = cl::CommandQueue(context, device, 0/*CL_QUEUE_PROFILING_ENABLE*/, &err));
		std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
		cl::Program program(context, {device}, bins, nullptr, &err);

		if (err != CL_SUCCESS)
		{
			std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
		}
		else
		{
			std::cout << "Device[" << i << "]: program successful!\n";
			OCL_CHECK(err, krnl = cl::Kernel(program, "top", &err));
			valid_device = true;
			break; // we break because we found a valid device
		}
	}
	if (!valid_device)
	{
		std::cout << "Failed to program any device found, exit!\n";
		exit(EXIT_FAILURE);
	}

	return krnl;
}

void setKernelArgs(cl_int err, cl::Kernel &krnl, cl::Buffer in_buf, cl::Buffer weights_buf, cl::Buffer out_buf)
{
	int n_arg=0;

	OCL_CHECK(err, err = krnl.setArg(n_arg++, in_buf));

	OCL_CHECK(err, err = krnl.setArg(n_arg++, weights_buf));

	OCL_CHECK(err, err = krnl.setArg(n_arg++, out_buf));


	//cout<<"Kernel args setted"<<endl<<endl;
}




bool testeq(vector<vector<bitsx>> v1, vector<vector<bitsx>>v2)
{
	bool eq=true;
	if(v1.size()!=v2.size() || v1[0].size()!=v2[0].size())
	{
		cout<<"dims no coinciden "<<endl;
		return false;
	}
	else
	{
		for(int i=0;i<v2.size();i++)
		{
			for(int j=0;j<v2[0].size();j++)
			{
				if(v1[i][j]!=v2[i][j])
				{
					cout<<"not equal in position "<<i<<" "<<j<<": v1 -> "<<v1[i][j]<<" != v2 -> "<<v2[i][j]<<endl;
					return false;
				}
			}
		}
	}
	return eq;
}

//calcula la matriz traspuesta de la dada por parametro
vector<vector<bitsx>> transpose(vector<vector<bitsx>> A)
{
    int filsA = A.size();
    int colsA = A[0].size();

    vector<vector<bitsx>> B(colsA, vector<bitsx>(filsA, 0));

    for(int i=0; i<filsA; i++) 
    {
        for(int j=0; j<colsA; j++) 
        {
            B[j][i] = A[i][j];
        }
    }

    return B;
}

//devuelve la matriz resultado de multipilcar A * B
vector<vector<bitsx>> matrixmul(vector<vector<bitsx>> A,vector<vector<bitsx>> B)
{
    int filsA = A.size();
    int colsB = B[0].size();

    if(A[0].size()!=B.size())
    {
        cout<<"Error, incorrect dims for the input matrixes : "<<A.size()<<"x"<<A[0].size()<<" multiplied by "<<B.size()<<"x"<<B[0].size()<<endl;

        exit(-1);
    }
    vector<vector<bitsx>> C(filsA, vector<bitsx>(colsB, 0));

    for(int i=0; i<filsA; i++) 
    {
        for(int j=0; j<colsB; j++) 
        {
            for(int k=0; k<B.size(); k++) 
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

tuple<vector<vector<vector<bitsx>>>,vector<vector<vector<bitsx>>>> create_random_matrix_products(int n_matrixes)
{
	const int MAX_RANDOM_VALUE=20;

	vector<vector<vector<bitsx>>> mats,results;
	vector<vector<bitsx>> result,cleft,cright;
	vector<bitsx> left,right;

	for(int i=0;i<n_matrixes;i++)
	{
		//create left
		for(int j=0;j<TILE_SIZE;j++)
		{
			for(int k=0;k<TILE_SIZE;k++)
			{
				left.push_back(rand()%MAX_RANDOM_VALUE);
			}
			cleft.push_back(left);
			left.clear();
		}
		//cout<<"left dims: "<<cleft.size()<<"x"<<cleft[0].size()<<endl;
		for(int j=0;j<TILE_SIZE;j++)
		{
			for(int k=0;k<TILE_SIZE;k++)
			{
				right.push_back(rand()%MAX_RANDOM_VALUE);
			}
			cright.push_back(right);
			right.clear();
		}
		//cout<<"right dims: "<<cright.size()<<"x"<<cright[0].size()<<endl;

		result = matrixmul(cleft,cright);

		mats.push_back(cleft);
		mats.push_back(cright);
		results.push_back(result);

		cleft.clear();
		cright.clear();
	}

	return tuple<vector<vector<vector<bitsx>>>,vector<vector<vector<bitsx>>>>(mats,results);
}


void test_kernel(int n, cl::CommandQueue &q,cl::Kernel &krnl, cl::Context ctx)
{
	tuple<vector<vector<vector<bitsx>>>,vector<vector<vector<bitsx>>>> data=create_random_matrix_products(n);
	//cout<<"test data created..."<<endl;
	vector<vector<vector<bitsx>>> mats = get<0>(data);
	vector<vector<vector<bitsx>>> results = get<1>(data);
	vector<vector<bitsx>> left, right, result, res(TILE_SIZE,vector<bitsx>(TILE_SIZE, 0));
	cl_int err;


	for(int i=0;i<n;i++)
	{
		left = mats.at(2*i);
		right = mats.at((2*i)+1);
		result = results.at(i);

		//cout<<"dims "<<left.size()<<"x"<<left[0].size()<<"x"<<right[0].size()<<endl;
		vector<bitsx, aligned_allocator<bitsx>> input_fpga(TILE_SIZE*TILE_SIZE);
		vector<bitsx, aligned_allocator<bitsx>> weights_fpga(TILE_SIZE*TILE_SIZE);
	    vector<bitsx, aligned_allocator<bitsx>> result_fpga(TILE_SIZE*TILE_SIZE);

		OCL_CHECK(err, cl::Buffer input_buffer(ctx, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,(TILE_SIZE*TILE_SIZE)*sizeof(bitsx), input_fpga.data(),&err));
		OCL_CHECK(err, cl::Buffer weights_buffer(ctx, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,(TILE_SIZE*TILE_SIZE)*sizeof(bitsx), weights_fpga.data(),&err));
		OCL_CHECK(err, cl::Buffer result_buffer(ctx, CL_MEM_USE_HOST_PTR |  CL_MEM_READ_WRITE,(TILE_SIZE*TILE_SIZE)*sizeof(bitsx), result_fpga.data(),&err));
		//cout<<"buffers created..."<<endl;


		//cout<<"loading left..."<<endl;
		load_buffer_mat(input_fpga,left);

		//cout<<"loading right"<<endl;
		load_buffer_mat(weights_fpga,right);

		//cout<<"matrixes loaded..."<<endl;

		//cout<<"setting kernel args..."<<endl;
		setKernelArgs(err, krnl, input_buffer,weights_buffer,result_buffer);
		                   		//cout<<"kernel args setted"<<endl;
		//cout<<"launching kernel..."<<endl;
		launch_MM_Kernel(input_buffer,weights_buffer, result_buffer, q, krnl);
		//cout<<"kernel done..."<<endl;

		//volcar buffer de salida en layer output input
		//cout<<"reading results..."<<endl;
		get_buffer_mat(res, result_fpga);
		//cout<<"results read..."<<endl;

		if(!testeq(res,result))
		{
			cout<<"Results not equal, test failed "<<endl;
			exit(0);
		}
		/*
		cout<<"res"<<endl;
		for(int j=0;j<res.size();j++)
		{
			for(int k=0;k<res[0].size();k++)
			{
				cout<<res[j][k]<<" ";
			}
			cout<<endl;
		}

		cout<<endl<<"result"<<endl;
		for(int j=0;j<result.size();j++)
		{
			for(int k=0;k<result[0].size();k++)
			{
				cout<<result[j][k]<<" ";
			}
			cout<<endl;
		}

		cout<<"DONE"<<endl;*/


	}

	cout<<n<<" matrixes of size "<<TILE_SIZE<<"x"<<TILE_SIZE<<" tested OK "<<endl;



}

//devuelve el resultado de hacer la funcion relu en n
bitsx relu (bitsx n)
{
    return n>0?n:(bitsx)0;
}


//devuelve el resultado de aplicar la funcion de activacion f a n
bitsx activation(string f,bitsx n)
{
    bitsx result;
    if(strstr(f.c_str(),"relu"))
    {
        result=relu(n);
    }/*Añadir varias activation functions*/
    else if(strstr(f.c_str(),"linear"))
    {
        result=n;
    }
    else
    {
        cout<<"Activation function not recognised..."<<endl;
        exit(-1);
    }

    return result;
}

//aplica la funcion de activacion f sobre todos los valores de output
void apply_activation(string f, vector<vector<vector<vector<bitsx>>>> &output)
{
    for(int l=0;l<output.size();l++)
   	{
   		for(int i=0;i<output[l].size();i++)
   		{
   			for(int j=0;j<output[l][i].size();j++)
   			{
   				for(int k=0;k<output[l][i][j].size();k++)
   				{
   					output[l][i][j][k]=activation(f,output[l][i][j][k]);
   				}
   			}
   		}
   	}
}

//suma el valor de del sesgo correspoindiente del vector bias al resultado correspondiente
void apply_bias(vector<bitsx>bias , vector<vector<vector<vector<bitsx>>>> &output)
{
	for(int l=0;l<output.size();l++)
	{
		for(int i=0;i<output[l].size();i++)
		{
			for(int j=0;j<output[l][i].size();j++)
			{
				for(int k=0;k<output[l][i][j].size();k++)
				{
					output[l][i][j][k]+=bias[i];
				}
			}
		}
	}

}

void passResultDense(vector<vector<vector<vector<bitsx>>>> (&dest), vector<vector<bitsx>> og)
{

	vector<vector<vector<vector<bitsx>>>> res(og.size(),vector<vector<vector<bitsx>>>(1,vector<vector<bitsx>>(1,vector<bitsx>(og[0].size(),0))));

	vector<bitsx> aux;

	dest.clear();

	for(int l=0;l<og.size();l++)
	{
		for(int i=0;i<1;i++)
		{
			for(int j=0;j<1;j++)
			{
				for(int k=0;k<og[0].size();k++)
				{
					//cout<<"assigning "<<l<<" "<<k<<" to "<<l<<" "<<i<<" "<<j<<" "<<k<<" "<<endl;
					res[l][i][j][k]=og[l][k];
				}
			}
		}
	}

	dest=res;


}

void passResult(vector<vector<bitsx>> (&dest), vector<vector<bitsx>> og)
{
	dest.clear();
	vector<bitsx> aux;

	for(int i=0;i<og.size();i++)
	{
		aux.clear();
		for(int j=0;j<og[0].size();j++)
		{
			aux.push_back(og[i][j]);
		}
		dest.push_back(aux);
	}



}


void locateTile(vector<vector<bitsx>> tile,vector<vector<bitsx>> &(result), int tile_fil, int tile_col)
{
	int fil = tile_fil*TILE_SIZE;
	int col = tile_col*TILE_SIZE;

	//cout<<"locating tile "<<tile_fil<<" "<<tile_col<<endl;

	for(int i=fil;i<fil+TILE_SIZE;i++)
	{
		for(int j=col;j<col+TILE_SIZE;j++)
		{
			if(!(i>=result.size() || j>=result[0].size())) //mientras el tile no se salga de la matriz
			{
				//cout<<"result ["<<i<<"]["<<j<<"] += tile["<<i%TILE_SIZE<<"]["<<j%TILE_SIZE<<"]locating in "<<i<<" "<<j<<endl;
				result[i][j]+=tile[i%TILE_SIZE][j%TILE_SIZE];
			}
			else
			{
				//cout<<"out of bounds "<<i<<" "<<j<<endl;
			}
		}
	}
}

vector<vector<bitsx>> getChannelMat(vector<vector<vector<vector<bitsx>>>> input, int channel)
{
	vector<vector<bitsx>> mat(input[0][0].size(),vector<bitsx>(input.size(),0));
	int mat_fils=mat.size();
	int mat_cols=mat[0].size();

	cout<<"mat dims: "<<mat.size()<<"x"<<mat[0].size()<<endl;

	for(int i=0;i<mat_fils;i++)
	{
		for(int j=0;j<mat_cols;j++)
		{
			mat[i][j]=input[j][channel][i][0];
		}
	}

	return mat;
}



//devuelve el resultado de procesar el input por la capa convolucional que tiene los valores de pesos de weights
vector<vector<vector<vector<bitsx>>>> inferenceConv(vector<vector<vector<vector<bitsx>>>>input,
                                             vector<vector<vector<vector<bitsx>>>> weights,
											 cl::Kernel krnl,
											 cl::CommandQueue q,
											 cl::Context context,
											 size_t total_size
											 )
{
	int ninputs = input.size();
    int input_height = input[0][0].size();
    int input_width = input[0][0][0].size();
    int num_channels_input = input[0].size();
    //cout<<"pre-transpose check"<<endl;
    //cout<<"Weights dims "<<weights.size()<<"x"<<weights[0].size()<<"x"<<weights[0][0].size()<<"x"<<weights[0][0][0].size()<<endl;
    //cout<<"Input dims "<<input.size()<<"x"<<input[0].size()<<"x"<<input[0][0].size()<<"x"<<input[0][0][0].size()<<endl<<endl;

  
    int num_channels_weights = weights.size();
    int num_filters = weights[0].size();
    int filter_height = weights[0][0].size();
    int filter_width = weights[0][0][0].size();


    //cout<<"num filters "<<num_filters<<endl<<"filter width "<<filter_width<<endl<<"filter height "<<filter_height<<endl;

    if(!(filter_width == input_height)) //dimensiones no coinciden
    {
        if(filter_width==input_width) //hace falta traspuesta
        {
            //cout<<"Trasposing input..."<<endl;

            for(int k=0;k<input.size();k++)
            {
            	for(int i=0;i<input[0].size();i++)//para cada canal se hace la traspuesta de todas las matrices de input
            	{
            		//cout<<"Trasposing channel "<<i<<endl;
            		vector<vector<bitsx>>transposed = transpose(input[k][i]);

            		//cout<<"Input dims "<<input.size()<<"x"<<input[i].size()<<"x"<<input[i][0].size()<<endl;
            		//cout<<"Trasposed dims "<<transposed.size()<<"x"<<transposed[0].size()<<endl;
            		input[k][i] = transposed;

            		//cout<<"Dims after trasposing channel "<<i<<"  "<<input.size()<<"x"<<input[i].size()<<"x"<<input[i][0].size()<<endl;

            		transposed.clear();
            	}
            }

            //cout<<"Input dims after trasnsposing "<<input.size()<<"x"<<input[0].size()<<"x"<<input[0][0].size()<<endl;
            //cout<<"num filters "<<num_filters<<endl<<"filter width "<<filter_width<<endl<<"filter height "<<filter_height<<endl;

            input_height = input[0][0].size();
            input_width = input[0][0][0].size();
            num_channels_input = input[0].size();
        }
        else
        {
            cout<<"error, input and weights matrixes dimensions are incorrect!  input  "<<input_height<<"x"<<input_width<<"      output  "<<filter_height<<"x"<<filter_width<<endl;
            exit(-1);
        }
    }
  
    if(!(num_channels_input==num_channels_weights))
    {
        cout<<"Canales no coinciden"<<endl;
        exit(-1);
    }

    vector<vector<vector<vector<bitsx>>>> output=vector<vector<vector<vector<bitsx>>>>(ninputs, vector<vector<vector<bitsx>>>(num_filters,vector<vector<bitsx>>(filter_height,vector<bitsx>(input_width,0))));

    //cout<<"output dims "<<output.size()<<"x"<<output[0].size()<<"x"<<output[0][0].size()<<"x"<<output[0][0][0].size()<<endl<<endl;

    cl_int err;

    //cout<<"post-transpose check"<<endl;
    //cout<<"Weights dims "<<weights.size()<<"x"<<weights[0].size()<<"x"<<weights[0][0].size()<<"x"<<weights[0][0][0].size()<<endl;
    //cout<<"Input dims "<<input.size()<<"x"<<input[0].size()<<"x"<<input[0][0].size()<<"x"<<input[0][0][0].size()<<endl<<endl;

 	//cout<<"num channels: "<<num_channels_input<<endl;
 	//cout<<"num filters: "<<num_filters<<endl<<endl;


 	 vector<vector<vector<bitsx>>> acc(num_filters,vector<vector<bitsx>>(filter_height,vector<bitsx>(ninputs,0)));
 	//cout<<"acc dims: "<<acc.size()<<"x"<<acc[0].size()<<"x"<<acc[0][0].size()<<endl;


    for(int j=0;j<num_channels_input;j++)
    {
    	vector<vector<bitsx>> input_img=getChannelMat(input,j); //ERR (tantas filas como pixeles y tantas columnas como entradas)

        vector<vector<vector<vector<bitsx>>>> slicedWeights = sliceMatrix(input_img); //ERR

        int weights_tiles_cols=slicedWeights[0].size(); //ERR



        for(int i=0;i<num_filters;i++)
        {
        	//cout<<"filter "<<i<<" channel "<<j<<endl;
            vector<vector<bitsx>> filter = weights[j][i];

        	vector<bitsx, aligned_allocator<bitsx>> input_fpga(TILE_SIZE*TILE_SIZE);
        	vector<bitsx, aligned_allocator<bitsx>> weights_fpga(TILE_SIZE*TILE_SIZE);
        	vector<bitsx, aligned_allocator<bitsx>> result_fpga(TILE_SIZE*TILE_SIZE);

        	/* dividir filter e input_img en tile_size x tile_size e ir pasandolos al kernel */
        	vector<vector<vector<vector<bitsx>>>> slicedInput = sliceMatrix(filter);

            //cout<<"creating buffers fpga..."<<endl<<endl;
            //creamos los buffers asociendo los vectores creados anteriormente
            OCL_CHECK(err, cl::Buffer input_buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, total_size, input_fpga.data(),&err));
            OCL_CHECK(err, cl::Buffer weights_buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, total_size, weights_fpga.data(),&err));
        	OCL_CHECK(err, cl::Buffer result_buffer(context, CL_MEM_USE_HOST_PTR |  CL_MEM_READ_WRITE, total_size, result_fpga.data(),&err));

        	vector<vector<bitsx>> res;

        	vector<vector<bitsx>> final(filter.size(),vector<bitsx>(input_img[0].size(),0));

        	//cout<<"final dims: "<<final.size()<<"x"<<final[0].size()<<endl;
        	//exit(0);
        	/*if(contx>0)
        	  {
        	  	  exit(0);
        	   }
        	        contx++;
        	*/

        	vector<vector<vector<bitsx>>> partials;

            int input_tiles_fils=slicedInput.size();
            int input_tiles_cols=slicedInput[0].size();

            //cout<<"starting matrix product..."<<endl;
            for(int s=0;s<input_tiles_fils;s++) //se calcula el producto de las matrices
            {
            	for(int k=0;k<input_tiles_cols;k++)
            	{
            		for(int l=0;l<weights_tiles_cols;l++)
            		{
            			//cout<<s<<" "<<k<<" -> "<<k<<" "<<l<<std::endl;

                   		//cout<<"loading input"<<endl;
                   		load_buffer_mat(input_fpga,slicedInput[s][k]);
                   		//cout<<"input loaded "<<endl;

                   		//cout<<"loading weights"<<endl;
                   		load_buffer_mat(weights_fpga,slicedWeights[k][l]);
                   		//cout<<"weights loaded "<<endl;

                   		//cout<<"setting kernel args..."<<endl;
                   		setKernelArgs(err, krnl, input_buffer,weights_buffer,result_buffer);
                   		//cout<<"kernel args setted"<<endl;

                   		launch_MM_Kernel(input_buffer,weights_buffer, result_buffer, q, krnl);
                 		//volcar buffer de salida en layer output input

                   		get_buffer_mat(res, result_fpga);
                   		//exit(0);

                   		partials.push_back(res);
                   	}
                }

            	for(int x=0;x<partials.size();x++)
                {
            		//se ubica en el tile i,l%weights_tiles_col
                    locateTile(partials[x],final,s,x%weights_tiles_cols);
                }

            	partials.clear();
            }


            //cout<<"final dims "<<final.size()<<"x"<<final[0].size()<<endl;


            //HAY QUE ACUMULAR LOS RESULTADOS DEL MISMO FILTRO EN DIFERENTES CANALES
          //  cout<<"accumulating results filter "<<i<<" channel "<<j<<endl;
            for(int u=0;u<final.size();u++)
            {
                for(int v=0;v<final[0].size();v++)
                {

                	/*
                	 * se acumula en una matriz 2d, y después cada columna se guarda como resultado de
                	 * cada entrada para el mismo canal
                	 *
                	 * */
                	//cout<<"acc "<<u<<" "<<v<<" -> final "<<u<<" "<<v<<endl;
                    acc[i][u][v]+=final[u][v]; //ERR  // se podría no resetear final al principio del
                    							   // bucle y acumular directamente los resultados en final?
                }
                //cout<<endl;
            }
            //cout<<"final dims "<<final.size()<<"x"<<final[0].size()<<endl;
            //cout<<"acc dims "<<acc.size()<<"x"<<acc[0].size()<<endl;

            final.clear();


        }

        /*asignar cada fila de acc al canal de salida correspondiente al bucle para cada entrada*/
        //20x2x676x1
    }

    //cout<<"saving final outputs"<<endl;
    //cout<<"output dims "<<output.size()<<"x"<<output[0].size()<<"x"<<output[0][0].size()<<"x"<<output[0][0][0].size()<<endl;
    //cout<<"acc dims "<<acc.size()<<"x"<<acc[0].size()<<"x"<<acc[0][0].size()<<endl;


    for(int m=0;m<acc.size();m++)
    {
    	for(int n=0;n<acc[0].size();n++)
    	{
    		for(int y=0;y<acc[0][0].size();y++)
    		{
    			output[y][m][n][0]=acc[m][n][y];
    		}

    	}
    }




    return output;
}


//devuelve el resultado de aplanar el contenido de input
vector<vector<vector<vector<bitsx>>>> flatten(vector<vector<vector<vector<bitsx>>>> input)
{
	vector<vector<vector<vector<bitsx>>>> result;
    vector<vector<vector<bitsx>>> output;
    vector<vector<bitsx>> aux;
    vector<bitsx> flattened;

    int ninputs = input.size();
    int channels = input[0].size();
    int input_height = input[0][0].size();
    int input_width = input[0][0][0].size();
   
    for(int l=0;l<ninputs;l++)
    {
    	for(int j=0;j<input_height;j++)
    	{
    		for(int k=0;k<input_width;k++)
    		{
    			for(int i=0;i<channels;i++)
    			{
    				flattened.push_back(input[l][i][j][k]);
    			}
    		}
    	}

    	aux.push_back(flattened);
    	output.push_back(aux);
    	result.push_back(output);

    	output.clear();
    	aux.clear();
    	flattened.clear();
    }

    cout<<"flatten results dims: "<<result.size()<<"x"<<result[0].size()<<"x"<<result[0][0].size()<<"x"<<result[0][0][0].size()<<endl;
    return result;
}


//calcula 3l resultado de la inferencia de la entrada input en el modelo con los datos de inference_data
vector<vector<vector<vector<bitsx>>>> model_inference( Model m, vector<vector<vector<vector<bitsx>>>> input, cl::Context context,cl::Kernel krnl, cl::CommandQueue q)
{
	cout<<"Getting inference data... "<<endl;
    vector<tuple<string,string,vector<bitsx>,vector<vector<vector<vector<bitsx>>>>>> inference_data = m.getInferenceData();
    cout<<"Inference data recieved..."<<endl<<endl;

    vector<vector<vector<vector<bitsx>>>> layer_output_input=input;

    cl_int err;
    size_t total_size=0;

    for(int i=0;i<inference_data.size();i++)
    {
        string type = get<0>(inference_data[i]);
        string activation = get<1>(inference_data[i]);
        vector<bitsx> bias = get<2>(inference_data[i]);
        vector<vector<vector<vector<bitsx>>>> weights= get<3>(inference_data[i]);

        if(strstr(type.c_str(),"dense"))
		{
        	vector<vector<bitsx>> in_mat,ws,in_mat2;
        	total_size = TILE_SIZE*TILE_SIZE*sizeof(bitsx);

            /* dividir weights 0 0 y layer_input_output en tile_size x tile_size e ir pasandolos al kernel */

        	vector<vector<bitsx>> res,m1(3,vector<bitsx>(3,0)),m2(3,vector<bitsx>(3,0));
        	vector<vector<vector<bitsx>>> partials;
        	/*
        	cout<<"creating test mats"<<endl;
        	for(int i=0;i<3;i++)
        	{
        		for(int j=0;j<3;j++)
        		{
        			m1[i][j]=3*i+j;

        		}
        	}
        	for(int i=0;i<3;i++)
        	{
        		for(int j=0;j<3;j++)
        	    {
        	        m2[i][j]=6*i+j;
        	    }
        	}
        	cout<<"m1"<<endl;
        	for(int i=0;i<3;i++)
        	{
        		for(int j=0;j<3;j++)
        	    {
        			cout<<m1[i][j]<<" ";
        	    }
        	    cout<<endl;
         	 }
        	cout<<endl<<endl;

        	cout<<"m2"<<endl;
        	for(int i=0;i<3;i++)
        	{
        		for(int j=0;j<3;j++)
        	    {
        			cout<<m2[i][j]<<" ";
        	    }
        	    cout<<endl;
        	 }
        	 cout<<endl<<endl;
        	 */

        	 int input_fils=layer_output_input.size();//3
        	 int input_cols=layer_output_input[0][0][0].size();//6
        	 int weights_cols=weights[0][0][0].size();//3



        	vector<vector<vector<vector<bitsx>>>> slicedInput = sliceMatrixInputDense(layer_output_input); //sliceMatrix(m1);//
        	//cout<<"input sliced "<<endl;

        	vector<vector<vector<vector<bitsx>>>> slicedWeights = sliceMatrix(/*m2*/weights[0][0]);

          	//cout<<"input dims "<<input_fils<<"x"<<input_cols<<endl;
            //cout<<"weights dims "<<input_cols<<"x"<<weights_cols<<endl<<endl<<endl;

        	//cout<<"sliced input tiles dims "<<slicedInput.size()<<"x"<<slicedInput[0].size()<<endl;
        	//cout<<"sliced weights tiles dims "<<slicedWeights.size()<<"x"<<slicedWeights[0].size()<<endl;

        	//cout<<"final matrix dims "<<input_fils<<"x"<<weights_cols;
        	vector<vector<bitsx>> final(input_fils,vector<bitsx>(weights_cols,0));

        	int input_tiles_fils=slicedInput.size(); //ERR
        	int input_tiles_cols=slicedInput[0].size(); //ERR
        	int weights_tiles_cols=slicedWeights[0].size();

        	vector<bitsx, aligned_allocator<bitsx>> input_fpga(TILE_SIZE*TILE_SIZE);
        	vector<bitsx, aligned_allocator<bitsx>> weights_fpga(TILE_SIZE*TILE_SIZE);
        	vector<bitsx, aligned_allocator<bitsx>> result_fpga(TILE_SIZE*TILE_SIZE);

        	//cout<<"creating buffers fpga..."<<endl<<endl;
        	//creamos los buffers asociendo los vectores creados anteriormente
        	OCL_CHECK(err, cl::Buffer input_buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, total_size, input_fpga.data(),&err));
        	OCL_CHECK(err, cl::Buffer weights_buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, total_size, weights_fpga.data(),&err));
        	OCL_CHECK(err, cl::Buffer result_buffer(context, CL_MEM_USE_HOST_PTR |  CL_MEM_WRITE_ONLY, total_size, result_fpga.data(),&err));

        	/*cout<<"final pre kernel"<<endl;
        	for(int i=0;i<input_fils;i++)
        	{
        		for(int j=0;j<weights_cols;j++)
        	    {
        			cout<<final[i][j]<<" ";
        	    }
        	    cout<<endl;
        	}
        	cout<<endl<<endl;*/


        	//cout<<"all sliced "<<endl<<endl;

        	//cout<<"input_tiles_fils: "<<input_tiles_fils<<endl;
        	//cout<<"input_tiles_cols: "<<input_tiles_cols<<endl;
        	//cout<<"weights_tiles_cols: "<<weights_tiles_cols<<endl<<endl;



        	for(int j=0;j<input_tiles_fils;j++) //se calcula el producto de los tiles
        	{
        		for(int k=0;k<input_tiles_cols;k++)
        		{
        			for(int l=0;l<weights_tiles_cols;l++)
        			{
       					//cout<<j<<" "<<k<<" -> "<<k<<" "<<l<<std::endl;

       					//cout<<"loading input"<<endl;
       					load_buffer_mat(input_fpga,slicedInput[j][k]);
       					//cout<<"input loaded "<<endl;

       					//cout<<"loading weights"<<endl;
       					load_buffer_mat(weights_fpga,slicedWeights[k][l]);
       					//cout<<"weights loaded "<<endl;

       					//cout<<"setting kernel args..."<<endl;
       					setKernelArgs(err, krnl, input_buffer,weights_buffer,result_buffer);
       					//cout<<"kernel args setted"<<endl;
       					//contx++;
       					//cout<<"launching kernel"<<endl;
       					launch_MM_Kernel(input_buffer,weights_buffer, result_buffer, q, krnl);
       					//cout<<"kernel finished "<<endl;
       					//volcar buffer de salida en layer output input
       					//cout<<"writing kernel results to host"<<endl;
       					get_buffer_mat(res, result_fpga);
       					//cout<<"writing results done"<<endl;

       					//exit(0);		//poner para HW emu

       					input_fpga.clear();
       					weights_fpga.clear();

       					//cout<<endl<<endl<<endl;

       					partials.push_back(res);
       				}
        		}
        		//cout<<"locating tiles "<<endl;
        		for(int x=0;x<partials.size();x++)
        		{
        			//se ubica en el tile i,l%weights_tiles_col
        			locateTile(partials[x],final,j,x%weights_tiles_cols);
        		}
        		//cout<<"tiles located "<<endl;
        		partials.clear();
        	}

        	/*
        	cout<<"final in host"<<endl;
        	for(int i=0;i<input_fils;i++)
        	{
        		for(int j=0;j<weights_cols;j++)
        		{
        			cout<<final[i][j]<<" ";
        		}
        		cout<<endl;
        	}
        	exit(0);*/

        passResultDense(layer_output_input,final); //ERR

        apply_bias(bias,layer_output_input); //ERR

        apply_activation(activation,layer_output_input); //ERR




        }
		else if(strstr(type.c_str(),"conv2d"))
		{

		    total_size = TILE_SIZE*TILE_SIZE*sizeof(bitsx);



            //cout<<"convolutional inference "<<endl;
            layer_output_input = inferenceConv(layer_output_input,weights,krnl,q,context,total_size); //ERR

            apply_bias(bias,layer_output_input); //ERR
            
            apply_activation(activation,layer_output_input); //ERR

            //cout<<"Dims: "<<layer_output_input.size()<<"x"<<layer_output_input[0].size()<<"x"<<layer_output_input[0][0].size()<<endl;

        }
        else if(strstr(type.c_str(),"flatten"))
		{
            //cout<<"flatten inference "<<endl;
            layer_output_input = flatten(layer_output_input);



            //cout<<"Flattened output dims: "<<layer_output_input.size()<<"x"<<layer_output_input[0].size()<<"x"<<layer_output_input[0][0].size()<<endl;
        }
        else
        {
            cout<<"Layer not detected: "<<get<0>(inference_data[i])<<endl;
        }
    }

    //cout<<"returned loi dims: "<<layer_output_input.size()<<"x"
    //       										 <<layer_output_input[0].size()<<"x"
   	//											 <<layer_output_input[0][0].size()<<"x"
   	//											 <<layer_output_input[0][0][0].size()<<endl<<endl<<endl<<endl;
    return layer_output_input;
}

vector<vector<bitsx>> getExpectedResults(string filename)
{
    //leemos los resultados de tensorflow
    ifstream file;
    string l;
    vector<vector<bitsx>> v;
    string file_location=LIB_PATH+"/"+filename;
    file.open(file_location);

    if(file.is_open())
    {
    	while (getline(file,l))
    	{
    		v.push_back(procline<bitsx>(l));
    	}
    }
    else
    {
    	cout<<"No se pudo abrir el archivo "<<filename<<endl;
    	cout<<"Asegurate que el fichero se encuentra en "<<file_location<<endl;
    	exit(-1);
    }

    file.close();
    return v;

}

double getMSEmean(vector<vector<bitsx>> expected, vector<vector<vector<vector<bitsx>>>> actual)
{
    vector<bitsx> mses;
    bitsx acc=0;
    int ninputs=actual.size();
    int cols=actual[0][0][0].size();
    double mseAcc=0;

    bitsx diff=0;

    for(int i=0;i<ninputs;i++)
    {
        for(int j=0;j<cols;j++)
        {
        	diff= expected[i][j]-actual[i][0][0][j];
            acc += diff*diff;
        }

        acc= acc/cols;
        mseAcc+=(double)acc;
        acc=0;
    }

    return mseAcc/ninputs;
}

double mean(vector<double> times)
{
	double acc =0;

	for(int i=0;i<times.size();i++)
	{
		acc+=times[i];
	}

	return acc/times.size();
}

void test_inference(Model m, string inputs, string exp_results, cl::Context context, cl::Kernel krnl,cl::CommandQueue q, int ninputs)
{
	vector<double> times;

    vector<vector<bitsx>> data = importSharedMnistData(inputs);
    vector<vector<vector<vector<bitsx>>>> results,aux;
    vector<vector<vector<bitsx>>> aux2;
    vector<vector<bitsx>> d, expected_results  = getExpectedResults(exp_results);

	for(int i=0;i<ninputs;i++) //los inputs
    {
        d.push_back(data[i]);
        aux2.push_back(d);
        aux.push_back(aux2);

        d.clear();
        aux2.clear();
    }

	cout<<"Expected results dims: "<<aux.size()<<"x"<<expected_results[0].size()<<endl;

	cout<<"Model input dims "<<aux.size()<<"x"<<aux[0].size()<<"x"<<aux[0][0].size()<<"x"<<aux[0][0][0].size()<<endl;

    cout<<"Model inference "<<endl;

    clock_t start = clock();
    results = model_inference(m,aux,context,krnl,q); //ERR
    clock_t end = clock();

    clock_t clocksprocess = end - start;
    double timeSecs = clocksprocess / (double) CLOCKS_PER_SEC;

    //cout<<ninputs<<" inputs processed in "<<timeSecs<<" secs with an average of "<<(timeSecs/ninputs)<<" seconds per input"<<endl;

        /*cout<<"Results: "<<endl;
        for(int i=0;i<encdata.size();i++)
        {
        	for(int j=0;j<encdata[0].size();j++)
        	{
        		for(int k=0;k<encdata[0][0].size();k++)
        		{
        			for(int l=0;l<encdata[0][0][0].size();l++)
        			{
        				cout<<encdata[i][j][k][l]<<" ";
        			}
        		}
        	}
        	cout<<endl;
        }
        cout<<endl;*/

        //cout<<"results dims "<<results.size()<<"x"<<results[0].size()<<"x"<<results[0][0].size()<<"x"<<results[0][0][0].size()<<endl;

    double meanMSE= getMSEmean(expected_results,results);
    //vector<vector<bitsx>> enc;
    //enc.push_back(stats);



    if(meanMSE>1)
    {
        cout<<"MSE mean: "<<meanMSE<<endl;
        cout<<"MSE mean higher than 1, NOT CORRECT"<<endl;
        exit(-1);
    }
    else
    {
        cout<<"MSE mean: "<<meanMSE<<endl;
        cout<<"MSE mean lower than 1, CORRECT"<<endl;
    }

}


