#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string.h>
#include <algorithm>
#include <stdio.h>
#include <string>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>

#include "../../Vitis_Library/Import_engine/import_engine.h"
#include "../../Vitis_Library/library_constants.h"


bool shown = true;

using namespace std;



// Pinta por pantalla la matriz vect
template <typename T>
void showMat(vector<vector<T>> vect){

	//cout<<"showMat "<<endl;
	
	//cout<<"Vect size (fils): "<<vect.size();
	//cout<<"Vect size (cols): "<<vect.at(0).size();
	
	int fils=vect.size();
	int cols=vect.at(0).size();




	//cout<<"showMat data"<<endl;

	for(int i=0;i<fils;i++)
	{
		for(int j=0;j<cols;j++)
		{
			cout<<vect[i][j]<<" ";
		}
		cout<<"\n";
	}

	cout<<endl;
}


// Devuelve la fila numero index de la matriz de convolucion del kernel v
vector<bitsx> getConvMatRow(vector<vector<bitsx>> v, int index, int shift_p, int zeros_bet,int shift, int zeros_after, int zeros_b)
{
	vector<bitsx> fil;


	//se aaden los zeros before
	for(int i=0;i<zeros_b;i++)
	{
		fil.push_back(0);
	}

	for(int j=0;j<v.size();j++)
	{
		//se aade la fila
		for(int k=0;k<v.at(j).size();k++)
		{
			fil.push_back(v[j][k]);
		}

		if(j!=(v.size()-1))
		{
			//se aaden los z_between
			for(int k=0;k<zeros_bet;k++)
			{
				fil.push_back(0);
			}
		}

	}


	for(int i=0;i<zeros_after;i++)
	{
		fil.push_back(0);
	}

	return fil;

}

// Devuelve la matriz de convolucin completa del kernel v2
vector<vector<bitsx>> getConvolutionMatrix(vector<vector<bitsx>> v2,int input_fils,int input_cols)
{
	//cout<<"Fils: "<<input_fils<<endl;
	//cout<<"Cols: "<<input_cols<<endl;
	//cout<<"V2"<<endl;
	/*for(int i=0;i<v2.size();i++)
	{
		for(int j=0;j<v2.at(0).size();j++)
		{
			cout<<v2[i][j]<<" ";
		}
		cout<<endl;
	}*/

	vector<vector<bitsx>> cmat;
	int kernel_fil,kernel_col,zeros_between,zeros_after_f,shift,shift_period,convs;

	int line_size=input_fils*input_cols;

	kernel_col=v2.at(0).size();
	kernel_fil=v2.size();

	zeros_between=(input_cols-kernel_col);

	shift_period=(input_cols-kernel_col)+1;
	shift = kernel_col-1;

	convs=((input_cols-kernel_col)+1)*((input_fils-kernel_fil)+1);

	//cout<<"Convs: "<<fils<<"-"<<kernel_fil<<"+1    *    "<<cols<<"-"<<kernel_col<<"+1"<<" = "<<convs<<endl;
	for(int i=0;i<convs;i++)
	{
		int zeros_before=i+((int)(i/(shift_period))*shift);

		int kernel_line_size = ((kernel_col*kernel_fil)+(zeros_between*(kernel_fil-1)));


		zeros_after_f=line_size - (kernel_line_size+zeros_before);


		vector<bitsx> fil = getConvMatRow(v2,i,shift_period, zeros_between,shift,zeros_after_f,zeros_before);

		cmat.push_back(fil);
	}




	/*cout<<"Kernel fil: "<<kernel_fil<<endl<<"Kernel col: "<<kernel_col<<endl;
	cout<<"zeros_between: "<<zeros_between<<endl;
	cout<<"Convolution matrix: "<<endl;
	for(int i=0;i<cmat.size();i++)
	{
		cout<<"fil size: "<<cmat.at(i).size()<<" index: "<<i<<"\t\t -> ";
		for(int j=0;j<cmat.at(i).size();j++)
		{
			cout<<cmat[i][j]<<" ";
		}
		cout<<endl;
	}*/
	if(!shown)
	{
		//showMat(cmat);
		//cout<<"Input fils: "<<input_fils<<endl<<"Input cols: "<<input_cols<<endl;
		//cout<<"Output_dims: "<<input_fils-v2.size()<<"x"<<input_cols-v2.at(0).size()<<endl;
		shown=true;
	}

	return cmat;
}

// Lee y añade al modelo los pesos de una capa fully connected del flujo de entrada data
void procWeightsFC(ifstream &data, int layer_ind, string layer_typ, string activation, vector<int> dims, Model& mod)
{
	string l1;
	Layer* layfc= new FC(layer_ind, layer_typ);
	layfc->setDims(dims);
	layfc->setActivation(activation);
	vector<vector<bitsx>> w = layfc->getWeightsFC();
	int lins = 0;

	while(getline(data,l1)&&lins < layfc->getDim(0))
	{
		vector<bitsx> v = procline<bitsx>(l1);
		w.push_back(v);

		lins++;
	}
	layfc->setWeightsFC(w);
	//cout<<"FC matrix dims: "<<w.size()<<"x"<<w.at(0).size()<<endl;
	

	getline(data,l1);

	vector<bitsx> b;
	if(strstr(l1.c_str(),"Bias"))
	{
		getline(data,l1);
		b = procline<bitsx>(l1);

	}
	layfc->setBias(b);

	mod.addLayer(layfc);
}

// Lee y añade al modelo los pesos de una capa convolucional del flujo de entrada data
void procWeightsConv(ifstream &file, int layer_ind, string layer_typ, string activation, vector<int> dims, Model& mod)
{
	Layer* layconv = new Convolutional(layer_ind, layer_typ, dims);
	vector<vector<vector<vector<bitsx>>>> weits = layconv->getWeightsConv();
	string l1;

	//cout<<"Procweightsconv"<<endl;

	vector<vector<vector<bitsx>>> v1;
	vector<vector<bitsx>>v2, convMatrix;


	getline(file,l1);
	int idx = l1.rfind('.'), pos=0;
	//cout<<"Dims line: "<<l1<<endl;
	string dimstr =l1.substr(idx+1);
	//cout<<"Dimstr: "<<dimstr<<endl;
	vector<int> input_shape;
	string str_delimiter = "x";

	while ((pos = dimstr.find(str_delimiter)) != (int)string::npos)
	{
		input_shape.push_back(stoi(dimstr.substr(0, pos)));
	    dimstr.erase(0, pos+str_delimiter.length());
	}

	input_shape.push_back(stoi(dimstr.substr(0, dimstr.length())));

	//cout<<"input_shape "<<input_shape[0]<<"x"<<input_shape[1]<<endl;
	//cout<<"dims size: "<<dims.size()<<endl;

	for(int i=0;i<dims.at(2);i++) //por cada canal
	{
		

		for(int j=0;j<dims.at(3);j++) //por cada filtro
		{
		
//		cout<<"chekcpoint 2"<<endl;
			for(int k=0;k<dims.at(0);k++)//por cada linea del filtro
			{
//	cout<<"checkpoint 3"<<endl;
				getline(file,l1);
				vector<bitsx> v= procline<bitsx>(l1);
				v2.push_back(v);
			}
	
//cout<<"convmat"<<endl;
			// EN ESTE PUNTO V2 TIENE EL FILTRO, POR LO QUE SE CONSTRUYE LA MATRIZ USANDO
			// V2, Y SE GUARDA ESTA LA MATRIZ USANDO V2, Y SE GUARDA ESTA EN VEZ DE V2
			convMatrix = getConvolutionMatrix(v2,input_shape[0],input_shape[1]);

		//	showMat(convMatrix);

			v1.push_back(convMatrix); //push_back a convmatrix
			v2.clear();
		}
		weits.push_back(v1);
		v1.clear();

	}

	//cout<<"input_shape[0]"<<input_shape[0]<<endl<<"input_shape[1]"<<input_shape[1]<<endl;
	//cout<<"Convmatrix dims: "<<convMatrix.size()<<"x"<<convMatrix.at(0).size()<<endl;
	layconv->setWeightsConv(weits);

	//cout<<"Tremendas dims: "<<weits.size()<<"x"<<weits.at(0).size()<<"x"<<weits.at(0).at(0).size()<<"x"<<weits.at(0).at(0).at(0).size()<<endl;

	getline(file,l1);

	vector<bitsx> b;
	if(strstr(l1.c_str(),"Bias"))
	{
		getline(file,l1);
		b = procline<bitsx>(l1);

	}
	layconv->setBias(b);
	layconv->setActivation(activation);

	mod.addLayer(layconv);
}

//procesa una linea de fichero de datos separados por ', ' y los devuelve en un vector
template <typename T>
vector<T> procline(string s)
{
	int cont=0;

	vector<T> row;

	string space_delimiter = ", ";

	int p=0;
	while ((p = s.find(space_delimiter)) != (int) string::npos)
	{
		row.push_back(stod(s.substr(0, p)));
	    s.erase(0, p + space_delimiter.length());
	    cont++;
	}

	return row;
	//lay.setWeights(weights);
}

//importa los datos del dataset MNIST del archivo filename
vector<vector<bitsx>> importSharedMnistData(string filename)
{
	string file_location = LIB_PATH+"/"+filename;
	ifstream mnistData (file_location);
	string line;
	vector<vector<bitsx>> sharedMnistdata;

	cout<<"file location: "<<file_location<<endl;

	if (mnistData.is_open())
	{
		vector<bitsx> mnistimage;
		cout<<"Open success: "<<filename<<endl;
	    while ( getline(mnistData,line) )
		{
			mnistimage=procline<bitsx>(line);
			sharedMnistdata.push_back(mnistimage);
		}
		mnistimage.clear();
	}
	else
	{
		cout<<endl<<"No se pudo abrir el archivo "<<filename<<endl;
		cout<<"-----Asegurate que este en "<<file_location<<endl;
	}


	return sharedMnistdata;
}



bool exist_file (string name)
{
    ifstream file(name.c_str());

    return file.good();
}


// Importa los datos del modelo guardado en el archivo 'filename' solo soporta formato .pb y .h5
void import_model(string filename, Model& model)
{
	int idx = filename.rfind('.');
	string extension = filename.substr(idx+1);
	string model_data_path;

	string models_path=LIB_PATH+"/Import_engine/Models";

	int h5 = extension.compare("h5");
	int pb = extension.compare("pb");

	cout<<"---------------------LIB PATHS--------------------------\n";
	cout<<"LIB_PATH: "<<LIB_PATH<<endl;
	cout<<"MODELS_PATH: "<<models_path<<endl;
	cout<<"-----------------------------------------------\n";

	if(h5==0) //extension h5
	{
		model_data_path = LIB_PATH+"/Import_engine/Models/Model_data/"+filename+"/model_data.txt";
	}
	else if(pb==0)
	{
		string pbname = filename.substr(0,filename.size()-3);
		model_data_path = LIB_PATH+"/Import_engine/Models/Model_data/"+pbname+"/model_data.txt";
		//si es pb, hay que cambiar la ruta, ya que se lee sin la extensi�n
	}
	else
	{
		cout<<"Error al cargar el modelo, formato no soportado"<<endl;
		exit(-1);
	}

	cout<<"Intentando abrir "<<model_data_path<<endl;
	ifstream data (model_data_path);
	string l1, typ, activation;
	int lay_idx;

	if (data.is_open())
	{
		cout<<"Open success: "<<filename<<endl;
	    while ( getline (data,l1) )
		{
	    	if(strstr(l1.c_str(),"Layer"))//se guarda el numero de la layer
	    	{
	    		int idx = l1.rfind('.');
	    		string snum =l1.substr(idx+1);
	    		int num = stoi(snum);

	    		lay_idx = num;
	    	}
	    	else if (strstr(l1.c_str(),"Type")) //se guarda el tipo
	    	{
	    		int idx = l1.rfind('.');
	    		string type =l1.substr(idx+1);
	    		//cout<<"Tipo: "<<type<<endl;
	    		typ=type;
	    	}
	    	else if (strstr(l1.c_str(),"Activation")) //se guarda la funcion de activacion
	    	{
	    		int idx = l1.rfind('.');
	    		activation =l1.substr(idx+1);
	    	}
	    	else if (strstr(l1.c_str(),"weights_dim")) //dimensiones de la matriz de pesos
	    	{
	    		int idx = l1.rfind('.'), pos=0;
	    		string dimstr =l1.substr(idx+1);

	    		vector<int> dims;

	    		string str_delimiter = "x";

	    		while ((pos = dimstr.find(str_delimiter)) != (int)string::npos)
	    		{
	    			dims.push_back(stoi(dimstr.substr(0, pos)));
   				    dimstr.erase(0, pos+str_delimiter.length());
   				}

	    		dims.push_back(stoi(dimstr.substr(0, dimstr.length())));
				
				//cout<<endl;
	    		if(strstr(typ.c_str(),"dense") && dims.size()==2)
	    		{
	    			//cout<<"type FC: "<<typ<<endl;
	    			procWeightsFC(data, lay_idx, typ, activation, dims, model);
	    		}
	    		else if(strstr(typ.c_str(),"conv") && dims.size()==4)
	    		{
	    			//cout<<"type Conv: "<<typ<<endl;
	    			procWeightsConv(data, lay_idx, typ, activation, dims, model);
	    		}
	    		else
	    		{
	    			//cout<<"Type func: "<<typ<<endl;
	    			Layer *l = new FC(lay_idx,typ);
	    			model.addLayer(l);
	    		}
	    	}
	    	else if(strstr(l1.c_str(),"ENDLAYER"))//indicador fin de layer
	    	{
				//cout<<"Added"<<endl<<endl;
				//shown=false;
	    		//model.addLayer(l); //en funcion del tipo se guarda como fc o como conv (TBD)
	    	}
		}
		data.close();
	}
	else
	{
		cout<<"Error al cargar el modelo "<<filename<<" Has ejecutado el importer para este modelo?"<<endl;
		exit(-1);
	}
	cout<<"Model "<<filename<<" imported succesfully!"<<endl<<endl;
	data.close();

	return;
}

//carga 'nimages' imagenes del dataset de test de mnist en el vector 'data'
void load_mnist_test(vector<vector<bitsx>> &data, int nimages)
{
	string l1;
	string space_delimiter = ", ";
	ifstream mnist ("mnist_data.txt");
	int fila=0,index=0;
	int ims=0;

	data = vector<vector<bitsx>>(nimages);

	for(int i=0;i<nimages;i++)
	{
		data.at(i)= vector<bitsx>(784);
	}
	cout<<endl;
	if (mnist.is_open())
	{
		while (ims<nimages && getline (mnist,l1)  )
		{


			if(strstr(l1.c_str(),"Ind"))//Si contiene ind es que es el comienzo de una foto nueva
			{
				int pos = l1.rfind('.');
				string snum =l1.substr(pos+1);
				index = stoi(snum);

				ims++;

				fila=0;
			}
			else //se lee una l�nea
			{

				string space_delimiter = ", ";

				int pos=0,cont=0;
				while ((pos = l1.find(space_delimiter)) != (int)string::npos)
				{

					data[index][((fila*28)+cont)]=stod(l1.substr(0, pos));

				    l1.erase(0, pos + space_delimiter.length());

				    cont++;
				}

				fila++;
				cont=0;
			}
		}
	}
	else
	{
		cout<<"No se pudo abrir el archivo mnist_data.txt";
	}

	mnist.close();
}

//shows the image in the position 'index'
void show_mnist_image(vector<vector<bitsx>> data, int index, bool beauty)
{
	for(int i=0;i<28;i++)
	{
		for(int j=0;j<28;j++)
		{
			if(beauty && data[index][((i*28)+j)] != 0)
			{
				cout<<"  ";
			}
			else
			{
				cout<<data[index][((i*28)+j)]<<" ";
			}
		}
		cout<<endl;
	}
	cout<<endl;
}

//stores the values of the array passed as arg1 into the vector passed as arg 2
void tovect(bitsx** arr,vector<vector<bitsx>> &v)
{
	for(int i=0;i<(int)v.size();i++)
	{
		vector<bitsx>fil;
		for(int j=0;j<(int)v.at(0).size();j++)
		{
			fil.push_back(arr[i][j]);
		}
		v[i]=fil;
	}
}

//return an array with the values of the 2d vector argument
bitsx** toarr(vector<vector<bitsx>> v)
{
	bitsx** w;

	int fil = (int)v.size();
	int col = (int)v.at(0).size();


	w= new bitsx*[fil];

	for (int i = 0; i < fil; ++i) {
	  w[i] = new bitsx[col];
	  for(int j=0;j<col;j++)
	  {
		  w[i][j]=0;
	  }

	}

	for(int i=0;i<fil;i++)
	{
		for(int j=0;j<col;j++)
		{
			w[i][j]= v[i][j];
		}
	}

	return w;
}




