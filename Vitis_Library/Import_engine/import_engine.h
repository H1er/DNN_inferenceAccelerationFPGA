#ifndef import_engine_H
#define import_engine_H

#include <iostream>
#include <cstring>
#include <string>
#include <vector>
#include <tuple>
#include <ap_fixed.h>

#include "../../Vitis_Library/library_constants.h"


using namespace std;

// Clase padre layer en la que se especifica las cosas que debe tener una layer
class Layer {

public:

	//COMMON
	virtual void setBias(vector<bitsx> b)=0;
	virtual void setDims(vector<int> d)=0;
	virtual int getIdx()=0;
	virtual string getType()=0;
	virtual int getDim(int i)=0;
	virtual vector<bitsx> getBias()=0;
	virtual string getActivation()=0;
	virtual void setActivation(string s)=0;
	virtual vector<int> getDims()=0;


	//FC
	virtual vector<vector<bitsx>> getWeightsFC()=0;
	virtual void setWeightsFC(vector<vector<bitsx>> we)=0;

	//CONV
	virtual vector<vector<vector<vector<bitsx>>>> getWeightsConv()=0;
	virtual void setWeightsConv(vector<vector<vector<vector<bitsx>>>> we)=0;

};

// Clase especifica fully connected en la que se definen metodos y elementos especificos de este tipo de capa
class FC : public Layer {
private:
	vector<int> dims;
	int input_shape[2];
	string activation;
	vector<bitsx>bias;
	vector<vector<bitsx>>weights;
	int index;
	string type;

public:
	FC(int idx, string typ){
		index=idx;
		type=typ;

	}

	~FC() {

	}

	void setActivation(string s)
	{
		activation=s;
	}


	string getActivation()
	{
		return activation;
	}

	vector<vector<bitsx>> getWeightsFC(){
		return weights;
	}
	void setWeightsFC(vector<vector<bitsx>> we){
		weights=we;
	}
	void setBias(vector<bitsx> b){
		bias=b;
	}
	void setDims(vector<int> d){
			dims = d;
	}

	int getIdx(){
		return index;
	}

	string getType(){
		return type;
	}

	int getDim(int i){
		return dims.at(i);
	}

	vector<int> getDims()
	{
		return dims;
	}

	vector<bitsx> getBias(){
		return bias;
	}

	vector<vector<vector<vector<bitsx>>>> getWeightsConv(){return vector<vector<vector<vector<bitsx>>>>();}
	void setWeightsConv(vector<vector<vector<vector<bitsx>>>> we){}
	
};

// Clase especifica convolucional en la que se definen metodos y elementos especificos de este tipo de capa
class Convolutional : public Layer {
private:
	vector<int> dims;
	int input_shape[2];
	string activation;
	vector<bitsx>bias;
	vector<vector<vector<vector<bitsx>>>> weights;
	int index;
	string type;

public:
	Convolutional(int idx, string typ,vector<int> d){
		index=idx;
		type=typ;
		dims =d;
	}

	~Convolutional() {

	}

	void setActivation(string s)
	{
		activation=s;
	}

	string getActivation()
	{
		return activation;
	}

	vector<int> getDims()
	{
		return dims;
	}

	vector<vector<vector<vector<bitsx>>>> getWeightsConv(){
		return weights;
	}
	void setWeightsConv(vector<vector<vector<vector<bitsx>>>> we){
		weights=we;
	}
	void setBias(vector<bitsx> b){
		bias=b;
	}
	void setDims(vector<int> d){
			dims = d;
	}

	int getIdx()
	{
		return index;
	}

	string getType()
	{
		return type;
	}

	int getDim(int i)
	{
		return dims.at(i);
	}

	vector<bitsx> getBias(){
		return bias;
	}

	vector<vector<bitsx>> getWeightsFC(){return vector<vector<bitsx>>();}
	void setWeightsFC(vector<vector<bitsx>> we){}
};

// Clase en la que se especifica de que esta compuesto un modelo 
class Model {
	vector<Layer*> layers;

public:
	Model()
	{

	}

	Layer* getLayer(int i)
	{
		return layers[i];
	}

	void addLayer(Layer* l)
	{
		//cout<<"Entra al addlayer";
		layers.push_back(l);

	}

	vector<tuple<string,string,vector<bitsx>,vector<vector<vector<vector<bitsx>>>>>> getInferenceData()
	{
		//tipo,activation,bias,weights
		vector<tuple<string,string,vector<bitsx>,vector<vector<vector<vector<bitsx>>>>>> model_inference_data;
		vector<vector<vector<vector<bitsx>>>> convWeights;
		vector<vector<vector<vector<bitsx>>>> auxFCWeights;
		vector<vector<vector<bitsx>>> aux;
		vector<vector<bitsx>> fcWeights;


		for(int i=0;i<layers.size();i++)
		{
			string type = layers[i]->getType();
			if(strstr(type.c_str(),"dense"))
			{
				//cout<<"Type FC "<<i<<" "<<type<<endl;
				fcWeights=layers[i]->getWeightsFC();

				vector<bitsx> bias = layers[i]->getBias();
				string act = layers[i]->getActivation();

				//cout<<"Dims: "<<fcWeights.size()<<"x"<<fcWeights.at(0).size()<<endl;

				//hay que encapsular la matriz para que entre todo junto
				aux.push_back(fcWeights);
				auxFCWeights.push_back(aux);

				//cout<<"Dims: "<<auxFCWeights.size()<<"x"<<auxFCWeights[0].size()<<"x"<<auxFCWeights[0][0].size()<<"x"<<auxFCWeights[0][0][0].size()<<endl;
				model_inference_data.push_back(tuple<string,string,vector<bitsx>,vector<vector<vector<vector<bitsx>>>>>(type,act,bias,auxFCWeights));

				auxFCWeights.clear();
				aux.clear();
			}
			else if(strstr(type.c_str(),"conv2d"))
			{
				//cout<<"Type Conv "<<i<<" "<<type<<endl;
				convWeights=layers[i]->getWeightsConv();

				vector<bitsx> bias = layers[i]->getBias();
				string act = layers[i]->getActivation();
				//cout<<"Activation "<<act<<endl;


				//cout<<"Dims: "<<convWeights.size()<<"x"<<convWeights.at(0).size()<<"x"<<convWeights.at(0).at(0).size()<<"x"<<convWeights.at(0).at(0).at(0).size()<<endl;
				model_inference_data.push_back(tuple<string,string,vector<bitsx>,vector<vector<vector<vector<bitsx>>>>>(type,act,bias,convWeights));
			}
			else if(!strstr(type.c_str(),"conv2d")&&!strstr(type.c_str(),"dense")&&(strstr(type.c_str(),"flatten")||strstr(type.c_str(),"max_pooling2d")||strstr(type.c_str(),"average_pooling2d")))
			{
				//cout<<"Type Func "<<i<<" "<<type<<endl;
				model_inference_data.push_back(tuple<string,string,vector<bitsx>,vector<vector<vector<vector<bitsx>>>>>(type,"",NULL,NULL));
			}
			else
			{
				cout<<"Layer not supported "<<i<<" "<<type<<endl;
				exit(-1);
			}
			//cout<<"Layer "<<i+1<<" done"<<endl;

		}
		cout<<endl;

		return model_inference_data;

	}

	vector<vector<int>> getLayerWeightsSizes()
	{
		vector<vector<int>> totalSizes;
		vector<int> sizes;

		for(int i=0;i<layers.size();i++)
		{
			Layer *l = getLayer(i);
			string typ = l->getType();

			if(strstr(typ.c_str(),"flatten"))//se guarda el numero de la layer
			{
				sizes.push_back(0);
				sizes.push_back(0);
			}
			else if(strstr(typ.c_str(),"conv2d"))//se guarda el numero de la layer
			{
				sizes.push_back(l->getWeightsConv()[0][0].size());
				sizes.push_back(l->getWeightsConv()[0][0][0].size());
			}
			else if(strstr(typ.c_str(),"dense"))
			{
				sizes=l->getDims();

			}
			else
			{
				cout<<"Layer not supported"<<endl;
				exit(-1);
			}


			totalSizes.push_back(sizes);
			sizes.clear();
		}

		return totalSizes;
	}

	vector<vector<vector<vector<bitsx>>>> getWeights();
};

// Importa los datos del dataset MNIST de filename (Se usa para las pruebas con tf )
vector<vector<bitsx>> importSharedMnistData(string filename);

// Importa los datos del modelo guardado en el archivo 'filename' solo soporta formato .pb y .h5
void import_model(string filename, Model &mod);

// Pinta por pantalla la matriz vect
void showMat(vector<vector<bitsx>> a);

void show_mnist_image(vector<vector<bitsx>> data, int index, bool beauty);

void load_mnist_test(vector<vector<bitsx>> &data, int nimages);

//procesa una linea de fichero de datos separados por ', ' y los devuelve en un vector
template<typename T>
vector<T> procline(string s);



#endif
