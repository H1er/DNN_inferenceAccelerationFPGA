#include <iostream>
#include <fstream>

using namespace std;

const string PYTHON_COMMAND = "python3";
const string LIB_PATH = "/home/h1er/Vitis_TFG_workspace/matrix_product/src/Vitis_Library";

/*
	compile with g++ instead of gcc
*/


void generate_model_data(string filename)
{
	int idx = filename.rfind('.');
	string extension = filename.substr(idx+1);
	string model_data_path;

	string cwd = LIB_PATH;
	string models_path=cwd+"/Import_engine/Models";

	int h5 = extension.compare("h5");
	int pb = extension.compare("pb");


	cout<<"\n---------------------LIB PATHS--------------------------\n";
	cout<<"LIB_PATH: "<<LIB_PATH<<endl;
	cout<<"MODELS_PATH: "<<models_path<<endl;
	cout<<"-----------------------------------------------\n\n";


	if(h5==0) //extension h5
	{
		cout<<"System command -> "<<(PYTHON_COMMAND+" "+cwd+"/Import_engine/import.py "+filename+" "+LIB_PATH).c_str()<<endl;
		system((PYTHON_COMMAND+" "+LIB_PATH+"/Import_engine/import.py "+filename+" "+LIB_PATH).c_str());
		//model_data_path = cwd+"/Model_data/"+filename+"/model_data.txt";
	}
	else if(pb==0)
	{
		string pbname = filename.substr(0,filename.size()-3);
		//cout<<"substr nombre: "<<pbname;
		system((PYTHON_COMMAND+" "+cwd+"/Import_engine/import.py "+filename+" "+LIB_PATH).c_str());
		//model_data_path = cwd+"/Model_data/"+pbname+"/model_data.txt";
		//si es pb, hay que cambiar la ruta, ya que se lee sin la extension
	}
	else
	{
		cout<<"Formato de modelo no soportado"<<endl;
		exit(-1);
	}
}

int main()
{
	generate_model_data("tryned.h5");

	return 0;
}
