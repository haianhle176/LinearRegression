#include <iostream>
#include "MLLIB.h"
#include<fstream>
using namespace std;
int main(){
	//Khoi tao cac tham so
	int N = 25793, D = 15, K = 1; int epoch = 1000; float lr = 0.2;	 			
	Dataset S(N,D,K);	Weight P(D,K);	Loss_History L; vector<float> VIF_arr(D);
	string sel = "mae", fs_sel = "standard";
	//Nap Du Lieu Vao Dataset
	ifstream fin("data/dln.txt");
	for (int i = 0;i < S.N;i++){
		for (int j = 0;j<S.D;j++) fin >> S.atX(i,j);
		for (int j = 0;j<S.K;j++) fin >> S.atY(i,j);
	}
    fin.close();
    //Tinh VIF
    VIF(S,VIF_arr);
	cout << "VIF: "; ShowVecto(VIF_arr);
	//Training
	StartTime();
	TrainFunction(S,P,lr,epoch,fs_sel,sel,L);
	StopTime();
	//In Ket Qua
	L.showfinal();P.show();ShowTime();
}

