#include <iostream>
#include "MLLIB.h"
#include<fstream>	
#define lr 0.2
#define epoch 10000
using namespace std;
const int N = 1000, D = 20, K = 10;
int main(){
	Dataset S(N,D,K);
	Weight P(D,K);
	Loss_History L;
	string sel, fs_sel;
	float t;
	ifstream fin("dataset.txt");
	cin >> sel >> fs_sel;;
	for (int i = 0;i < S.N;i++){
		for (int j = 0;j<S.D;j++){
			fin >> t;
			S.atX(i,j) = t;
		}
		for (int j = 0;j<S.K;j++){
			fin >> t;
			S.atY(i,j) = t;
		}
	}
    fin.close();
    P.initial(0,0);
    Scaler scaler(D);
    feature_scaling(S,fs_sel, scaler);
	LinearRegression(S, P, lr, epoch, sel, L);
	rescale_weights(P, scaler);
	L.show();
	P.show();
}
