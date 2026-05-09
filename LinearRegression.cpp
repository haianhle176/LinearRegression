#include <iostream>
#include "MLLIB.h"
#include<fstream>	
#include <chrono>
#define lr 0.5
#define epoch 400
using namespace std;
using namespace std::chrono;
const int N = 500, D = 10, K = 1; // Sample - Feature - Output

int main(){
	Dataset S(N,D,K);
	Weight P(D,K); P.initial(0,0);
	Loss_History L;
	string sel, fs_sel;
	float t;
	ifstream fin("dataset.txt");
	cin >> sel >> fs_sel;;
	for (int i = 0;i < S.N;i++){
		for (int j = 0;j<S.D;j++){fin >> S.atX(i,j);}
		for (int j = 0;j<S.K;j++){fin >> S.atY(i,j);}
	}
    fin.close();
    vector<float> VIF_arr(D);
	VIF_Cal(S,VIF_arr.data());
	cout << "VIF: ";
	for (float x : VIF_arr){
		cout << x << " ";
	}
    auto start = high_resolution_clock::now();
    Scaler scaler(D);
  	feature_scaling(S,fs_sel, scaler);
	LinearRegression(S, P, lr, epoch, sel, L);
	rescale_weights(P, scaler);
	//L.show();
	auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "\n\n--- KET QUA ---" << endl;
    cout << "Thoi gian thuc thi: " << duration.count() << " ms" << endl;
	P.show();
}
