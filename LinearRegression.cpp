#include <iostream>
#include "MLLIB.h"
#include<fstream>	
#include <chrono>
#define lr 0.2
#define epoch 10000
using namespace std;
using namespace std::chrono;
const int N = 1000, D = 20, K = 10; // Sample - Feature - Output
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
    auto start = high_resolution_clock::now();
    P.initial(0,0);
    Scaler scaler(D);
    feature_scaling(S,fs_sel, scaler);
	LinearRegression(S, P, lr, epoch, sel, L);
	rescale_weights(P, scaler);
//	L.show();
	auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "\n--- KET QUA ---" << endl;
    cout << "Thoi gian thuc thi: " << duration.count() << " ms" << endl;
//	P.show();
	
}
