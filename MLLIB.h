#ifndef MLLIB_H 
#define MLLIB_H
#include<vector>
#include <iostream>
#include <string>
using namespace std;

//Struct
struct Dataset {
    int N, D, K; //N:So Sample, D: So Feature, K: So Output
    vector<float> X;
    vector<float> Y;
    Dataset(int n, int d, int k);
	float& atX(int Sample, int FeaturePlus);
	const float& atX(int Sample, int FeaturePlus) const;
    float& atY(int Sample, int Output);
    const float& atY(int Sample, int Output) const;
    void addBias();
    int SizeX() const;
    int	SizeY() const;
};
struct Weight{
	int D, K;
	vector<float> W;   
    Weight(int d, int k);
    float& atW(int Feature, int Output);
	float& Bias(int Output);
	float* BiasVector();
	int SizeW() const;
	void initial(float w_ini,float b_ini);
	void show();
};
struct Loss_History {
	vector <float> Loss;
    vector <Weight> W_History;
    void save(const Weight& P,float l);
    void show();
};
struct Scaler {
    vector<float> mu;
    vector<float> sigma; 
    vector<float> inv_sigma;
    int D;
    Scaler(int d);
};
string toLowerCase(string s);

//Phep Toan Voi Vecto
float dot(const float* A,const float* B, int n);
float sum_elements_abs(const float *A, int n);
void vector_mul(const float* A,float B,float* C, int vec_size);
void vector_sub(const float* A,const float* B,float* C, int vec_size);
void vector_add(const float* A,const float* B,float* C, int vec_size);
void VxV(const float* A,const float* B,float* C, int vec_size);
void vector_fma_scalar(float A,const float* B,float* C, int vec_size);

//Phep Toan Voi Matrix
void MxM(const float* A,const float* B, float* C, int N, int K, int M);
void MTxM(const float* A, const float* B, float* C, int N, int K, int M);

//PhepToanError
void Error_Cal(const Dataset& S,const Weight& P,float *Error);
float MSE(const float *Error,int total);
float MAE(const float *Error,int total);
void Y_Pred_LN(const Dataset& S,const Weight& P,float *Y);

//GradientDescent
void Grad_MSE(const Dataset& S,float *Error,float* Grad);
void Grad_MAE(const Dataset& S,float *Error,float* Grad);
void Update_WB(Weight& P,float* Grad,float lr);

//PhepToanHuuDung
float sum_elements(const float* A,int n);
int scalar_sgn(float x);
void sgn_neglect(float *A,int n);
void sgn(float* A, int n);
void fast_fill(float* data, int n, float value);

//FeatureScaling
void feature_scaling(Dataset& S,string type,Scaler& scaler);
void rescale_weights(Weight& P, const Scaler& scaler);
//Model
void LinearRegression(Dataset S,Weight& P,float lr,int epoch,string sel,Loss_History& L);
#endif
