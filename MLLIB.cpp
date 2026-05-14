#include "MLLIB.h"
#include<cmath>
#include <immintrin.h>
#include <omp.h>
#include <random>

Dataset::Dataset(int n, int d, int k) : N(n), D(d),K(k), X(n * (d + 1)), Y(n * k) {addBias();}
float& Dataset::atX(int Sample, int FeaturePlus) {return X[Sample * (D + 1) + FeaturePlus];}
const float& Dataset::atX(int Sample, int FeaturePlus) const {return X[Sample * (D + 1) + FeaturePlus];}
float& Dataset::atY(int Sample, int Output) {return Y[Sample * K + Output];}
const float& Dataset::atY(int Sample, int Output) const {return Y[Sample * K + Output];}
void Dataset::addBias() {for (int i = 0; i < N; i++) {atX(i, D) = 1.0f;}}
int Dataset::SizeX() const {return N * (D + 1);}
int Dataset::SizeY() const {return N * K;}
void Dataset::Reset() {fast_fill_scalar(X.data(),0,N * (D + 1));fast_fill_scalar(Y.data(),0,N * K);addBias();}

Weight::Weight(int d, int k) : D(d), K(k), W((d+1) * k) {}
float& Weight::atW(int Feature, int Output) {return W[Feature * K + Output];}
float& Weight::Bias(int Output){return W[D * K + Output];}
float* Weight::BiasVector(){return &W[D * K];}
int Weight::SizeW() const {return (D + 1) * K;}
void Weight::initial(float w_ini,float b_ini){fast_fill_scalar(W.data(),w_ini,D * K);fast_fill_scalar(W.data() + D * K,b_ini,K);}
void Weight::initial(const float* w_ini,const float* b_ini){fast_fill(W.data(),w_ini,D * K);fast_fill(W.data() + D * K,b_ini,K);}
void Weight::show(){
	cout << "\nW final : ";
	for (int j = 0;j < SizeW();j++){
		cout << W[j] << " ";
	}
}
void Loss_History::save(const Weight& P,float l){W_History.push_back(P);Loss.push_back(l);}
void Loss_History::show(){
	for (int i = 0;i<Loss.size();i++){
		cout << "\nLoss " << i <<": " << Loss[i];
		cout << " W: ";
		for (int j = 0;j < W_History[0].SizeW();j++){
			cout << W_History[i].W[j] << " ";
		}
	}
	cout<<"\nLoss final: "<<Loss[Loss.size() - 1];
}
void Loss_History::showfinal(){cout<<"\nLoss final: "<<Loss[Loss.size() - 1];}

Scaler::Scaler(int d) : D(d), mu(d, 0.0f), sigma(d, 0.0f), inv_sigma(d, 1.0f) {}

string toLowerCase(string s) {
    for (char &c : s) {
        c = tolower(c);
    }
    return s;
}

float dot(const float* A,const float* B,int n){
	int i = 0;float result = 0;
	__m256 sum1 = _mm256_setzero_ps();
	__m256 sum2 = _mm256_setzero_ps();
	__m256 sum3 = _mm256_setzero_ps();
	__m256 sum4 = _mm256_setzero_ps();
	for(;i<= n - 32;i+=32){
		sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(A + i),_mm256_loadu_ps(B + i),sum1);
		sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(A + i + 8),_mm256_loadu_ps(B + i + 8),sum2);
		sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(A + i + 16),_mm256_loadu_ps(B + i + 16),sum3);
		sum4 = _mm256_fmadd_ps(_mm256_loadu_ps(A + i + 24),_mm256_loadu_ps(B + i + 24),sum4);
	}
	__m256 sum = _mm256_add_ps(_mm256_add_ps(sum1, sum2),_mm256_add_ps(sum3, sum4));
    for (; i <= n - 8; i += 8) {
        sum = _mm256_fmadd_ps(_mm256_loadu_ps(A + i),_mm256_loadu_ps(B + i), sum);
    }
	__m128 low  = _mm256_castps256_ps128(sum);
    __m128 high = _mm256_extractf128_ps(sum, 1);
    __m128 s = _mm_add_ps(low, high);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    result = _mm_cvtss_f32(s);
    for (;i<n;i++){
    	result += A[i]*B[i];	
	}
    return result;
}
float Dist(const float* A,const float* B, int n){
	int i = 0;float result = 0;
	__m256 sum1 = _mm256_setzero_ps();
	__m256 sum2 = _mm256_setzero_ps();
	__m256 sum3 = _mm256_setzero_ps();
	__m256 sum4 = _mm256_setzero_ps();
	__m256 sub1, sub2, sub3, sub4;
	for(;i<= n - 32;i+=32){
		sub1 = _mm256_sub_ps(_mm256_loadu_ps(A + i),_mm256_loadu_ps(B + i));
		sub2 = _mm256_sub_ps(_mm256_loadu_ps(A + i + 8),_mm256_loadu_ps(B + i + 8));
		sub3 = _mm256_sub_ps(_mm256_loadu_ps(A + i + 16),_mm256_loadu_ps(B + i + 16));
		sub4 = _mm256_sub_ps(_mm256_loadu_ps(A + i + 24),_mm256_loadu_ps(B + i + 24));
		sum1 = _mm256_fmadd_ps(sub1,sub1,sum1);
		sum2 = _mm256_fmadd_ps(sub2,sub2,sum2);
		sum3 = _mm256_fmadd_ps(sub3,sub3,sum3);
		sum4 = _mm256_fmadd_ps(sub4,sub4,sum4);
	}
	sum1 = _mm256_add_ps(_mm256_add_ps(sum1, sum2),_mm256_add_ps(sum3, sum4));
    for (; i <= n - 8; i += 8) {
        sub1 = _mm256_sub_ps(_mm256_loadu_ps(A + i),_mm256_loadu_ps(B + i));
        sum1 = _mm256_fmadd_ps(sub1,sub1,sum1);
    }
	__m128 low  = _mm256_castps256_ps128(sum1);
    __m128 high = _mm256_extractf128_ps(sum1, 1);
    __m128 s = _mm_add_ps(low, high);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    result = _mm_cvtss_f32(s);
    for (;i<n;i++){
    	result += (A[i]-B[i])*(A[i]-B[i]);	
	}
    return result;
}
float Dist(const float* A,float B, int n){
	int i = 0;float result = 0;
	__m256 sum1 = _mm256_setzero_ps();
	__m256 sum2 = _mm256_setzero_ps();
	__m256 sum3 = _mm256_setzero_ps();
	__m256 sum4 = _mm256_setzero_ps();
	__m256 sub1, sub2, sub3, sub4;
	__m256 one = _mm256_set1_ps(B);
	for(;i<= n - 32;i+=32){
		sub1 = _mm256_sub_ps(_mm256_loadu_ps(A + i),one);
		sub2 = _mm256_sub_ps(_mm256_loadu_ps(A + i + 8),one);
		sub3 = _mm256_sub_ps(_mm256_loadu_ps(A + i + 16),one);
		sub4 = _mm256_sub_ps(_mm256_loadu_ps(A + i + 24),one);
		sum1 = _mm256_fmadd_ps(sub1,sub1,sum1);
		sum2 = _mm256_fmadd_ps(sub2,sub2,sum2);
		sum3 = _mm256_fmadd_ps(sub3,sub3,sum3);
		sum4 = _mm256_fmadd_ps(sub4,sub4,sum4);
	}
	sum1 = _mm256_add_ps(_mm256_add_ps(sum1, sum2),_mm256_add_ps(sum3, sum4));
    for (; i <= n - 8; i += 8) {
        sub1 = _mm256_sub_ps(_mm256_loadu_ps(A + i),one);
        sum1 = _mm256_fmadd_ps(sub1,sub1,sum1);
    }
	__m128 low  = _mm256_castps256_ps128(sum1);
    __m128 high = _mm256_extractf128_ps(sum1, 1);
    __m128 s = _mm_add_ps(low, high);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    result = _mm_cvtss_f32(s);
    for (;i<n;i++){
    	result += (A[i]-B)*(A[i]-B);	
	}
    return result;
}
float sum_elements_abs(const float *A,int n) {
    int i = 0;
    __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 sum_v = _mm256_setzero_ps();

    for (; i <= n - 8; i += 8) {
        __m256 data = _mm256_loadu_ps(A + i);
        __m256 abs_data = _mm256_and_ps(data, abs_mask);
        sum_v = _mm256_add_ps(sum_v, abs_data);
    }
    float temp[8];
    _mm256_storeu_ps(temp, sum_v);
    float result = 0;
    for(int j=0; j<8; j++) result += temp[j];
    for (; i < n; i++) {
        result += fabsf(A[i]);
    }
    return result;
}
void vector_mul(const float* A,float B,float* C,int vec_size){
	int i = 0;
	__m256 sum1, sum2, sum3, sum4;
	__m256 scalar_vec = _mm256_set1_ps(B);
	for(;i<= vec_size - 32;i+=32){
		sum1 = _mm256_mul_ps(_mm256_loadu_ps(A + i),scalar_vec);
		sum2 = _mm256_mul_ps(_mm256_loadu_ps(A + i + 8),scalar_vec);
		sum3 = _mm256_mul_ps(_mm256_loadu_ps(A + i + 16),scalar_vec);
		sum4 = _mm256_mul_ps(_mm256_loadu_ps(A + i + 24),scalar_vec);
		_mm256_storeu_ps(C + i,sum1);
		_mm256_storeu_ps(C + i + 8,sum2);
		_mm256_storeu_ps(C + i + 16,sum3);
		_mm256_storeu_ps(C + i + 24,sum4);
	}
    for (; i <= vec_size - 8; i += 8) {
        sum1 = _mm256_mul_ps(_mm256_loadu_ps(A + i),scalar_vec);
        _mm256_storeu_ps(C + i,sum1);
    }
    for (;i<vec_size;i++){
    	C[i] = A[i] * B;	
	}
}
void vector_sub(const float* A,const float* B,float* C,int vec_size){
	int i = 0;
	__m256 sum1, sum2, sum3, sum4;
	for(;i<= vec_size - 32;i+=32){
		sum1 = _mm256_sub_ps(_mm256_loadu_ps(A + i),_mm256_loadu_ps(B + i));
		sum2 = _mm256_sub_ps(_mm256_loadu_ps(A + i + 8),_mm256_loadu_ps(B + i + 8));
		sum3 = _mm256_sub_ps(_mm256_loadu_ps(A + i + 16),_mm256_loadu_ps(B + i + 16));
		sum4 = _mm256_sub_ps(_mm256_loadu_ps(A + i + 24),_mm256_loadu_ps(B + i + 24));
		_mm256_storeu_ps(C + i,sum1);
		_mm256_storeu_ps(C + i + 8,sum2);
		_mm256_storeu_ps(C + i + 16,sum3);
		_mm256_storeu_ps(C + i + 24,sum4);
	}
    for (; i <= vec_size - 8; i += 8) {
        sum1 = _mm256_sub_ps(_mm256_loadu_ps(A + i),_mm256_loadu_ps(B + i));
        _mm256_storeu_ps(C + i,sum1);
    }
    for (;i<vec_size;i++){
    	C[i] = A[i] - B[i];	
	}
}
void vector_add(const float* A,const float* B,float* C,int vec_size){
	int i = 0;
	__m256 sum1, sum2, sum3, sum4;
	for(;i<= vec_size - 32;i+=32){
		sum1 = _mm256_add_ps(_mm256_loadu_ps(A + i),_mm256_loadu_ps(B + i));
		sum2 = _mm256_add_ps(_mm256_loadu_ps(A + i + 8),_mm256_loadu_ps(B + i + 8));
		sum3 = _mm256_add_ps(_mm256_loadu_ps(A + i + 16),_mm256_loadu_ps(B + i + 16));
		sum4 = _mm256_add_ps(_mm256_loadu_ps(A + i + 24),_mm256_loadu_ps(B + i + 24));
		_mm256_storeu_ps(C + i,sum1);
		_mm256_storeu_ps(C + i + 8,sum2);
		_mm256_storeu_ps(C + i + 16,sum3);
		_mm256_storeu_ps(C + i + 24,sum4);
	}
    for (; i <= vec_size - 8; i += 8) {
        sum1 = _mm256_add_ps(_mm256_loadu_ps(A + i),_mm256_loadu_ps(B + i));
        _mm256_storeu_ps(C + i,sum1);
    }
    for (;i<vec_size;i++){
    	C[i] = A[i] + B[i];	
	}
}
void VxV(const float* A,const float* B,float* C,int vec_size){
	int i = 0;
	__m256 sum1, sum2, sum3, sum4;
	for(;i<= vec_size - 32;i+=32){
		sum1 = _mm256_mul_ps(_mm256_loadu_ps(A + i),_mm256_loadu_ps(B + i));
		sum2 = _mm256_mul_ps(_mm256_loadu_ps(A + i + 8),_mm256_loadu_ps(B + i + 8));
		sum3 = _mm256_mul_ps(_mm256_loadu_ps(A + i + 16),_mm256_loadu_ps(B + i + 16));
		sum4 = _mm256_mul_ps(_mm256_loadu_ps(A + i + 24),_mm256_loadu_ps(B + i + 24));
		_mm256_storeu_ps(C + i,sum1);
		_mm256_storeu_ps(C + i + 8,sum2);
		_mm256_storeu_ps(C + i + 16,sum3);
		_mm256_storeu_ps(C + i + 24,sum4);
	}
    for (; i <= vec_size - 8; i += 8) {
        sum1 = _mm256_mul_ps(_mm256_loadu_ps(A + i),_mm256_loadu_ps(B + i));
        _mm256_storeu_ps(C + i,sum1);
    }
    for (;i<vec_size;i++){
    	C[i] = A[i] * B[i];	
	}
}
void vector_fma_scalar(float A,const float* B,float* C,int vec_size){
	int i = 0;
	__m256 a_vec = _mm256_set1_ps(A);
	for(;i<= vec_size - 32;i+=32){
		__m256 sum1 = _mm256_loadu_ps(C + i);
        __m256 sum2 = _mm256_loadu_ps(C + i + 8);
        __m256 sum3 = _mm256_loadu_ps(C + i + 16);
        __m256 sum4 = _mm256_loadu_ps(C + i + 24);
		sum1 = _mm256_fmadd_ps(a_vec,_mm256_loadu_ps(B + i),sum1);
		sum2 = _mm256_fmadd_ps(a_vec,_mm256_loadu_ps(B + i + 8),sum2);
		sum3 = _mm256_fmadd_ps(a_vec,_mm256_loadu_ps(B + i + 16),sum3);
		sum4 = _mm256_fmadd_ps(a_vec,_mm256_loadu_ps(B + i + 24),sum4);
		_mm256_storeu_ps(C + i,sum1);
		_mm256_storeu_ps(C + i + 8,sum2);
		_mm256_storeu_ps(C + i + 16,sum3);
		_mm256_storeu_ps(C + i + 24,sum4);
	}
    for (; i <= vec_size - 8; i += 8) {
    	__m256 sum1 = _mm256_loadu_ps(C + i);
        sum1 = _mm256_fmadd_ps(a_vec,_mm256_loadu_ps(B + i),sum1);
        _mm256_storeu_ps(C + i,sum1);
    }
    for (;i<vec_size;i++){
    	C[i] = A * B[i] + C[i];	
	}
}
void MxM(const float* A, const float* B, float* C, int N, int K,int M) {
    int BS = 32;
    int M32 = (M / 32) * 32;
    #pragma omp parallel for 
    for (int ii = 0; ii < N; ii += BS) {
        for (int jj = 0; jj < M32; jj += BS) {
            for (int i = ii; i < ii + BS && i < N; i++) {
				__m256 c1 = _mm256_setzero_ps();
				__m256 c2 = _mm256_setzero_ps();
				__m256 c3 = _mm256_setzero_ps();
				__m256 c4 = _mm256_setzero_ps();
                for (int kk = 0; kk < K; kk++) {
                    __m256 a_vec = _mm256_set1_ps(A[i * K + kk]);
                    c1 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&B[kk * M + jj]),      c1);
                    c2 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&B[kk * M + jj + 8]),  c2);
                    c3 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&B[kk * M + jj + 16]), c3);
                    c4 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&B[kk * M + jj + 24]), c4);
                }
                _mm256_storeu_ps(&C[i * M + jj],      c1);
                _mm256_storeu_ps(&C[i * M + jj + 8],  c2);
                _mm256_storeu_ps(&C[i * M + jj + 16], c3);
                _mm256_storeu_ps(&C[i * M + jj + 24], c4);
            }
        }
        if (M32 < M) {
            for (int i = ii; i < ii + BS && i < N; i++) {
                int j = M32;
                for (; j + 8 <= M; j += 8) {
                    __m256 c = _mm256_setzero_ps();
                    for (int kk = 0; kk < K; kk++) {
                        __m256 a_vec = _mm256_set1_ps(A[i * K + kk]);
                        c = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&B[kk * M + j]), c);
                    }
                    _mm256_storeu_ps(&C[i * M + j], c);
                }
                for (; j < M; j++) {
                    float acc = 0.0f;
                    for (int kk = 0; kk < K; kk++)
                        acc += A[i * K + kk] * B[kk * M + j];
                    C[i * M + j] = acc;
                }
            }
        }
    }
}
void MTxM(const float* A, const float* B, float* C, int N, int K,int M) {
    int BS = 32;
    int M32 = (M / 32) * 32;
    #pragma omp parallel for 
    for (int kk = 0; kk < K; kk += BS) {
        for (int jj = 0; jj < M32; jj += BS) {
            for (int k = kk; k < kk + BS && k < K; k++) {
				__m256 c1 = _mm256_setzero_ps();
				__m256 c2 = _mm256_setzero_ps();
				__m256 c3 = _mm256_setzero_ps();
				__m256 c4 = _mm256_setzero_ps();
                for (int i = 0; i < N; i++) {
                    __m256 a_vec = _mm256_set1_ps(A[i * K + k]);
                    c1 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&B[i * M + jj]),      c1);
                    c2 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&B[i * M + jj + 8]),  c2);
                    c3 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&B[i * M + jj + 16]), c3);
                    c4 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&B[i * M + jj + 24]), c4);
                }

                _mm256_storeu_ps(&C[k * M + jj],      c1);
                _mm256_storeu_ps(&C[k * M + jj + 8],  c2);
                _mm256_storeu_ps(&C[k * M + jj + 16], c3);
                _mm256_storeu_ps(&C[k * M + jj + 24], c4);
            }
        }
        if (M32 < M) {
            for (int k = kk; k < kk + BS && k < K; k++) {
                int j = M32;
                for (; j + 8 <= M; j += 8) {
                    __m256 c = _mm256_setzero_ps();
                    for (int i = 0; i < N; i++) {
                        __m256 a_vec = _mm256_set1_ps(A[i * K + k]);
                        c = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&B[i * M + j]), c);
                    }
                    _mm256_storeu_ps(&C[k * M + j], c);
                }
                for (; j < M; j++) {
                    float acc = 0.0f;
                    for (int i = 0; i < N; i++)
                        acc += A[i * K + k] * B[i * M + j];
                    C[k * M + j] = acc;
                }
            }
        }
    }
}

void Error_Cal(const Dataset& S,const Weight& P,float *Error){
	Y_Pred_LN(S,P,Error);
	vector_sub(Error,S.Y.data(),Error,S.SizeY());
}
float MSE(const float *Error,int total){
	return dot(Error,Error,total)/total;
}
float MAE(const float *Error,int total){
	return sum_elements_abs(Error,total)/total;
}
void Y_Pred_LN(const Dataset& S,const Weight& P,float *Y){
	MxM(S.X.data(),P.W.data(),Y,S.N,S.D + 1,S.K);
}
void Grad_MSE(const Dataset& S,float *Error,float* Grad){
	MTxM(S.X.data(),Error,Grad,S.N,S.D + 1,S.K);
	vector_mul(Grad,1.0f/S.SizeY(),Grad,(S.D + 1) * S.K);
}
void Grad_MAE(const Dataset& S,float *Error,float* Grad){
	sgn(Error,S.SizeY());
	MTxM(S.X.data(),Error,Grad,S.N,S.D + 1,S.K);
	vector_mul(Grad,1.0f/S.SizeY(),Grad,(S.D + 1) * S.K);
}
void Update_WB(Weight& P,float* Grad,float lr){
	vector_fma_scalar(-lr,Grad,P.W.data(),P.SizeW());
}
float VIF_Cal(const float* feature,const float *feature_pred ,float feature_mean,int n){
	float RSS, TSS;
	RSS = Dist(feature,feature_pred,n);
	TSS = Dist(feature,feature_mean,n);
	return TSS/RSS;
}
void VIF(const Dataset &S, vector<float>& VIF_arr,string sel ,int epo_vif,float lr_vif) {
    Dataset S_temp(S.N, S.D - 1, 1);
    Weight P_temp(S.D - 1, 1);
    vector<float> feature_pred(S.N);
    for (int i = 0; i < (int)VIF_arr.size(); i++) {
        P_temp.initial(0.0f, 0.0f);
        for (int j = 0; j < S.N; j++) {
            S_temp.Y[j] = S.atX(j, i);
            int col = 0;
            for (int e = 0; e < S.D; e++) {
                if (e != i)
                    S_temp.atX(j, col++) = S.atX(j, e);
            }
        }
        float feature_mean = (1.0f / S.N) * sum_elements(S_temp.Y.data(), S.N);
        vector<float> X_original = S_temp.X;
        Scaler scaler(S.D - 1);
        feature_scaling(S_temp,"standard", scaler);
        LinearRegression(S_temp, P_temp, lr_vif, epo_vif, sel);
        rescale_weights(P_temp, scaler);
        MxM(X_original.data(),P_temp.W.data(),feature_pred.data(),S_temp.N,S_temp.D + 1,S_temp.K);
        VIF_arr[i] = VIF_Cal(S_temp.Y.data(), feature_pred.data(), feature_mean, S.N);
    }
}
float sum_elements(const float* A,int n){
	int i = 0;float result = 0;
	__m256 sum1 = _mm256_setzero_ps();
	__m256 sum2 = _mm256_setzero_ps();
	__m256 sum3 = _mm256_setzero_ps();
	__m256 sum4 = _mm256_setzero_ps();
	for(;i<= n - 32;i+=32){
		sum1 = _mm256_add_ps(_mm256_loadu_ps(A + i),sum1);
		sum2 = _mm256_add_ps(_mm256_loadu_ps(A + i + 8),sum2);
		sum3 = _mm256_add_ps(_mm256_loadu_ps(A + i + 16),sum3);
		sum4 = _mm256_add_ps(_mm256_loadu_ps(A + i + 24),sum4);
	}
	__m256 sum = _mm256_add_ps(_mm256_add_ps(sum1, sum2),_mm256_add_ps(sum3, sum4));
    for (; i <= n - 8; i += 8) {
        sum = _mm256_add_ps(_mm256_loadu_ps(A + i), sum);
    }
	__m128 low  = _mm256_castps256_ps128(sum);
    __m128 high = _mm256_extractf128_ps(sum, 1);
    __m128 s = _mm_add_ps(low, high);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    result = _mm_cvtss_f32(s);
    for (;i<n;i++){
    	result += A[i];	
	}
    return result;
}
int scalar_sgn(float x){return (x>0)?1:(x == 0) ? 0 : -1;}
void sgn_neglect(float *A,int n){
	int i = 0;
	__m256 sign_mask = _mm256_set1_ps(-0.0f); 
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 sign_1, sign_2, sign_3, sign_4;
    for(;i<= n - 32;i+=32){
		sign_1 = _mm256_and_ps(_mm256_loadu_ps(A + i), sign_mask);
		sign_2 = _mm256_and_ps(_mm256_loadu_ps(A + i + 8), sign_mask);
		sign_3 = _mm256_and_ps(_mm256_loadu_ps(A + i + 16), sign_mask);
		sign_4 = _mm256_and_ps(_mm256_loadu_ps(A + i + 24), sign_mask);
		_mm256_storeu_ps(A + i, _mm256_or_ps(one, sign_1));
		_mm256_storeu_ps(A + i + 8, _mm256_or_ps(one, sign_2));
		_mm256_storeu_ps(A + i + 16, _mm256_or_ps(one, sign_3));
		_mm256_storeu_ps(A + i + 24, _mm256_or_ps(one, sign_4));
	}
	for (; i <= n - 8; i += 8) {
        sign_1 = _mm256_and_ps(_mm256_loadu_ps(A + i), sign_mask);
        _mm256_storeu_ps(A + i, _mm256_or_ps(one, sign_1));
    }
    for (;i<n;i++){
    	uint32_t temp = *(uint32_t*)&A[i];
        temp = (temp & 0x80000000) | 0x3f800000;
        A[i] = *(float*)&temp;
	}
}
void sgn(float* A, int n) {
	int i = 0;
    __m256 zero = _mm256_setzero_ps();
    __m256 pos1 = _mm256_set1_ps(1.0f);
    __m256 neg1 = _mm256_set1_ps(-1.0f);
    __m256 val1,val2;
    for (; i <n - 16; i += 16) {
        val1 = _mm256_loadu_ps(A + i);
        val2 = _mm256_loadu_ps(A + i + 8);
        __m256 mask_gt1 = _mm256_cmp_ps(val1, zero, _CMP_GT_OS);
        __m256 mask_lt1 = _mm256_cmp_ps(val1, zero, _CMP_LT_OS);
		__m256 mask_gt2 = _mm256_cmp_ps(val2, zero, _CMP_GT_OS);
        __m256 mask_lt2 = _mm256_cmp_ps(val2, zero, _CMP_LT_OS);
        __m256 res1 = _mm256_blendv_ps(zero, pos1, mask_gt1);
        res1 = _mm256_blendv_ps(res1, neg1, mask_lt1);
		__m256 res2 = _mm256_blendv_ps(zero, pos1, mask_gt2);
        res2 = _mm256_blendv_ps(res2, neg1, mask_lt2);
        _mm256_storeu_ps(A + i, res1);
		_mm256_storeu_ps(A + i + 8, res2);
    }
	for (; i < n - 8; i += 8) {
        val1 = _mm256_loadu_ps(A + i);
        __m256 mask_gt1 = _mm256_cmp_ps(val1, zero, _CMP_GT_OS);
        __m256 mask_lt1 = _mm256_cmp_ps(val1, zero, _CMP_LT_OS);
        __m256 res1 = _mm256_blendv_ps(zero, pos1, mask_gt1);
        res1 = _mm256_blendv_ps(res1, neg1, mask_lt1);
        _mm256_storeu_ps(A + i, res1);
    }
    for (;i<n;i++){
    	A[i] = (float)((A[i] > 0.0f) - (A[i] < 0.0f));
	}
}
float RandUni(float a,float b){
    static std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    uniform_real_distribution<float> dis(a, b);
    return dis(gen);
}
void FillNormal(float* A, int size, float mu, float sigma) {
    static std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    static std::normal_distribution<float> dis(mu, sigma);
    dis.param(std::normal_distribution<float>::param_type(mu, sigma));
    for (int i = 0; i < size; ++i) {
        A[i] = dis(gen);
    }
}
float Min(float a, float b){return (a<=b) ? a:b;}
float Max(float a, float b){return (a>=b) ? a:b;}
void fast_fill_scalar(float* data, float value, int n) {
    __m256 v_val = _mm256_set1_ps(value);
    int i = 0;
    for (; i <= n - 32; i += 32) {
        _mm256_storeu_ps(data + i,      v_val);
        _mm256_storeu_ps(data + i + 8,  v_val);
        _mm256_storeu_ps(data + i + 16, v_val);
        _mm256_storeu_ps(data + i + 24, v_val);
    }
    for (; i < n; i++) data[i] = value;
}
void fast_fill(float* A, const float *B, int n) {
    int i = 0;
    for (; i <= n - 32; i += 32) {
        _mm256_storeu_ps(A + i,_mm256_loadu_ps(B + i));
        _mm256_storeu_ps(A + i + 8,_mm256_loadu_ps(B + i + 8));
        _mm256_storeu_ps(A + i + 16,_mm256_loadu_ps(B + i + 16));
        _mm256_storeu_ps(A + i + 24,_mm256_loadu_ps(B + i + 24));
    }
    for (; i < n; i++) A[i] = B[i];
}	
void feature_scaling(Dataset& S, string type, Scaler& scaler){
    if (toLowerCase(type) == "normal"){
        vector<float> range(S.D);
        for (int j = 0; j < S.D; j++){
            scaler.mu[j] = S.atX(0, j);
            float max_val = scaler.mu[j];
            for (int i = 1; i < S.N; i++){
                scaler.mu[j] = Min(scaler.mu[j], S.atX(i, j));
                max_val      = Max(max_val,       S.atX(i, j));
            }
            scaler.sigma[j]     = max_val - scaler.mu[j];
            scaler.inv_sigma[j] = 1.0f / scaler.sigma[j];
        }
        for (int i = 0; i < S.N; i++){
            vector_sub(&S.X[i*(S.D+1)], scaler.mu.data(),      &S.X[i*(S.D+1)], S.D);
            VxV       (&S.X[i*(S.D+1)], scaler.inv_sigma.data(),&S.X[i*(S.D+1)], S.D);
        }
    }
    else if (toLowerCase(type) == "standard"){
        vector<float> temp(S.D);
        for (int i = 0; i < S.N; i++){
            vector_add(scaler.mu.data(), &S.X[i*(S.D+1)], scaler.mu.data(), S.D);
		}
        vector_mul(scaler.mu.data(), 1.0f/S.N, scaler.mu.data(), S.D);
        for (int i = 0; i < S.N; i++){
            vector_sub(&S.X[i*(S.D+1)], scaler.mu.data(), temp.data(), S.D);
            VxV(temp.data(), temp.data(), temp.data(), S.D);
            vector_add(scaler.sigma.data(), temp.data(), scaler.sigma.data(), S.D);
        }
        vector_mul(scaler.sigma.data(), 1.0f/S.N, scaler.sigma.data(), S.D);
        for (int j = 0; j < S.D; j++){
            scaler.sigma[j]     = sqrt(scaler.sigma[j]);
            scaler.inv_sigma[j] = 1.0f / scaler.sigma[j];
        }
        for (int i = 0; i < S.N; i++){
            vector_sub(&S.X[i*(S.D+1)], scaler.mu.data(),       &S.X[i*(S.D+1)], S.D);
            VxV       (&S.X[i*(S.D+1)], scaler.inv_sigma.data(), &S.X[i*(S.D+1)], S.D);
        }
    }
}
void rescale_weights(Weight& P, const Scaler& scaler){
    // W_true = W_scaled / sigma  (divide, not multiply)
    for (int j = 0; j < P.D; j++)
        vector_mul(P.W.data() + j*P.K, scaler.inv_sigma[j], P.W.data() + j*P.K, P.K);

    // b_true = b_scaled - sum_j(W_true[j][k] * mu[j])
    for (int j = 0; j < P.D; j++)
        vector_fma_scalar(-scaler.mu[j], P.W.data() + j*P.K, P.BiasVector(), P.K);
}
void LinearRegression(const Dataset& S,Weight& P,float lr,int epoch,string sel,Loss_History& L){
	sel = toLowerCase(sel);
	vector <float> Grad(P.SizeW());
	vector <float> Error(S.SizeY());
	if (sel == "mse"){
		for (int i = 0;i<epoch;i++){
			Error_Cal(S,P,Error.data());
			L.save(P, MSE(Error.data(),Error.size()));
			Grad_MSE(S,Error.data(),Grad.data());
			Update_WB(P,Grad.data(),lr);
		}
	}
	else if (sel == "mae"){
		for (int i = 0;i<epoch;i++){
			Error_Cal(S,P,Error.data());
			L.save(P, MAE(Error.data(),Error.size()));
			Grad_MAE(S,Error.data(),Grad.data());
			Update_WB(P,Grad.data(),lr);
		}
	}
}
void LinearRegression(const Dataset& S,Weight& P,float lr,int epoch,string sel){
	sel = toLowerCase(sel);
	vector <float> Grad(P.SizeW());
	vector <float> Error(S.SizeY());
	if (sel == "mse"){
		for (int i = 0;i<epoch;i++){
			Error_Cal(S,P,Error.data());
			Grad_MSE(S,Error.data(),Grad.data());
			Update_WB(P,Grad.data(),lr);
		}
	}
	else if (sel == "mae"){
		for (int i = 0;i<epoch;i++){
			Error_Cal(S,P,Error.data());
			Grad_MAE(S,Error.data(),Grad.data());
			Update_WB(P,Grad.data(),lr);
		}
	}
}
void transform_poly(const Dataset &S,Dataset &S_trans,int degree){
	if (degree < 2) S_trans = S;
	else {
		S_trans.Y = S.Y;
		for (int i = 0; i < S.N; i++){
			float x_val = S.atX(i, 0);
	        for (int d = 1; d <= degree; d++) {
	            S_trans.atX(i, d-1) = pow(x_val, d);
	        }
    	}
	}
}
void TrainFunction(Dataset S,Weight& P,float lr,int epoch,string fs_sel,string sel,Loss_History &L){
	int N = S.N, D = S.D, K = S.K;
	vector <float> w_ini(D * K); vector <float> b_ini(K);
	FillNormal(w_ini.data(),w_ini.size(),0,1);
	FillNormal(b_ini.data(),b_ini.size(),0,1);
	P.initial(w_ini.data(),b_ini.data());
	Scaler scaler(D); feature_scaling(S,fs_sel, scaler);
	LinearRegression(S, P, lr, epoch, sel, L); 	rescale_weights(P, scaler);
}
void FeatureEngineer(Dataset &S){
	for (int i = 0;i<S.N;i++){
		S.atX(i,1) = S.atX(i,0) * S.atX(i,1); 
	}
}
void ShowVecto(const vector<float>& A){
	for (float x : A){
		cout << x << " ";
	}
}
void StartTime() {start = high_resolution_clock::now();}
void StopTime() {stop = high_resolution_clock::now();}
void ShowTime() {
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "\nThoi gian thuc thi: " << duration.count() << " ms" << endl;
}
