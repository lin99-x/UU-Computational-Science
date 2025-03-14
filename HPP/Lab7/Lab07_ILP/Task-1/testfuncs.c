#include "testfuncs.h"

void f_std(const double * __restrict a, 
	   const double * __restrict b, 
	   double * __restrict c, int N) {
  int i;
  double x = 0.3;
  for(i = 0; i < N; i++) {
    c[i] = a[i]*a[i] + b[i] + x;
  }
}

void f_opt(const double * __restrict a, 
	   const double * __restrict b, 
	   double * __restrict c, int N) {
  int i;
  double x = 0.3;
  for(i = 0; i < N; i += 4) {
    c[i] = a[i]*a[i] + b[i] + x;
    c[i+1] = a[i+1]*a[i+1] + b[i+1] + x;
    c[i+2] = a[i+2]*a[i+2] + b[i+2] + x;
    c[i+3] = a[i+3]*a[i+3] + b[i+3] + x;
    // c[i+4] = a[i+4]*a[i+4] + b[i+4] + x;
    // c[i+5] = a[i+5]*a[i+5] + b[i+5] + x;
    // c[i+6] = a[i+6]*a[i+6] + b[i+6] + x;
    // c[i+7] = a[i+7]*a[i+7] + b[i+7] + x;
    // c[i+8] = a[i+8]*a[i+8] + b[i+8] + x;
    // c[i+9] = a[i+9]*a[i+9] + b[i+9] + x;
    // c[i+10] = a[i+10]*a[i+10] + b[i+10] + x;
    // c[i+11] = a[i+11]*a[i+11] + b[i+11] + x;
    // c[i+12] = a[i+12]*a[i+12] + b[i+12] + x;
    // c[i+13] = a[i+13]*a[i+13] + b[i+13] + x;
    // c[i+14] = a[i+14]*a[i+14] + b[i+14] + x;
    // c[i+15] = a[i+15]*a[i+15] + b[i+15] + x;
    // c[i+16] = a[i+16]*a[i+16] + b[i+16] + x;
    // c[i+17] = a[i+17]*a[i+17] + b[i+17] + x;
    // c[i+18] = a[i+18]*a[i+18] + b[i+18] + x;
    // c[i+19] = a[i+19]*a[i+19] + b[i+19] + x;
    // c[i+20] = a[i+20]*a[i+20] + b[i+20] + x;
    // c[i+21] = a[i+21]*a[i+21] + b[i+21] + x;
    // c[i+22] = a[i+22]*a[i+22] + b[i+22] + x;
    // c[i+23] = a[i+23]*a[i+23] + b[i+23] + x;
    // c[i+24] = a[i+24]*a[i+24] + b[i+24] + x;
    // c[i+25] = a[i+25]*a[i+25] + b[i+25] + x;
    // c[i+26] = a[i+26]*a[i+26] + b[i+26] + x;
    // c[i+27] = a[i+27]*a[i+27] + b[i+27] + x;
    // c[i+28] = a[i+28]*a[i+28] + b[i+28] + x;
    // c[i+29] = a[i+29]*a[i+29] + b[i+29] + x;
    // c[i+30] = a[i+30]*a[i+30] + b[i+30] + x;
    // c[i+31] = a[i+31]*a[i+31] + b[i+31] + x;
    // c[i+32] = a[i+32]*a[i+32] + b[i+32] + x;
    // c[i+33] = a[i+33]*a[i+33] + b[i+33] + x;
    // c[i+34] = a[i+34]*a[i+34] + b[i+34] + x;
    // c[i+35] = a[i+35]*a[i+35] + b[i+35] + x;
    // c[i+36] = a[i+36]*a[i+36] + b[i+36] + x;
    // c[i+37] = a[i+37]*a[i+37] + b[i+37] + x;
    // c[i+38] = a[i+38]*a[i+38] + b[i+38] + x;
    // c[i+39] = a[i+39]*a[i+39] + b[i+39] + x;
  }
}

