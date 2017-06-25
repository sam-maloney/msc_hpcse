#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

/// Select which runtime measure to use
//#define USE_TIMER
#define USE_TSC

//#ifdef USE_TIMER
#include "timer.hpp"
//#endif // USE_TIMER

#ifdef USE_TSC
#include "tsc_x86.hpp"
#endif // USE_TSC

typedef double value_type;
typedef std::size_t size_type;

#ifndef M_PI
    constexpr value_type M_PI = 3.14159265358979323846;
#endif // M_PI

class Diffusion2D {
public:
    Diffusion2D(const value_type D,
                const size_type N,
                const value_type dt)
    : D_(D), N_(N), Ntot(N_*N_), dt_(dt)
    {
        /// real space grid spacing
        dh_ = 1.0 / (N_ - 1);

        /// stencil factor
        fac_ = dt_*D_ / (2.0*dh_*dh_);
        f1_ = 1/fac_ - 2;
        f2_ = 1+2*fac_;

        rho_.resize(Ntot, 0.0);
        rho_half.resize(Ntot, 0.0);

        c_.resize(N_, 0.0);
        c_rcp_.resize(N_, 0.0);
        d_.resize(12*N_, 0.0);

        n_step_ = 0;

        initialize_density();
        initialize_thomas();
    }

    void advance()
    {
        /// Dirichlet boundaries; central differences in space

        value_type c1 = -c_[1];
        size_type i;

        /// For each row, apply Thomas algorithm for implicit solution
        /// Loop unrolled by 12 for data reuse and preparation for AVX
        for(i = 1; i < N_-12; i += 12) {
            /// First is the forward sweep in x direction
            value_type tmp0, tmp1, tmp2 , tmp3 , tmp4, tmp5, tmp6, tmp7;
            value_type tmp8, tmp9, tmp10, tmp11, tmpf;

            tmp0  = rho_[(i   )*N_ + 1];
            tmp1  = rho_[(i+1 )*N_ + 1];
            tmp2  = rho_[(i+2 )*N_ + 1];
            tmp3  = rho_[(i+3 )*N_ + 1];
            tmp4  = rho_[(i+4 )*N_ + 1];
            tmp5  = rho_[(i+5 )*N_ + 1];
            tmp6  = rho_[(i+6 )*N_ + 1];
            tmp7  = rho_[(i+7 )*N_ + 1];
            tmp8  = rho_[(i+8 )*N_ + 1];
            tmp9  = rho_[(i+9 )*N_ + 1];
            tmp10 = rho_[(i+10)*N_ + 1];
            tmp11 = rho_[(i+11)*N_ + 1];

            d_[12] = c1*(rho_[(i-1)*N_ + 1] + f1_*tmp0 ) + c1*tmp1;
            d_[13] = c1*(tmp0               + f1_*tmp1 ) + c1*tmp2;
            d_[14] = c1*(tmp1               + f1_*tmp2 ) + c1*tmp3;
            d_[15] = c1*(tmp2               + f1_*tmp3 ) + c1*tmp4;
            d_[16] = c1*(tmp3               + f1_*tmp4 ) + c1*tmp5;
            d_[17] = c1*(tmp4               + f1_*tmp5 ) + c1*tmp6;
            d_[18] = c1*(tmp5               + f1_*tmp6 ) + c1*tmp7;
            d_[19] = c1*(tmp6               + f1_*tmp7 ) + c1*tmp8;
            d_[20] = c1*(tmp7               + f1_*tmp8 ) + c1*tmp9;
            d_[21] = c1*(tmp8               + f1_*tmp9 ) + c1*tmp10;
            d_[22] = c1*(tmp9               + f1_*tmp10) + c1*tmp11;
            d_[23] = c1*(tmp10              + f1_*tmp11) + c1*rho_[(i+12)*N_ + 1];

            for(size_type k = 2; k < N_-2; k++) {
                value_type tmp0, tmp1, tmp2 , tmp3 , tmp4, tmp5, tmp6, tmp7;
                value_type tmp8, tmp9, tmp10, tmp11, tmpf;

                tmp0  = rho_[(i   )*N_ + k];
                tmp1  = rho_[(i+1 )*N_ + k];
                tmp2  = rho_[(i+2 )*N_ + k];
                tmp3  = rho_[(i+3 )*N_ + k];
                tmp4  = rho_[(i+4 )*N_ + k];
                tmp5  = rho_[(i+5 )*N_ + k];
                tmp6  = rho_[(i+6 )*N_ + k];
                tmp7  = rho_[(i+7 )*N_ + k];
                tmp8  = rho_[(i+8 )*N_ + k];
                tmp9  = rho_[(i+9 )*N_ + k];
                tmp10 = rho_[(i+10)*N_ + k];
                tmp11 = rho_[(i+11)*N_ + k];
                tmpf = c_rcp_[k];

                d_[k*12     ] = ( rho_[(i-1)*N_ + k] + f1_*tmp0 + tmp1 +
                                  d_[(k-1)*12     ] ) * tmpf;
                d_[k*12 + 1 ] = ( tmp0  + f1_*tmp1  + tmp2  +
                                  d_[(k-1)*12 + 1 ] ) * tmpf;
                d_[k*12 + 2 ] = ( tmp1  + f1_*tmp2  + tmp3  +
                                  d_[(k-1)*12 + 2 ] ) * tmpf;
                d_[k*12 + 3 ] = ( tmp2  + f1_*tmp3  + tmp4  +
                                  d_[(k-1)*12 + 3 ] ) * tmpf;
                d_[k*12 + 4 ] = ( tmp3  + f1_*tmp4  + tmp5  +
                                  d_[(k-1)*12 + 4 ] ) * tmpf;
                d_[k*12 + 5 ] = ( tmp4  + f1_*tmp5  + tmp6  +
                                  d_[(k-1)*12 + 5 ] ) * tmpf;
                d_[k*12 + 6 ] = ( tmp5  + f1_*tmp6  + tmp7  +
                                  d_[(k-1)*12 + 6 ] ) * tmpf;
                d_[k*12 + 7 ] = ( tmp6  + f1_*tmp7  + tmp8  +
                                  d_[(k-1)*12 + 7 ] ) * tmpf;
                d_[k*12 + 8 ] = ( tmp7  + f1_*tmp8  + tmp9  +
                                  d_[(k-1)*12 + 8 ] ) * tmpf;
                d_[k*12 + 9 ] = ( tmp8  + f1_*tmp9  + tmp10 +
                                  d_[(k-1)*12 + 9 ] ) * tmpf;
                d_[k*12 + 10] = ( tmp9  + f1_*tmp10 + tmp11 +
                                  d_[(k-1)*12 + 10] ) * tmpf;
                d_[k*12 + 11] = ( tmp10 + f1_*tmp11 + rho_[(i+12)*N_ + k] +
                                  d_[(k-1)*12 + 11] ) * tmpf;
            }

            /// Second is the back substitution for the half time step
            tmp0  = rho_[(i+1 )*N_ - 2];
            tmp1  = rho_[(i+2 )*N_ - 2];
            tmp2  = rho_[(i+3 )*N_ - 2];
            tmp3  = rho_[(i+4 )*N_ - 2];
            tmp4  = rho_[(i+5 )*N_ - 2];
            tmp5  = rho_[(i+6 )*N_ - 2];
            tmp6  = rho_[(i+7 )*N_ - 2];
            tmp7  = rho_[(i+8 )*N_ - 2];
            tmp8  = rho_[(i+9 )*N_ - 2];
            tmp9  = rho_[(i+10)*N_ - 2];
            tmp10 = rho_[(i+11)*N_ - 2];
            tmp11 = rho_[(i+12)*N_ - 2];
            tmpf = c_rcp_[N_-2];

            rho_half[(i+1 )*N_ - 2] = ( rho_[i*N_ - 2]  + f1_*tmp0 + tmp1 +
                                        d_[12*N_ - 36] ) * tmpf;
            rho_half[(i+2 )*N_ - 2] = ( tmp0  + f1_*tmp1  + tmp2 +
                                        d_[12*N_ - 35] ) * tmpf;
            rho_half[(i+3 )*N_ - 2] = ( tmp1  + f1_*tmp2  + tmp3 +
                                        d_[12*N_ - 34] ) * tmpf;
            rho_half[(i+4 )*N_ - 2] = ( tmp2  + f1_*tmp3  + tmp4 +
                                        d_[12*N_ - 33] ) * tmpf;
            rho_half[(i+5 )*N_ - 2] = ( tmp3  + f1_*tmp4  + tmp5 +
                                        d_[12*N_ - 32] ) * tmpf;
            rho_half[(i+6 )*N_ - 2] = ( tmp4  + f1_*tmp5  + tmp6 +
                                        d_[12*N_ - 31] ) * tmpf;
            rho_half[(i+7 )*N_ - 2] = ( tmp5  + f1_*tmp6  + tmp7 +
                                        d_[12*N_ - 30] ) * tmpf;
            rho_half[(i+8 )*N_ - 2] = ( tmp6  + f1_*tmp7  + tmp8 +
                                        d_[12*N_ - 29] ) * tmpf;
            rho_half[(i+9 )*N_ - 2] = ( tmp7  + f1_*tmp8  + tmp9 +
                                        d_[12*N_ - 28] ) * tmpf;
            rho_half[(i+10)*N_ - 2] = ( tmp8  + f1_*tmp9  + tmp10 +
                                        d_[12*N_ - 27] ) * tmpf;
            rho_half[(i+11)*N_ - 2] = ( tmp9  + f1_*tmp10 + tmp11 +
                                        d_[12*N_ - 26] ) * tmpf;
            rho_half[(i+12)*N_ - 2] = ( tmp10 + f1_*tmp11 + rho_[(i+13)*N_ - 2] +
                                        d_[12*N_ - 25] ) * tmpf;

            for(size_type k = N_-3; k > 0; k--) {
                value_type tmpc = c_[k];

                rho_half[(i   )*N_ + k] = d_[k*12     ] -
                                         tmpc*rho_half[(i   )*N_ + k + 1];
                rho_half[(i+1 )*N_ + k] = d_[k*12 + 1 ] -
                                         tmpc*rho_half[(i+1 )*N_ + k + 1];
                rho_half[(i+2 )*N_ + k] = d_[k*12 + 2 ] -
                                         tmpc*rho_half[(i+2 )*N_ + k + 1];
                rho_half[(i+3 )*N_ + k] = d_[k*12 + 3 ] -
                                         tmpc*rho_half[(i+3 )*N_ + k + 1];
                rho_half[(i+4 )*N_ + k] = d_[k*12 + 4 ] -
                                         tmpc*rho_half[(i+4 )*N_ + k + 1];
                rho_half[(i+5 )*N_ + k] = d_[k*12 + 5 ] -
                                         tmpc*rho_half[(i+5 )*N_ + k + 1];
                rho_half[(i+6 )*N_ + k] = d_[k*12 + 6 ] -
                                         tmpc*rho_half[(i+6 )*N_ + k + 1];
                rho_half[(i+7 )*N_ + k] = d_[k*12 + 7 ] -
                                         tmpc*rho_half[(i+7 )*N_ + k + 1];
                rho_half[(i+8 )*N_ + k] = d_[k*12 + 8 ] -
                                         tmpc*rho_half[(i+8 )*N_ + k + 1];
                rho_half[(i+9 )*N_ + k] = d_[k*12 + 9 ] -
                                         tmpc*rho_half[(i+9 )*N_ + k + 1];
                rho_half[(i+10)*N_ + k] = d_[k*12 + 10] -
                                         tmpc*rho_half[(i+10)*N_ + k + 1];
                rho_half[(i+11)*N_ + k] = d_[k*12 + 11] -
                                         tmpc*rho_half[(i+11)*N_ + k + 1];
            }
        }

        /// Complete any remaining rows
        for(; i < N_-1; ++i) {
            /// First is the forward sweep in x direction
            d_[1] = c1*(rho_[(i-1)*N_ + 1] + f1_*rho_[i*N_ + 1]) +
                    c1*rho_[(i+1)*N_ + 1];
            for(size_type k = 2; k < N_-2; k++) {
                d_[k] = ( rho_[(i-1)*N_ + k] + f1_*rho_[i*N_ + k] +
                          rho_[(i+1)*N_ + k] + d_[k-1] ) * c_rcp_[k];
            }
            /// Second is the back substitution for the half time step
            rho_half[i*N_ + N_ - 2] = (    rho_[(i  )*N_ - 2] +
                                       f1_*rho_[(i+1)*N_ - 2] +
                                           rho_[(i+2)*N_ - 2] +
                                           d_[N_-3] ) * c_rcp_[N_-2];
            for(size_type k = N_-3; k > 0; k--) {
                rho_half[i*N_ + k] = d_[k] - c_[k]*rho_half[i*N_ + k + 1];
            }
        }

        size_type j;

        /// For each column, apply Thomas algorithm for implicit solution
        /// Loop unrolled by 12 for scalar replacement and preparation for AVX
        for(j = 1; j < N_-12; j += 12) {
            /// First is the forward sweep in y direction
            value_type tmp0, tmp1, tmp2 , tmp3 , tmp4, tmp5, tmp6, tmp7;
            value_type tmp8, tmp9, tmp10, tmp11, tmpf;

            tmp0  = rho_half[N_ + j     ];
            tmp1  = rho_half[N_ + j + 1 ];
            tmp2  = rho_half[N_ + j + 2 ];
            tmp3  = rho_half[N_ + j + 3 ];
            tmp4  = rho_half[N_ + j + 4 ];
            tmp5  = rho_half[N_ + j + 5 ];
            tmp6  = rho_half[N_ + j + 6 ];
            tmp7  = rho_half[N_ + j + 7 ];
            tmp8  = rho_half[N_ + j + 8 ];
            tmp9  = rho_half[N_ + j + 9 ];
            tmp10 = rho_half[N_ + j + 10];
            tmp11 = rho_half[N_ + j + 11];

            d_[12] = c1*(rho_half[N_ + j - 1] + f1_*tmp0 ) + c1*tmp1;
            d_[13] = c1*(tmp0                 + f1_*tmp1 ) + c1*tmp2;
            d_[14] = c1*(tmp1                 + f1_*tmp2 ) + c1*tmp3;
            d_[15] = c1*(tmp2                 + f1_*tmp3 ) + c1*tmp4;
            d_[16] = c1*(tmp3                 + f1_*tmp4 ) + c1*tmp5;
            d_[17] = c1*(tmp4                 + f1_*tmp5 ) + c1*tmp6;
            d_[18] = c1*(tmp5                 + f1_*tmp6 ) + c1*tmp7;
            d_[19] = c1*(tmp6                 + f1_*tmp7 ) + c1*tmp8;
            d_[20] = c1*(tmp7                 + f1_*tmp8 ) + c1*tmp9;
            d_[21] = c1*(tmp8                 + f1_*tmp9 ) + c1*tmp10;
            d_[22] = c1*(tmp9                 + f1_*tmp10) + c1*tmp11;
            d_[23] = c1*(tmp10                + f1_*tmp11) + c1*rho_half[N_ + j + 12];

            for(size_type k = 2; k < N_-2; k++) {
                value_type tmp0, tmp1, tmp2 , tmp3 , tmp4, tmp5, tmp6, tmp7;
                value_type tmp8, tmp9, tmp10, tmp11, tmpf;

                tmp0  = rho_half[k*N_ + j     ];
                tmp1  = rho_half[k*N_ + j + 1 ];
                tmp2  = rho_half[k*N_ + j + 2 ];
                tmp3  = rho_half[k*N_ + j + 3 ];
                tmp4  = rho_half[k*N_ + j + 4 ];
                tmp5  = rho_half[k*N_ + j + 5 ];
                tmp6  = rho_half[k*N_ + j + 6 ];
                tmp7  = rho_half[k*N_ + j + 7 ];
                tmp8  = rho_half[k*N_ + j + 8 ];
                tmp9  = rho_half[k*N_ + j + 9 ];
                tmp10 = rho_half[k*N_ + j + 10];
                tmp11 = rho_half[k*N_ + j + 11];
                tmpf = c_rcp_[k];

                d_[k*12     ] = ( rho_half[k*N_ + j - 1] + f1_*tmp0 + tmp1 +
                                  d_[(k-1)*12     ] ) * tmpf;
                d_[k*12 + 1 ] = ( tmp0  + f1_*tmp1  + tmp2  +
                                  d_[(k-1)*12 + 1 ] ) * tmpf;
                d_[k*12 + 2 ] = ( tmp1  + f1_*tmp2  + tmp3  +
                                  d_[(k-1)*12 + 2 ] ) * tmpf;
                d_[k*12 + 3 ] = ( tmp2  + f1_*tmp3  + tmp4  +
                                  d_[(k-1)*12 + 3 ] ) * tmpf;
                d_[k*12 + 4 ] = ( tmp3  + f1_*tmp4  + tmp5  +
                                  d_[(k-1)*12 + 4 ] ) * tmpf;
                d_[k*12 + 5 ] = ( tmp4  + f1_*tmp5  + tmp6  +
                                  d_[(k-1)*12 + 5 ] ) * tmpf;
                d_[k*12 + 6 ] = ( tmp5  + f1_*tmp6  + tmp7  +
                                  d_[(k-1)*12 + 6 ] ) * tmpf;
                d_[k*12 + 7 ] = ( tmp6  + f1_*tmp7  + tmp8  +
                                  d_[(k-1)*12 + 7 ] ) * tmpf;
                d_[k*12 + 8 ] = ( tmp7  + f1_*tmp8  + tmp9  +
                                  d_[(k-1)*12 + 8 ] ) * tmpf;
                d_[k*12 + 9 ] = ( tmp8  + f1_*tmp9  + tmp10 +
                                  d_[(k-1)*12 + 9 ] ) * tmpf;
                d_[k*12 + 10] = ( tmp9  + f1_*tmp10 + tmp11 +
                                  d_[(k-1)*12 + 10] ) * tmpf;
                d_[k*12 + 11] = ( tmp10 + f1_*tmp11 + rho_half[k*N_ + j + 12] +
                                  d_[(k-1)*12 + 11] ) * tmpf;
            }

            /// Second is the back substitution for the full time step
            tmp0  = rho_half[(N_-2)*N_ + j     ];
            tmp1  = rho_half[(N_-2)*N_ + j + 1 ];
            tmp2  = rho_half[(N_-2)*N_ + j + 2 ];
            tmp3  = rho_half[(N_-2)*N_ + j + 3 ];
            tmp4  = rho_half[(N_-2)*N_ + j + 4 ];
            tmp5  = rho_half[(N_-2)*N_ + j + 5 ];
            tmp6  = rho_half[(N_-2)*N_ + j + 6 ];
            tmp7  = rho_half[(N_-2)*N_ + j + 7 ];
            tmp8  = rho_half[(N_-2)*N_ + j + 8 ];
            tmp9  = rho_half[(N_-2)*N_ + j + 9 ];
            tmp10 = rho_half[(N_-2)*N_ + j + 10];
            tmp11 = rho_half[(N_-2)*N_ + j + 11];
            tmpf = c_rcp_[N_-2];

            rho_[(N_-2)*N_ + j     ] = ( rho_half[(N_-2)*N_ + j - 1] + f1_*tmp0 +
                                         tmp1  + d_[12*N_ - 36] ) * tmpf;
            rho_[(N_-2)*N_ + j + 1 ] = ( tmp0  + f1_*tmp1  +
                                         tmp2  + d_[12*N_ - 35] ) * tmpf;
            rho_[(N_-2)*N_ + j + 2 ] = ( tmp1  + f1_*tmp2  +
                                         tmp3  + d_[12*N_ - 34] ) * tmpf;
            rho_[(N_-2)*N_ + j + 3 ] = ( tmp2  + f1_*tmp3  +
                                         tmp4  + d_[12*N_ - 33] ) * tmpf;
            rho_[(N_-2)*N_ + j + 4 ] = ( tmp3  + f1_*tmp4  +
                                         tmp5  + d_[12*N_ - 32] ) * tmpf;
            rho_[(N_-2)*N_ + j + 5 ] = ( tmp4  + f1_*tmp5  +
                                         tmp6  + d_[12*N_ - 31] ) * tmpf;
            rho_[(N_-2)*N_ + j + 6 ] = ( tmp5  + f1_*tmp6  +
                                         tmp7  + d_[12*N_ - 30] ) * tmpf;
            rho_[(N_-2)*N_ + j + 7 ] = ( tmp6  + f1_*tmp7  +
                                         tmp8  + d_[12*N_ - 29] ) * tmpf;
            rho_[(N_-2)*N_ + j + 8 ] = ( tmp7  + f1_*tmp8  +
                                         tmp9  + d_[12*N_ - 28] ) * tmpf;
            rho_[(N_-2)*N_ + j + 9 ] = ( tmp8  + f1_*tmp9  +
                                         tmp10 + d_[12*N_ - 27] ) * tmpf;
            rho_[(N_-2)*N_ + j + 10] = ( tmp9  + f1_*tmp10 +
                                         tmp11 + d_[12*N_ - 26] ) * tmpf;
            rho_[(N_-2)*N_ + j + 11] = ( tmp10 + f1_*tmp11 +
                                         rho_half[(N_-2)*N_ + j + 12] + d_[12*N_ - 25] ) * tmpf;

            for(size_type k = N_-3; k > 0; k--) {
                value_type tmpc = c_[k];

                rho_[k*N_ + j     ] = d_[k*12     ] - tmpc*rho_[(k + 1)*N_ + j     ];
                rho_[k*N_ + j + 1 ] = d_[k*12 + 1 ] - tmpc*rho_[(k + 1)*N_ + j + 1 ];
                rho_[k*N_ + j + 2 ] = d_[k*12 + 2 ] - tmpc*rho_[(k + 1)*N_ + j + 2 ];
                rho_[k*N_ + j + 3 ] = d_[k*12 + 3 ] - tmpc*rho_[(k + 1)*N_ + j + 3 ];
                rho_[k*N_ + j + 4 ] = d_[k*12 + 4 ] - tmpc*rho_[(k + 1)*N_ + j + 4 ];
                rho_[k*N_ + j + 5 ] = d_[k*12 + 5 ] - tmpc*rho_[(k + 1)*N_ + j + 5 ];
                rho_[k*N_ + j + 6 ] = d_[k*12 + 6 ] - tmpc*rho_[(k + 1)*N_ + j + 6 ];
                rho_[k*N_ + j + 7 ] = d_[k*12 + 7 ] - tmpc*rho_[(k + 1)*N_ + j + 7 ];
                rho_[k*N_ + j + 8 ] = d_[k*12 + 8 ] - tmpc*rho_[(k + 1)*N_ + j + 8 ];
                rho_[k*N_ + j + 9 ] = d_[k*12 + 9 ] - tmpc*rho_[(k + 1)*N_ + j + 9 ];
                rho_[k*N_ + j + 10] = d_[k*12 + 10] - tmpc*rho_[(k + 1)*N_ + j + 10];
                rho_[k*N_ + j + 11] = d_[k*12 + 11] - tmpc*rho_[(k + 1)*N_ + j + 11];
            }
        }

        /// Complete any remaining columns
        for(; j < N_-1; ++j) {
            /// First is the forward sweep in y direction
            d_[1] = c1*(rho_half[N_ + j - 1] + f1_*rho_half[N_ + j]) +
                    c1*rho_half[N_ + j + 1];
            for(size_type k = 2; k < N_-2; k++) {
                d_[k] = ( rho_half[k*N_ + j - 1] + f1_*rho_half[k*N_ + j] +
                          rho_half[k*N_ + j + 1] + d_[k-1] ) * c_rcp_[k];
            }
            /// Second is the back substitution for the full time step
            rho_[(N_ - 2)*N_ + j] = (    rho_half[(N_-2)*N_ + j - 1] +
                                     f1_*rho_half[(N_-2)*N_ + j]     +
                                         rho_half[(N_-2)*N_ + j + 1] +
                                         d_[N_-3] ) * c_rcp_[N_-2];
            for(size_type k = N_-3; k > 0; k--) {
                rho_[k*N_ + j] = d_[k] - c_[k]*rho_[(k + 1)*N_ + j];
            }
        }

        n_step_++;
    }

    void write_density(std::string const& filename) const
    {
        std::ofstream out_file(filename, std::ios::out);

        for(size_type i = 0; i < N_; ++i) {
            for(size_type j = 0; j < N_; ++j) {
                out_file << (i*dh_) << '\t' << (j*dh_) << '\t'
                         << rho_[i*N_ + j] << "\n";
            }
            out_file << "\n";
        }
        out_file.close();
    }

    void write_reference(std::string const& filename)
    {
        std::ofstream out_file(filename, std::ios::out);

        value_type ref_value, t_f;
        t_f = time();

        for(size_type j = 0; j < N_; ++j) {
            out_file << 0.0 << '\t' << (j*dh_) << '\t' << 0.0 << "\n";
        }

        for(size_type i = 1; i < N_-1; ++i) {
            out_file << (i*dh_) << '\t' << 0.0 << '\t' << 0.0 << "\n";
            for(size_type j = 1; j < N_-1; ++j) {
                ref_value = sin(M_PI*i*dh_) * sin(M_PI*j*dh_) *
                            exp(-2*D_*t_f*M_PI*M_PI);
                out_file << (i*dh_) << '\t' << (j*dh_) << '\t'
                         << ref_value << "\n";
            }
            out_file << (i*dh_) << '\t' << ((N_-1)*dh_) << '\t' << 0.0 << "\n";
            out_file << "\n";
        }

        for(size_type j = 0; j < N_; ++j) {
            out_file << ((N_-1)*dh_) << '\t' << (j*dh_) << '\t' << 0.0 << "\n";
        }

        out_file.close();
    }

    value_type compute_rms_error()
    {
        rms_error_ = 0.0;
        value_type t_f = time();

        for(size_type i = 0; i < N_; i++) {
            for(size_type j = 0; j < N_; j++) {
                value_type ref_value = sin(M_PI*i*dh_) * sin(M_PI*j*dh_) *
                                       exp(-2*D_*t_f*M_PI*M_PI);
                rms_error_ += pow(rho_[i*N_ + j] - ref_value, 2);
            }
        }

        rms_error_ = sqrt(rms_error_/(N_*N_));

        return rms_error_;
    }

    value_type rms_error() const
    {
        return rms_error_;
    }

    value_type CFL() const
    {
        return fac_;
    }

    value_type time() const
    {
        return n_step_ * dt_;
    }

    size_type time_step() const
    {
        return n_step_;
    }

    value_type dt() const
    {
        return dt_;
    }

private:

    void initialize_density()
    {
        /// initialize rho(x,y,t=0) = sin(pi*x)*sin(pi*y)
        for (size_type i = 1; i < N_-1; ++i) {
            for (size_type j = 1; j < N_-1; ++j) {
                rho_[i*N_ + j] = sin(M_PI*i*dh_) * sin(M_PI*j*dh_);
            }
        }
    }

    void initialize_thomas()
    {
        c_[1] = -fac_ / f2_;
        for(size_type i = 2; i < N_-1; i++) {
            c_[i] = -fac_ / (f2_ - fac_*c_[i-1]);
            c_rcp_[i] = fac_/(f2_ + fac_*c_[i-1]);
        }
    }

    value_type D_;
    size_type N_, Ntot, n_step_;

    value_type dh_, dt_, fac_, f1_, f2_, rms_error_;

    std::vector<value_type> rho_, rho_half, c_, d_, c_rcp_;
};

int main(int argc, char* argv[])
{
    timer t_total;
    t_total.start();

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " D N dt (t_max)" << std::endl;
        return 1;
    }

    const value_type D  = std::stod (argv[1]);
    const size_type  N  = std::stoul(argv[2]);
    const value_type dt = std::stod (argv[3]);

    value_type t_max ;

    if (argc > 4) {
        t_max = std::stoul(argv[5]);
    } else {
        t_max = 0.1;
    }

    std::cout << "Running Scalar_original_12 Simulations" << '\n';
    std::cout << "N = " << N << '\t' << "dt = " << dt << std::endl;

    myInt64 min_cycles = 0;
    value_type e_rms;
    size_type n_runs = 1;

    for(size_type i = 0; i < n_runs; i++) {
        Diffusion2D system(D, N, dt);

#ifdef USE_TIMER
        timer t;
        t.start();
#endif // USE_TIMER

#ifdef USE_TSC
        myInt64 start, cycles;
        start = start_tsc();
#endif // USE_TSC

        while (system.time() < t_max) {
            system.advance();
        }

#ifdef USE_TIMER
        t.stop();
//        std::cout << "Timing: " << N << " " << t.get_timing() << std::endl;
#endif // USE_TIMER

#ifdef USE_TSC
        cycles = stop_tsc(start);
//        std::cout << "Cycles = " << cycles << std::endl;
#endif // USE_TSC

        system.compute_rms_error();
        e_rms = system.rms_error();

        if ( (e_rms < 0.001) && ( (cycles < min_cycles) || (min_cycles == 0) ) ) {
            min_cycles = cycles;
        }
    }

    std::cout << "RMS Error of final run = " << e_rms << '\n';
    std::cout << "Minimum Cycles over " << n_runs << " runs = " << min_cycles << '\n';

    t_total.stop();
    double timing = t_total.get_timing();
//    std::cout << "Total program execution time = " << timing << " seconds\n" << std::endl;

    unsigned hours = 0, minutes = 0;
    if ( timing >= 3600 ) {
        hours = static_cast<unsigned>(floor(timing/3600));
        timing -= hours*3600;
    }
    if ( timing >= 60 ) {
        minutes = static_cast<unsigned>(floor(timing/60));
        timing -= minutes*60;
    }
    std::cout << "Total program execution time = " << hours << "h : " << minutes
              << "m : " << timing << "s\n" << std::endl;

    return 0;
}
