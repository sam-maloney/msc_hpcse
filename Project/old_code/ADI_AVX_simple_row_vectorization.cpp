#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <x86intrin.h>
#include <cstdlib>

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

        n_step_ = 0;

        initialize_density();
        initialize_thomas();
    }

    void advance()
    {
        /// Dirichlet boundaries; central differences in space

        value_type c1 = -c_[1];
        size_type i;
        __m256d f1_v = _mm256_set1_pd( f1_  );
        __m256d c1_v = _mm256_set1_pd(-c_[1]);

        /// For each row, apply Thomas algorithm for implicit solution
        /// Loop unrolled by 4 for scalar replacement and preparation for AVX
        for(i = 1; i < N_-4; i += 4) {
            /// First is the forward sweep in x direction
            __m256d rho_v0 , rho_v1 , rho_v2 , rho_v3 , rho_v4 , rho_v5; 
            __m256d tmp_v0 , tmp_v1 , tmp_v2 , tmp_v3 , tmp_v4 , tmp_v5;
            __m256d rho_uv0, rho_uv1, rho_uv2, rho_uv3;
            __m256d rho_cv0, rho_cv1, rho_cv2, rho_cv3;
            __m256d rho_dv0, rho_dv1, rho_dv2, rho_dv3;
            
            rho_v0 = _mm256_loadu_pd(rho_.data() + (i-1)*N_ + 1);
            rho_v1 = _mm256_loadu_pd(rho_.data() + (i  )*N_ + 1);
            rho_v2 = _mm256_loadu_pd(rho_.data() + (i+1)*N_ + 1);
            rho_v3 = _mm256_loadu_pd(rho_.data() + (i+2)*N_ + 1);
            rho_v4 = _mm256_loadu_pd(rho_.data() + (i+3)*N_ + 1);
            rho_v5 = _mm256_loadu_pd(rho_.data() + (i+4)*N_ + 1);
            
            tmp_v0 = _mm256_shuffle_pd(rho_v0,rho_v1, 0x0);
            tmp_v1 = _mm256_shuffle_pd(rho_v2,rho_v3, 0x0);
            tmp_v2 = _mm256_shuffle_pd(rho_v4,rho_v5, 0x0);
            tmp_v3 = _mm256_shuffle_pd(rho_v0,rho_v1, 0xF);
            tmp_v4 = _mm256_shuffle_pd(rho_v2,rho_v3, 0xF);
            tmp_v5 = _mm256_shuffle_pd(rho_v4,rho_v5, 0xF); 
            
            rho_uv0 = _mm256_permute2f128_pd(tmp_v0, tmp_v1, 0x20);
            rho_dv0 = _mm256_permute2f128_pd(tmp_v1, tmp_v2, 0x20);
            rho_uv1 = _mm256_permute2f128_pd(tmp_v3, tmp_v4, 0x20);
            rho_dv1 = _mm256_permute2f128_pd(tmp_v4, tmp_v5, 0x20);
            rho_uv2 = _mm256_permute2f128_pd(tmp_v0, tmp_v1, 0x31);
            rho_dv2 = _mm256_permute2f128_pd(tmp_v1, tmp_v2, 0x31);
            rho_uv3 = _mm256_permute2f128_pd(tmp_v3, tmp_v4, 0x31);
            rho_dv3 = _mm256_permute2f128_pd(tmp_v4, tmp_v5, 0x31);

            rho_cv0 = _mm256_shuffle_pd (rho_uv0, rho_dv0, 0x5);
            rho_cv1 = _mm256_shuffle_pd (rho_uv1, rho_dv1, 0x5);
            rho_cv2 = _mm256_shuffle_pd (rho_uv2, rho_dv2, 0x5);
            rho_cv3 = _mm256_shuffle_pd (rho_uv3, rho_dv3, 0x5);

            __m256d tmp00_v, tmp01_v, tmp02_v, tmp10_v, tmp11_v, tmp12_v;
            __m256d tmp20_v, tmp21_v, tmp22_v, tmp30_v, tmp31_v, tmp32_v;
            __m256d c_rcp_v0, c_rcp_v1, c_rcp_v2, c_rcp_v3;
            __m256d d_prv_v0, d_prv_v1, d_prv_v2, d_prv_v3;

            tmp00_v = _mm256_fmadd_pd(f1_v, rho_cv0, rho_uv0);
            tmp01_v = _mm256_mul_pd  (c1_v, rho_dv0);
            tmp02_v = _mm256_fmadd_pd(c1_v, tmp00_v, tmp01_v);
            
            _mm256_storeu_pd(d_.data() + 4, tmp02_v);
            
            c_rcp_v1 = _mm256_set1_pd(c_rcp_[2]);
            d_prv_v1 = _mm256_mul_pd (c_rcp_v1, tmp02_v);

            tmp10_v = _mm256_fmadd_pd(f1_v    , rho_cv1, rho_uv1 );
            tmp11_v = _mm256_fmadd_pd(c_rcp_v1, rho_dv1, d_prv_v1);
            tmp12_v = _mm256_fmadd_pd(c_rcp_v1, tmp10_v, tmp11_v );
            c_rcp_v2 = _mm256_set1_pd(c_rcp_[3]);
 
            _mm256_storeu_pd(d_.data() + 8, tmp12_v);
            d_prv_v2 = _mm256_mul_pd (c_rcp_v2, tmp12_v);

            tmp20_v = _mm256_fmadd_pd(f1_v    , rho_cv2, rho_uv2 );
            tmp21_v = _mm256_fmadd_pd(c_rcp_v2, rho_dv2, d_prv_v2);
            tmp22_v = _mm256_fmadd_pd(c_rcp_v2, tmp20_v, tmp21_v );
            c_rcp_v3 = _mm256_set1_pd(c_rcp_[4]);
 
            _mm256_storeu_pd(d_.data() + 12, tmp22_v);
            d_prv_v3 = _mm256_mul_pd (c_rcp_v3, tmp22_v);

            tmp30_v = _mm256_fmadd_pd(f1_v    , rho_cv3, rho_uv3 );
            tmp31_v = _mm256_fmadd_pd(c_rcp_v3, rho_dv3, d_prv_v3);
            tmp32_v = _mm256_fmadd_pd(c_rcp_v3, tmp30_v, tmp31_v );
            c_rcp_v0 = _mm256_set1_pd(c_rcp_[5]);
 
            _mm256_storeu_pd(d_.data() + 16, tmp32_v);
            d_prv_v0 = _mm256_mul_pd (c_rcp_v0, tmp32_v);

//            value_type u[4], c[4], d[4];
//            _mm256_storeu_pd(u, rho_uv0);
//            _mm256_storeu_pd(c, rho_cv0);
//            _mm256_storeu_pd(d, rho_dv0);
//
//            std::cout << "v = {" << rho_[(i-1)*N_ + 1] << ", " << tmp0 << ", " << tmp1
//                      << ", " << tmp2 << ", " << tmp3 << ", " << rho_[(i+4)*N_ + 1] << "}\n";
//            std::cout << "u = {" << u[0] << ", " << u[1] << ", " << u[2] << ", "
//                                 << u[3] << "}\n";
//            std::cout << "c = {" << c[0] << ", " << c[1] << ", " << c[2] << ", "
//                                 << c[3] << "}\n";
//            std::cout << "d = {" << d[0] << ", " << d[1] << ", " << d[2] << ", "
//                                 << d[3] << "}\n";
//            exit(EXIT_SUCCESS);
            
            
            for(size_type k = 5; k < N_-2; k++) {

                value_type tmp0 = rho_[    i*N_ + k];
                value_type tmp1 = rho_[(i+1)*N_ + k];
                value_type tmp2 = rho_[(i+2)*N_ + k];
                value_type tmp3 = rho_[(i+3)*N_ + k];
                value_type tmpf = c_rcp_[k];

                d_[k*4]     = ( rho_[(i-1)*N_ + k] + f1_*tmp0 + tmp1 +
                                d_[(k-1)*4] ) * tmpf;
                d_[k*4 + 1] = ( tmp0 + f1_*tmp1 + tmp2 +
                                d_[(k-1)*4 + 1] ) * tmpf;
                d_[k*4 + 2] = ( tmp1 + f1_*tmp2 + tmp3 +
                                d_[(k-1)*4 + 2] ) * tmpf;
                d_[k*4 + 3] = ( tmp2 + f1_*tmp3 + rho_[(i+4)*N_ + k] +
                                d_[(k-1)*4 + 3] ) * tmpf;
            }

            /// Second is the back substitution for the half time step
            value_type tmp0, tmp1, tmp2, tmp3, tmpf; 

            tmp0 = rho_[(i+1)*N_ - 2];
            tmp1 = rho_[(i+2)*N_ - 2];
            tmp2 = rho_[(i+3)*N_ - 2];
            tmp3 = rho_[(i+4)*N_ - 2];
            tmpf = c_rcp_[N_-2];

            rho_half[(i+1)*N_ - 2] = ( rho_[i*N_ - 2]  + f1_*tmp0 + tmp1 +
                                       d_[4*N_ - 12] ) * tmpf;
            rho_half[(i+2)*N_ - 2] = ( tmp0 + f1_*tmp1 + tmp2 +
                                       d_[4*N_ - 11] ) * tmpf;
            rho_half[(i+3)*N_ - 2] = ( tmp1 + f1_*tmp2 + tmp3 +
                                       d_[4*N_ - 10] ) * tmpf;
            rho_half[(i+4)*N_ - 2] = ( tmp2 + f1_*tmp3 + rho_[(i+5)*N_ - 2] +
                                       d_[4*N_ - 9] )  * tmpf;

            for(size_type k = N_-3; k > 0; k--) {

                value_type tmpc = c_[k];

                rho_half[    i*N_ + k] = d_[k*4] - tmpc*rho_half[i*N_ + k + 1];
                rho_half[(i+1)*N_ + k] = d_[k*4 + 1] -
                                         tmpc*rho_half[(i+1)*N_ + k + 1];
                rho_half[(i+2)*N_ + k] = d_[k*4 + 2] -
                                         tmpc*rho_half[(i+2)*N_ + k + 1];
                rho_half[(i+3)*N_ + k] = d_[k*4 + 3] -
                                         tmpc*rho_half[(i+3)*N_ + k + 1];
            }
        }

        /// Complete any remaining rows
        for(; i < N_-1; ++i) {
            /// First is the forward sweep in x direction
            d_[1] = c1*(rho_[(i-1)*N_ + 1] + f1_*rho_[i*N_ + 1]) +
                    c1*rho_[(i+1)*N_ + 1];
            for(size_type k = 2; k < N_-1; k++) {
                d_[k] = ( rho_[(i-1)*N_ + k] + f1_*rho_[i*N_ + k] +
                          rho_[(i+1)*N_ + k] + d_[k-1] ) * c_rcp_[k];
            }
            /// Second is the back substitution for the half time step
            rho_half[i*N_ + N_ - 2] = d_[N_ - 2];
            for(size_type k = N_-3; k > 0; k--) {
                rho_half[i*N_ + k] = d_[k] - c_[k]*rho_half[i*N_ + k + 1];
            }
        }

        size_type j;

        /// For each column, apply Thomas algorithm for implicit solution
        /// Loop unrolled by 4 for AVX
        for(j = 1; j < N_-4; j += 4) {
            /// First is the forward sweep in y direction
            __m256d rho_half_lv0 = _mm256_loadu_pd(rho_half.data() + N_ + j - 1);
            __m256d rho_half_cv0 = _mm256_loadu_pd(rho_half.data() + N_ + j    );
            __m256d rho_half_rv0 = _mm256_loadu_pd(rho_half.data() + N_ + j + 1);

            __m256d tmp0_v = _mm256_fmadd_pd(f1_v, rho_half_cv0, rho_half_lv0);
            __m256d tmp1_v = _mm256_mul_pd  (c1_v, rho_half_rv0);
            __m256d tmp2_v = _mm256_fmadd_pd(c1_v, tmp0_v      , tmp1_v);
            __m256d c_rcp_v0 = _mm256_set1_pd (c_rcp_[2]);
            __m256d d_prv_v0 = _mm256_mul_pd  (tmp2_v, c_rcp_v0);

            _mm256_storeu_pd(d_.data() + 4, tmp2_v);

            for(size_type k = 2; k < N_-2; k += 1) {
              __m256d rho_half_lv0 = _mm256_loadu_pd(rho_half.data() + k*N_ + j - 1);
              __m256d rho_half_cv0 = _mm256_loadu_pd(rho_half.data() + k*N_ + j    );
              __m256d rho_half_rv0 = _mm256_loadu_pd(rho_half.data() + k*N_ + j + 1);

              __m256d tmp0_v = _mm256_fmadd_pd(f1_v    , rho_half_cv0, rho_half_lv0);
              __m256d tmp1_v = _mm256_fmadd_pd(c_rcp_v0, rho_half_rv0, d_prv_v0    );
              __m256d tmp2_v = _mm256_fmadd_pd(c_rcp_v0, tmp0_v, tmp1_v);

              c_rcp_v0 = _mm256_set1_pd(c_rcp_[k+1]);
              d_prv_v0 = _mm256_mul_pd (tmp2_v, c_rcp_v0);
              _mm256_storeu_pd(d_.data() + 4*k, tmp2_v);
            }

            /// Second is the back substitution for the full time step
            rho_half_lv0 = _mm256_loadu_pd(rho_half.data() + (N_-2)*N_ + j - 1);
            rho_half_cv0 = _mm256_loadu_pd(rho_half.data() + (N_-2)*N_ + j    );
            rho_half_rv0 = _mm256_loadu_pd(rho_half.data() + (N_-2)*N_ + j + 1);
            tmp0_v = _mm256_fmadd_pd(f1_v    , rho_half_cv0, rho_half_lv0);
            tmp1_v = _mm256_fmadd_pd(c_rcp_v0, rho_half_rv0, d_prv_v0    );
            tmp2_v = _mm256_fmadd_pd(c_rcp_v0, tmp0_v      , tmp1_v      );

            _mm256_storeu_pd(rho_.data() + (N_-2)*N_ + j, tmp2_v);

            for(size_type k = N_-3; k > 0; k--) {
                __m256d c_v = _mm256_set1_pd(-c_[k]);
                __m256d d_v = _mm256_loadu_pd(d_.data() + k*4);
                __m256d rho_pr_v = _mm256_loadu_pd(rho_.data() + (k + 1)*N_ + j);

                __m256d tmp_v = _mm256_fmadd_pd(c_v, rho_pr_v, d_v);
                _mm256_storeu_pd(rho_.data() + k*N_ + j, tmp_v);
            }
        }

        /// Complete any remaining columns
        for(; j < N_-1; ++j) {
            /// First is the forward sweep in y direction
            d_[1] = c1*(rho_half[N_ + j - 1] + f1_*rho_half[N_ + j]) +
                    c1*rho_half[N_ + j + 1];
            for(size_type k = 2; k < N_-1; k++) {
                d_[k] = ( rho_half[k*N_ + j - 1] + f1_*rho_half[k*N_ + j] +
                          rho_half[k*N_ + j + 1] + d_[k-1] ) * c_rcp_[k];
            }
            /// Second is the back substitution for the full time step
            rho_[(N_ - 2)*N_ + j] = d_[N_ - 2];
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
        out_file << "\n";

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
        c_.resize(N_, 0.0);
        c_rcp_.resize(N_, 0.0);
        d_.resize(4*N_, 0.0);

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
        std::cerr << "Usage: " << argv[0] << " D N dt (tmax)" << std::endl;
        return 1;
    }

    const value_type D  = std::stod (argv[1]);
    const size_type  N  = std::stoul(argv[2]);
    const value_type dt = std::stod (argv[3]);

    value_type tmax ;

    if (argc > 4) {
        tmax = std::stoul(argv[5]);
    } else {
        tmax = 0.1;
    }

    std::cout << "Running AVX_simple Simulations" << '\n';
    std::cout << "N = " << N << '\t' << "dt = " << dt << std::endl;

    myInt64 minCycles = 0;
    size_type n_runs = 100;

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

        while (system.time() < tmax) {
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
//        std::cout << "RMS Error = " << system.rms_error() << '\n' << std::endl;
//        system.write_reference("Solutions/ADI_AVX.dat");

        if ( (system.rms_error() < 0.001) && ( (cycles < minCycles) || (minCycles == 0) ) ) {
            minCycles = cycles;
        }
    }

    std::cout << "Minimum Cycles over " << n_runs << " runs = " << minCycles << '\n';

    t_total.stop();
    std::cout << "Total program execution time = " << t_total.get_timing() << " s" << std::endl;

    return 0;
}

