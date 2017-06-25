#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <x86intrin.h>

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
//        size_type i;
        __m256d f1_v = _mm256_set1_pd( f1_  );
        __m256d c1_v = _mm256_set1_pd(-c_[1]);

//        /// For each row, apply Thomas algorithm for implicit solution
//        /// Loop unrolled by 4 for scalar replacement and preparation for AVX
//        for(i = 1; i < N_-4; i += 4) {
//            /// First is the forward sweep in x direction
//            value_type tmp0 = rho_[    i*N_ + 1];
//            value_type tmp1 = rho_[(i+1)*N_ + 1];
//            value_type tmp2 = rho_[(i+2)*N_ + 1];
//            value_type tmp3 = rho_[(i+3)*N_ + 1];
//
//            d_[4] = c1*(rho_[(i-1)*N_ + 1] + f1_*tmp0) + c1*tmp1;
//            d_[5] = c1*(tmp0               + f1_*tmp1) + c1*tmp2;
//            d_[6] = c1*(tmp1               + f1_*tmp2) + c1*tmp3;
//            d_[7] = c1*(tmp2               + f1_*tmp3) + c1*rho_[(i+4)*N_ + 1];
//
//            for(size_type k = 2; k < N_-2; k++) {
//
//                value_type tmp0 = rho_[    i*N_ + k];
//                value_type tmp1 = rho_[(i+1)*N_ + k];
//                value_type tmp2 = rho_[(i+2)*N_ + k];
//                value_type tmp3 = rho_[(i+3)*N_ + k];
//                value_type tmpf = c_rcp_[k];
//
//                d_[k*4]     = ( rho_[(i-1)*N_ + k] + f1_*tmp0 + tmp1 +
//                                d_[(k-1)*4] ) * tmpf;
//                d_[k*4 + 1] = ( tmp0 + f1_*tmp1 + tmp2 +
//                                d_[(k-1)*4 + 1] ) * tmpf;
//                d_[k*4 + 2] = ( tmp1 + f1_*tmp2 + tmp3 +
//                                d_[(k-1)*4 + 2] ) * tmpf;
//                d_[k*4 + 3] = ( tmp2 + f1_*tmp3 + rho_[(i+4)*N_ + k] +
//                                d_[(k-1)*4 + 3] ) * tmpf;
//            }
//
//            /// Second is the back substitution for the half time step
//            tmp0 = rho_[(i+1)*N_ - 2];
//            tmp1 = rho_[(i+2)*N_ - 2];
//            tmp2 = rho_[(i+3)*N_ - 2];
//            tmp3 = rho_[(i+4)*N_ - 2];
//            value_type tmpf = c_rcp_[N_-2];
//
//            rho_half[(i+1)*N_ - 2] = ( rho_[i*N_ - 2]  + f1_*tmp0 + tmp1 +
//                                       d_[4*N_ - 12] ) * tmpf;
//            rho_half[(i+2)*N_ - 2] = ( tmp0 + f1_*tmp1 + tmp2 +
//                                       d_[4*N_ - 11] ) * tmpf;
//            rho_half[(i+3)*N_ - 2] = ( tmp1 + f1_*tmp2 + tmp3 +
//                                       d_[4*N_ - 10] ) * tmpf;
//            rho_half[(i+4)*N_ - 2] = ( tmp2 + f1_*tmp3 + rho_[(i+5)*N_ - 2] +
//                                       d_[4*N_ - 9] )  * tmpf;
//
//            for(size_type k = N_-3; k > 0; k--) {
//
//                value_type tmpc = c_[k];
//
//                rho_half[    i*N_ + k] = d_[k*4] - tmpc*rho_half[i*N_ + k + 1];
//                rho_half[(i+1)*N_ + k] = d_[k*4 + 1] -
//                                         tmpc*rho_half[(i+1)*N_ + k + 1];
//                rho_half[(i+2)*N_ + k] = d_[k*4 + 2] -
//                                         tmpc*rho_half[(i+2)*N_ + k + 1];
//                rho_half[(i+3)*N_ + k] = d_[k*4 + 3] -
//                                         tmpc*rho_half[(i+3)*N_ + k + 1];
//            }
//        }
//
//        /// Complete any remaining rows
//        for(; i < N_-1; ++i) {
//            /// First is the forward sweep in x direction
//            d_[1] = c1*(rho_[(i-1)*N_ + 1] + f1_*rho_[i*N_ + 1]) +
//                    c1*rho_[(i+1)*N_ + 1];
//            for(size_type k = 2; k < N_-1; k++) {
//                d_[k] = ( rho_[(i-1)*N_ + k] + f1_*rho_[i*N_ + k] +
//                          rho_[(i+1)*N_ + k] + d_[k-1] ) * c_rcp_[k];
//            }
//            /// Second is the back substitution for the half time step
//            rho_half[i*N_ + N_ - 2] = d_[N_ - 2];
//            for(size_type k = N_-3; k > 0; k--) {
//                rho_half[i*N_ + k] = d_[k] - c_[k]*rho_half[i*N_ + k + 1];
//            }
//        }


        /// For each row, apply Thomas algorithm for implicit solution
        /// Loop unrolled by 4 for data reuse and AVX
        size_type j;
        for(j = 1; j < N_-4; j += 4) {
            /// First is the forward sweep in y direction
            __m256d rho_lv0 = _mm256_loadu_pd(rho_.data() + N_ + j - 1);
            __m256d rho_cv0 = _mm256_loadu_pd(rho_.data() + N_ + j    );
            __m256d rho_rv0 = _mm256_loadu_pd(rho_.data() + N_ + j + 1);

            __m256d tmp0_v = _mm256_fmadd_pd(f1_v, rho_cv0, rho_lv0);
            __m256d tmp1_v = _mm256_mul_pd  (c1_v, rho_rv0);
            __m256d tmp2_v = _mm256_fmadd_pd(c1_v, tmp0_v      , tmp1_v);
            __m256d c_rcp_v0 = _mm256_set1_pd (c_rcp_[2]);
            __m256d d_prv_v0 = _mm256_mul_pd  (tmp2_v, c_rcp_v0);

            _mm256_storeu_pd(d_.data() + 4, tmp2_v);

            for(size_type k = 2; k < N_-2; k += 1) {
              __m256d rho_lv0 = _mm256_loadu_pd(rho_.data() + k*N_ + j - 1);
              __m256d rho_cv0 = _mm256_loadu_pd(rho_.data() + k*N_ + j    );
              __m256d rho_rv0 = _mm256_loadu_pd(rho_.data() + k*N_ + j + 1);

              __m256d tmp0_v = _mm256_fmadd_pd(f1_v    , rho_cv0, rho_lv0);
              __m256d tmp1_v = _mm256_fmadd_pd(c_rcp_v0, rho_rv0, d_prv_v0    );
              __m256d tmp2_v = _mm256_fmadd_pd(c_rcp_v0, tmp0_v, tmp1_v);

              c_rcp_v0 = _mm256_set1_pd(c_rcp_[k+1]);
              d_prv_v0 = _mm256_mul_pd (tmp2_v, c_rcp_v0);
              _mm256_storeu_pd(d_.data() + 4*k, tmp2_v);
            }

            /// Second is the back substitution for the half time step
            rho_lv0 = _mm256_loadu_pd(rho_.data() + (N_-2)*N_ + j - 1);
            rho_cv0 = _mm256_loadu_pd(rho_.data() + (N_-2)*N_ + j    );
            rho_rv0 = _mm256_loadu_pd(rho_.data() + (N_-2)*N_ + j + 1);
            tmp0_v = _mm256_fmadd_pd(f1_v    , rho_cv0, rho_lv0);
            tmp1_v = _mm256_fmadd_pd(c_rcp_v0, rho_rv0, d_prv_v0);
            __m256d rho_half_pr_v = _mm256_fmadd_pd(c_rcp_v0, tmp0_v, tmp1_v);

            __m128d tmp3_v, tmp4_v;

            tmp3_v = _mm256_extractf128_pd(rho_half_pr_v, 0);
            tmp4_v = _mm256_extractf128_pd(rho_half_pr_v, 1);

            _mm_store_sd (rho_half.data() + (j+1)*N_ - 2, tmp3_v);
            _mm_storeh_pd(rho_half.data() + (j+2)*N_ - 2, tmp3_v);
            _mm_store_sd (rho_half.data() + (j+3)*N_ - 2, tmp4_v);
            _mm_storeh_pd(rho_half.data() + (j+4)*N_ - 2, tmp4_v);

//            _mm256_storeu_pd(rho_half.data() + (N_-2)*N_ + j, tmp2_v);

            for(size_type k = N_-3; k > 0; k--) {
                __m256d c_v, d_v;

                c_v = _mm256_set1_pd(-c_[k]);
                d_v = _mm256_loadu_pd(d_.data() + k*4);
                rho_half_pr_v = _mm256_fmadd_pd(c_v, rho_half_pr_v, d_v);

                __m128d tmp1_v, tmp2_v;

                tmp1_v = _mm256_extractf128_pd(rho_half_pr_v, 0);
                tmp2_v = _mm256_extractf128_pd(rho_half_pr_v, 1);

                _mm_store_sd (rho_half.data() + (j  )*N_ + k, tmp1_v);
                _mm_storeh_pd(rho_half.data() + (j+1)*N_ + k, tmp1_v);
                _mm_store_sd (rho_half.data() + (j+2)*N_ + k, tmp2_v);
                _mm_storeh_pd(rho_half.data() + (j+3)*N_ + k, tmp2_v);

//                _mm256_storeu_pd(rho_half.data() + k*N_ + j, tmp0_v);
            }
        }

        /// Complete any remaining rows
        for(; j < N_-1; ++j) {
            /// First is the forward sweep in x direction
            d_[1] = c1*(rho_[N_ + j - 1] + f1_*rho_[N_ + j]) +
                    c1* rho_[N_ + j + 1];
            for(size_type k = 2; k < N_-2; k++) {
                d_[k] = ( rho_[k*N_ + j - 1] + f1_*rho_[k*N_ + j] +
                          rho_[k*N_ + j + 1] + d_[k-1] ) * c_rcp_[k];
            }
            /// Second is the back substitution for the half time step
            rho_half[j*N_ + N_ - 2] = (    rho_[(N_-2)*N_ + j - 1] +
                                       f1_*rho_[(N_-2)*N_ + j]     +
                                           rho_[(N_-2)*N_ + j + 1] +
                                           d_[N_-3] ) * c_rcp_[N_-2];
            for(size_type k = N_-3; k > 0; k--) {
                rho_half[j*N_ + k] = d_[k] - c_[k]*rho_half[j*N_ + k + 1];
            }
        }


        /// For each column, apply Thomas algorithm for implicit solution
        /// Loop unrolled by 4 for data reuse and AVX
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
            __m256d rho_pr_v = _mm256_fmadd_pd(c_rcp_v0, tmp0_v, tmp1_v);

            __m128d tmp3_v, tmp4_v;

            tmp3_v = _mm256_extractf128_pd(rho_pr_v, 0);
            tmp4_v = _mm256_extractf128_pd(rho_pr_v, 1);

            _mm_store_sd (rho_.data() + (j+1)*N_ - 2, tmp3_v);
            _mm_storeh_pd(rho_.data() + (j+2)*N_ - 2, tmp3_v);
            _mm_store_sd (rho_.data() + (j+3)*N_ - 2, tmp4_v);
            _mm_storeh_pd(rho_.data() + (j+4)*N_ - 2, tmp4_v);

            for(size_type k = N_-3; k > 0; k--) {
                __m256d c_v, d_v, tmp0_v;

                c_v    = _mm256_set1_pd (-c_[k]);
                d_v    = _mm256_loadu_pd(d_.data() + k*4);
                rho_pr_v = _mm256_fmadd_pd(c_v, rho_pr_v, d_v);

                __m128d tmp1_v, tmp2_v;

                tmp1_v = _mm256_extractf128_pd(rho_pr_v, 0);
                tmp2_v = _mm256_extractf128_pd(rho_pr_v, 1);

                _mm_store_sd (rho_.data() + (j  )*N_ + k, tmp1_v);
                _mm_storeh_pd(rho_.data() + (j+1)*N_ + k, tmp1_v);
                _mm_store_sd (rho_.data() + (j+2)*N_ + k, tmp2_v);
                _mm_storeh_pd(rho_.data() + (j+3)*N_ + k, tmp2_v);
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
            rho_[j*N_ + N_ - 2] = (    rho_half[(N_-2)*N_ + j - 1] +
                                   f1_*rho_half[(N_-2)*N_ + j]     +
                                       rho_half[(N_-2)*N_ + j + 1] +
                                       d_[N_-3] ) * c_rcp_[N_-2];
            for(size_type k = N_-3; k > 0; k--) {
                rho_[j*N_ + k] = d_[k] - c_[k]*rho_[j*N_ + k + 1];
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
        c_.resize(N_, 0.0);
        c_rcp_.resize(N_, 0.0);
        d_.resize(4*N_, 0.0);

        c_[1] = -fac_ / f2_;
        for(size_type i = 2; i < N_-1; i++) {
            c_[i] = -fac_ / (f2_ - fac_*c_[i-1]);
            c_rcp_[i] = fac_/(f2_ + fac_*c_[i-1]);
        }
    }

    size_type N_, Ntot, n_step_;
    value_type D_, dh_, dt_, fac_, f1_, f2_, rms_error_;
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

    std::cout << "Running AVX_transposed Simulations" << '\n';
    std::cout << "N = " << N << '\t' << "dt = " << dt << std::endl;

    myInt64 min_cycles = 0;
    value_type e_rms;
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
