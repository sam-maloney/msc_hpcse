#include <iostream>
#include <fstream>
#include <string>
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
    : D_(D), N_(N), N2_(N_*N_), dt_(dt)
    {
        /// real space grid spacing
        dh_ = 1.0 / (N_ - 1);

        /// stencil factor
        fac_ = dt_*D_ / (2.0*dh_*dh_);
        f1_ = 1/fac_ - 2;
        f2_ = 1+2*fac_;

        rho_ = static_cast<value_type*>(aligned_alloc(64, (N2_ + 2*N_)*sizeof(value_type)));
        rho_half_ = static_cast<value_type*>(aligned_alloc(64, N2_*sizeof(value_type)));

        c_.resize(N_, 0.0);
        c_rcp_.resize(N_, 0.0);

        #pragma omp parallel
        {
        d_.resize(8*N_, 0.0);
        }

        n_step_ = 0;
        remainder_index_ = N_ - (N_ % 8);

        initialize_density();
        initialize_thomas();
    }

    ~Diffusion2D()
    {
        free(rho_);
        free(rho_half_);
    }

    void run_simulation(const value_type t_max)
    {
        /// Dirichlet boundaries; central differences in space

        value_type c1 = -c_[1];
        __m256d f1_v  = _mm256_set1_pd( f1_  );
        __m256d c1_v  = _mm256_set1_pd(-c_[1]);

        #pragma omp parallel
        {

        while( time() < t_max ) {

        /// For each row, apply Thomas algorithm for implicit solution
        /// Loop unrolled by 8 for data reuse and preparation for AVX
        #pragma omp for
        for(size_type i = 1; i < N_-6; i += 8) {
            /// First is the forward sweep in x direction
            value_type tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmpf;

            tmp0 = rho_[(i  )*N_ + 1];
            tmp1 = rho_[(i+1)*N_ + 1];
            tmp2 = rho_[(i+2)*N_ + 1];
            tmp3 = rho_[(i+3)*N_ + 1];
            tmp4 = rho_[(i+4)*N_ + 1];
            tmp5 = rho_[(i+5)*N_ + 1];
            tmp6 = rho_[(i+6)*N_ + 1];
            tmp7 = rho_[(i+7)*N_ + 1];

            d_[8 ] = c1*(rho_[(i-1)*N_ + 1] + f1_*tmp0) + c1*tmp1;
            d_[9 ] = c1*(tmp0 + f1_*tmp1) + c1*tmp2;
            d_[10] = c1*(tmp1 + f1_*tmp2) + c1*tmp3;
            d_[11] = c1*(tmp2 + f1_*tmp3) + c1*tmp4;
            d_[12] = c1*(tmp3 + f1_*tmp4) + c1*tmp5;
            d_[13] = c1*(tmp4 + f1_*tmp5) + c1*tmp6;
            d_[14] = c1*(tmp5 + f1_*tmp6) + c1*tmp7;
            d_[15] = c1*(tmp6 + f1_*tmp7) + c1*rho_[(i+8)*N_ + 1];

            for(size_type k = 2; k < N_-2; k++) {
                value_type tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmpf;

                tmp0 = rho_[(i  )*N_ + k];
                tmp1 = rho_[(i+1)*N_ + k];
                tmp2 = rho_[(i+2)*N_ + k];
                tmp3 = rho_[(i+3)*N_ + k];
                tmp4 = rho_[(i+4)*N_ + k];
                tmp5 = rho_[(i+5)*N_ + k];
                tmp6 = rho_[(i+6)*N_ + k];
                tmp7 = rho_[(i+7)*N_ + k];
                tmpf = c_rcp_[k];

                d_[k*8    ] = ( rho_[(i-1)*N_ + k] + f1_*tmp0 + tmp1 +
                                d_[(k-1)*8    ] ) * tmpf;
                d_[k*8 + 1] = ( tmp0 + f1_*tmp1 + tmp2 +
                                d_[(k-1)*8 + 1] ) * tmpf;
                d_[k*8 + 2] = ( tmp1 + f1_*tmp2 + tmp3 +
                                d_[(k-1)*8 + 2] ) * tmpf;
                d_[k*8 + 3] = ( tmp2 + f1_*tmp3 + tmp4 +
                                d_[(k-1)*8 + 3] ) * tmpf;
                d_[k*8 + 4] = ( tmp3 + f1_*tmp4 + tmp5 +
                                d_[(k-1)*8 + 4] ) * tmpf;
                d_[k*8 + 5] = ( tmp4 + f1_*tmp5 + tmp6 +
                                d_[(k-1)*8 + 5] ) * tmpf;
                d_[k*8 + 6] = ( tmp5 + f1_*tmp6 + tmp7 +
                                d_[(k-1)*8 + 6] ) * tmpf;
                d_[k*8 + 7] = ( tmp6 + f1_*tmp7 + rho_[(i+8)*N_ + k] +
                                d_[(k-1)*8 + 7] ) * tmpf;
            }

            /// Second is the back substitution for the half time step
            tmp0 = rho_[(i+1)*N_ - 2];
            tmp1 = rho_[(i+2)*N_ - 2];
            tmp2 = rho_[(i+3)*N_ - 2];
            tmp3 = rho_[(i+4)*N_ - 2];
            tmp4 = rho_[(i+5)*N_ - 2];
            tmp5 = rho_[(i+6)*N_ - 2];
            tmp6 = rho_[(i+7)*N_ - 2];
            tmp7 = rho_[(i+8)*N_ - 2];
            tmpf = c_rcp_[N_-2];

            rho_half_[(i  )*N_ - 2] = ( rho_[i*N_ - 2]  + f1_*tmp0 + tmp1 +
                                       d_[8*N_ - 24] ) * tmpf;
            rho_half_[(i+1)*N_ - 2] = ( tmp0 + f1_*tmp1 + tmp2 +
                                       d_[8*N_ - 23] ) * tmpf;
            rho_half_[(i+2)*N_ - 2] = ( tmp1 + f1_*tmp2 + tmp3 +
                                       d_[8*N_ - 22] ) * tmpf;
            rho_half_[(i+3)*N_ - 2] = ( tmp2 + f1_*tmp3 + tmp4 +
                                       d_[8*N_ - 21] ) * tmpf;
            rho_half_[(i+4)*N_ - 2] = ( tmp3 + f1_*tmp4 + tmp5 +
                                       d_[8*N_ - 20] ) * tmpf;
            rho_half_[(i+5)*N_ - 2] = ( tmp4 + f1_*tmp5 + tmp6 +
                                       d_[8*N_ - 19] ) * tmpf;
            rho_half_[(i+6)*N_ - 2] = ( tmp5 + f1_*tmp6 + tmp7 +
                                       d_[8*N_ - 18] ) * tmpf;
            rho_half_[(i+7)*N_ - 2] = ( tmp6 + f1_*tmp7 + rho_[(i+9)*N_ - 2] +
                                       d_[8*N_ - 17] ) * tmpf;

            for(size_type k = N_-3; k > 0; k--) {
                value_type tmpc = c_[k];

                rho_half_[(i-1)*N_ + k] = d_[k*8    ] -
                                         tmpc*rho_half_[(i-1)*N_ + k + 1];
                rho_half_[(i  )*N_ + k] = d_[k*8 + 1] -
                                         tmpc*rho_half_[(i  )*N_ + k + 1];
                rho_half_[(i+1)*N_ + k] = d_[k*8 + 2] -
                                         tmpc*rho_half_[(i+1)*N_ + k + 1];
                rho_half_[(i+2)*N_ + k] = d_[k*8 + 3] -
                                         tmpc*rho_half_[(i+2)*N_ + k + 1];
                rho_half_[(i+3)*N_ + k] = d_[k*8 + 4] -
                                         tmpc*rho_half_[(i+3)*N_ + k + 1];
                rho_half_[(i+4)*N_ + k] = d_[k*8 + 5] -
                                         tmpc*rho_half_[(i+4)*N_ + k + 1];
                rho_half_[(i+5)*N_ + k] = d_[k*8 + 6] -
                                         tmpc*rho_half_[(i+5)*N_ + k + 1];
                rho_half_[(i+6)*N_ + k] = d_[k*8 + 7] -
                                         tmpc*rho_half_[(i+6)*N_ + k + 1];
            }
        } // main row loop

        /// For each column, apply Thomas algorithm for implicit solution
        /// Loop unrolled by 8 for data reuse and AVX
        #pragma omp for nowait
        for(size_type j = 0; j < N_-7; j += 8) {
            /// First is the forward sweep in y direction
            __m256d rho_half0_lv, rho_half0_cv, rho_half0_rv;
            __m256d rho_half1_lv, rho_half1_cv, rho_half1_rv;
            __m256d tmp0_v , tmp1_v , tmp2_v , tmp3_v, tmp4_v, tmp5_v;
            __m256d d_pr0_v, d_pr1_v, c_rcp_v;

            rho_half0_lv = _mm256_loadu_pd(rho_half_ + N_ + j - 1);
            rho_half0_cv = _mm256_loadu_pd(rho_half_ + N_ + j    );
            rho_half0_rv = _mm256_loadu_pd(rho_half_ + N_ + j + 1);
            rho_half1_lv = _mm256_loadu_pd(rho_half_ + N_ + j + 3);
            rho_half1_cv = _mm256_loadu_pd(rho_half_ + N_ + j + 4);
            rho_half1_rv = _mm256_loadu_pd(rho_half_ + N_ + j + 5);

            tmp0_v = _mm256_fmadd_pd(f1_v, rho_half0_cv, rho_half0_lv);
            tmp1_v = _mm256_mul_pd  (c1_v, rho_half0_rv);
            tmp2_v = _mm256_fmadd_pd(c1_v, tmp0_v      , tmp1_v);
            tmp3_v = _mm256_fmadd_pd(f1_v, rho_half1_cv, rho_half1_lv);
            tmp4_v = _mm256_mul_pd  (c1_v, rho_half1_rv);
            tmp5_v = _mm256_fmadd_pd(c1_v, tmp3_v      , tmp4_v);

            c_rcp_v = _mm256_set1_pd(c_rcp_[2]);
            d_pr0_v = _mm256_mul_pd (tmp2_v, c_rcp_v);
            d_pr1_v = _mm256_mul_pd (tmp5_v, c_rcp_v);

            _mm256_storeu_pd(d_.data() + 8 , tmp2_v);
            _mm256_storeu_pd(d_.data() + 12, tmp5_v);

            for(size_type k = 2; k < N_-2; k += 1) {
                __m256d rho_half0_lv, rho_half0_cv, rho_half0_rv;
                __m256d rho_half1_lv, rho_half1_cv, rho_half1_rv;
                __m256d tmp0_v , tmp1_v , tmp2_v, tmp3_v, tmp4_v, tmp5_v;

                rho_half0_lv = _mm256_loadu_pd(rho_half_ + k*N_ + j - 1);
                rho_half0_cv = _mm256_loadu_pd(rho_half_ + k*N_ + j    );
                rho_half0_rv = _mm256_loadu_pd(rho_half_ + k*N_ + j + 1);
                rho_half1_lv = _mm256_loadu_pd(rho_half_ + k*N_ + j + 3);
                rho_half1_cv = _mm256_loadu_pd(rho_half_ + k*N_ + j + 4);
                rho_half1_rv = _mm256_loadu_pd(rho_half_ + k*N_ + j + 5);

                tmp0_v = _mm256_fmadd_pd(f1_v   , rho_half0_cv, rho_half0_lv);
                tmp1_v = _mm256_fmadd_pd(c_rcp_v, rho_half0_rv, d_pr0_v     );
                tmp2_v = _mm256_fmadd_pd(c_rcp_v, tmp0_v      , tmp1_v      );
                tmp3_v = _mm256_fmadd_pd(f1_v   , rho_half1_cv, rho_half1_lv);
                tmp4_v = _mm256_fmadd_pd(c_rcp_v, rho_half1_rv, d_pr1_v     );
                tmp5_v = _mm256_fmadd_pd(c_rcp_v, tmp3_v      , tmp4_v      );

                c_rcp_v = _mm256_set1_pd(c_rcp_[k+1]);
                d_pr0_v = _mm256_mul_pd (tmp2_v, c_rcp_v);
                d_pr1_v = _mm256_mul_pd (tmp5_v, c_rcp_v);
                _mm256_storeu_pd(d_.data() + 8*k    , tmp2_v);
                _mm256_storeu_pd(d_.data() + 8*k + 4, tmp5_v);
            }

            /// Second is the back substitution for the full time step
            __m256d rho_pr0_v, rho_pr1_v;

            rho_half0_lv = _mm256_loadu_pd(rho_half_ + (N_-2)*N_ + j - 1);
            rho_half0_cv = _mm256_loadu_pd(rho_half_ + (N_-2)*N_ + j    );
            rho_half0_rv = _mm256_loadu_pd(rho_half_ + (N_-2)*N_ + j + 1);
            rho_half1_lv = _mm256_loadu_pd(rho_half_ + (N_-2)*N_ + j + 3);
            rho_half1_cv = _mm256_loadu_pd(rho_half_ + (N_-2)*N_ + j + 4);
            rho_half1_rv = _mm256_loadu_pd(rho_half_ + (N_-2)*N_ + j + 5);

            tmp0_v    = _mm256_fmadd_pd(f1_v   , rho_half0_cv, rho_half0_lv);
            tmp1_v    = _mm256_fmadd_pd(c_rcp_v, rho_half0_rv, d_pr0_v     );
            rho_pr0_v = _mm256_fmadd_pd(c_rcp_v, tmp0_v      , tmp1_v      );
            tmp3_v    = _mm256_fmadd_pd(f1_v   , rho_half1_cv, rho_half1_lv);
            tmp4_v    = _mm256_fmadd_pd(c_rcp_v, rho_half1_rv, d_pr1_v     );
            rho_pr1_v = _mm256_fmadd_pd(c_rcp_v, tmp3_v      , tmp4_v      );

            _mm256_storeu_pd(rho_ + (N_-1)*N_ + j    , rho_pr0_v);
            _mm256_storeu_pd(rho_ + (N_-1)*N_ + j + 4, rho_pr1_v);

            for(size_type k = N_-3; k > 0; k--) {
                __m256d c_v, d0_v, d1_v;
                __m256d tmp0_v, tmp1_v;

                c_v  = _mm256_set1_pd(-c_[k]);
                d0_v = _mm256_loadu_pd(d_.data() + k*8);
                d1_v = _mm256_loadu_pd(d_.data() + k*8 + 4);

                rho_pr0_v = _mm256_fmadd_pd(c_v, rho_pr0_v, d0_v);
                rho_pr1_v = _mm256_fmadd_pd(c_v, rho_pr1_v, d1_v);
                _mm256_storeu_pd(rho_ + (k+1)*N_ + j    , rho_pr0_v);
                _mm256_storeu_pd(rho_ + (k+1)*N_ + j + 4, rho_pr1_v);
            }
        } // main column loop

        #pragma omp single
        {
        n_step_++;
        }

        }  // while time < t_max
        } // OMP prallel region
    }

    void write_density(std::string const& filename) const
    {
        std::ofstream out_file(filename, std::ios::out);

        for(size_type j = 0; j < N_; ++j) {
            out_file << 0.0 << '\t' << (j*dh_) << '\t' << 0.0 << "\n";
        }

        for(size_type i = 2; i < N_; ++i) {
            out_file << ((i-1)*dh_) << '\t' << 0.0 << '\t' << 0.0 << "\n";
            for(size_type j = 1; j < N_-1; ++j) {
                out_file << ((i-1)*dh_) << '\t' << (j*dh_) << '\t'
                         << rho_[i*N_ + j] << "\n";
            }
            out_file << ((i-1)*dh_) << '\t' << ((N_-1)*dh_) << '\t' << 0.0 << "\n";
            out_file << "\n";
        }

        for(size_type j = 0; j < N_; ++j) {
            out_file << ((N_-1)*dh_) << '\t' << (j*dh_) << '\t' << 0.0 << "\n";
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

        for(size_type i = 2; i < N_; ++i) {
            out_file << ((i-1)*dh_) << '\t' << 0.0 << '\t' << 0.0 << "\n";
            for(size_type j = 1; j < N_-1; ++j) {
                ref_value = sin(M_PI*(i-1)*dh_) * sin(M_PI*j*dh_) *
                            exp(-2*D_*t_f*M_PI*M_PI);
                out_file << ((i-1)*dh_) << '\t' << (j*dh_) << '\t'
                         << ref_value << "\n";
            }
            out_file << ((i-1)*dh_) << '\t' << ((N_-1)*dh_) << '\t' << 0.0 << "\n";
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

        for(size_type i = 2; i < N_; i++) {
            for(size_type j = 1; j < N_-1; j++) {
                value_type ref_value = sin(M_PI*(i-1)*dh_) * sin(M_PI*j*dh_) *
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
        for (size_type i = 2; i < N_; ++i) {
            for (size_type j = 1; j < N_-1; ++j) {
                rho_[i*N_ + j] = sin(M_PI*(i-1)*dh_) * sin(M_PI*j*dh_);
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

    size_type N_, N2_, n_step_, remainder_index_;
    value_type D_, dh_, dt_, fac_, f1_, f2_, rms_error_;
    std::vector<value_type> c_, c_rcp_;
    value_type *rho_, *rho_half_;

    static std::vector<value_type> d_;
    #pragma omp threadprivate(d_)
};

std::vector<value_type> Diffusion2D::d_;


int main(int argc, char* argv[])
{
    timer t_total;
    t_total.start();

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " D N dt (n_runs) (n_steps)" << std::endl;
        return 1;
    }

    const value_type D  = std::stod (argv[1]);
    size_type  N  = std::stoul(argv[2]);
    const value_type dt = std::stod (argv[3]);

    if ( N % 8 != 0 ) {
        N += 8 - (N % 8);
        std::cout << "N must be a multiple of 8 to prevent false sharing. Using N = "
                  << N << " instead." << '\n';
    }

    value_type t_max;
    size_type n_runs;

    if (argc > 4) {
        n_runs = std::stoul(argv[4]);
    } else {
        n_runs = 1;
    }

    if (argc > 5) {
        t_max = std::stoul(argv[5]) * dt;
    } else {
        t_max = 0.1;
    }

    std::cout << "Running OMP aligned_separate Simulations" << '\n';
    std::cout << "N = " << N << '\t' << "dt = " << dt << std::endl;

    myInt64 min_cycles = 0;
    value_type e_rms, final_time;

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

        system.run_simulation(t_max);

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
        final_time = system.time();

        if ( (e_rms < 0.001) && ( (cycles < min_cycles) || (min_cycles == 0) ) ) {
            min_cycles = cycles;
        }
    }

    std::cout << "RMS Error of final run = " << e_rms << '\n';
    std::cout << "At a final time = " << final_time << '\n';
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
