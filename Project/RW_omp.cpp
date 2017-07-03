#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <mkl_vsl.h>
#include <x86intrin.h>
#include <omp.h>
#include "timer.hpp"
#include "omp_mutex.hpp"

/// Select which runtime measure to use
//#define USE_TIMER
#define USE_TSC

#ifdef USE_TSC
#include "tsc_x86.hpp"
#endif // USE_TSC

typedef double value_type;
typedef std::size_t size_type;
typedef int particle_type;
typedef uint64_t M_type;

#ifndef M_PI
    constexpr value_type M_PI = 3.14159265358979323846;
#endif // M_PI

class Diffusion2D {
public:
    Diffusion2D(const value_type D,
                const size_type N,
                const M_type M,
                const value_type dt,
                const size_type seed,
                const value_type t_max)
    : D_(D), N_(N), Ntot(N_*N_), M_(M), dt_(dt), t_max_(t_max)
    {
        #pragma omp parallel
        {
            vslNewStream(&stream, VSL_BRNG_SFMT19937, seed);
            vslSkipAheadStream(stream, omp_get_thread_num()*
                4*Ntot*static_cast<long long int>(t_max_/dt_) );
        }

        /// real space grid spacing
        dh_ = 1.0 / (N_ - 1);

        /// lambda factor (probability to move in any single direction)
        lambda_ = dt_*D_ / (dh_*dh_);

        if ( lambda_ >= 0.25 ) {
            lambda_ = 0.2;
            dt_ = lambda_*dh_*dh_ / D_;
            std::cout << "Time-step too large for given n, instead using dt = "
                      << dt_ << std::endl;
        }

        /// conversion factor; m_ij = fac_ * rho_ij
        fac_ = M_*M_PI*M_PI/4.0;

        rho_.resize(Ntot, 0.0);
        m_.resize(Ntot, 0);
        m_tmp_.resize(Ntot, 0);
        locks_.resize(omp_get_max_threads() - 1);

        n_step_ = 0;

        M_real_ = initialize_density();
    }

    ~Diffusion2D()
    {
        vslDeleteStream(&stream);
    }

    void run_simulation()
    {
        /// Dirichlet boundaries

        __m128i zero_v = _mm_setzero_si128();

        #pragma omp parallel
        {

        size_type start_row, end_row;
        start_row =  omp_get_thread_num()    * (N-2)/omp_get_num_threads() + 1;
        end_row   = (omp_get_thread_num()+1) * (N-2)/omp_get_num_threads();

        int r[32] = {0};

        while ( time() < t_max_ )
        {

        /// The first row in the thread's section
        if ( omp_get_thread_num() != 0 ) {
            locks_[omp_get_thread_num()-1].lock();
        }
        size_type j;
        for(j = 1; j < N_-6; j += 6) {
            __m256i mc_v, md_v, t0_v;
            __m256i row0_v, row1_v, row2_v, row3_v;
            __m256  tmp0_v, tmp1_v, tmp2_v, tmp3_v;

            mu_v = _mm256_loadu_si256((__m256i*)(m_tmp_.data() + (start_row-1)*N_ + j - 1));
            mc_v = _mm256_loadu_si256((__m256i*)(m_tmp_.data() + (start_row  )*N_ + j - 1));
            md_v = _mm256_loadu_si256((__m256i*)(m_tmp_.data() + (start_row+1)*N_ + j - 1));

            if ( m_[start_row*N_ + j    ] > 0 ) {
                viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 4, r     , m_[start_row*N_ + j    ], lambda_);
            } else {
                _mm_storeu_si128((__m128i*)(r   ), zero_v);
            }

            if ( m_[start_row*N_ + j + 1] > 0 ) {
                viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 4, r + 8 , m_[start_row*N_ + j + 1], lambda_);
            } else {
                _mm_storeu_si128((__m128i*)(r+8 ), zero_v);
            }

            if ( m_[start_row*N_ + j + 2] > 0 ) {
                viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 4, r + 17, m_[start_row*N_ + j + 2], lambda_);
            } else {
                _mm_storeu_si128((__m128i*)(r+17), zero_v);
            }

            if ( m_[start_row*N_ + j + 3] > 0 ) {
                viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 4, r + 27, m_[start_row*N_ + j + 3], lambda_);
            } else {
                _mm_storeu_si128((__m128i*)(r+27), zero_v);
            }

            if ( m_[start_row*N_ + j + 4] > 0 ) {
                viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 4, r + 4 , m_[start_row*N_ + j + 4], lambda_);
            } else {
                _mm_storeu_si128((__m128i*)(r+4 ), zero_v);
            }

            if ( m_[start_row*N_ + j + 5] > 0 ) {
                viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 4, r + 12, m_[start_row*N_ + j + 5], lambda_);
            } else {
                _mm_storeu_si128((__m128i*)(r+12), zero_v);
            }

            row0_v = _mm256_loadu_si256((__m256i*)(r   ));
            row1_v = _mm256_loadu_si256((__m256i*)(r+8 ));
            row2_v = _mm256_loadu_si256((__m256i*)(r+16));
            row3_v = _mm256_loadu_si256((__m256i*)(r+24));

            tmp0_v = _mm256_shuffle_ps(_mm256_castsi256_ps(row0_v), _mm256_castsi256_ps(row1_v), 0xCC);
            tmp1_v = _mm256_shuffle_ps(_mm256_castsi256_ps(row1_v), _mm256_castsi256_ps(row2_v), 0x99);
            tmp2_v = _mm256_shuffle_ps(_mm256_castsi256_ps(row2_v), _mm256_castsi256_ps(row3_v), 0xCC);
            tmp3_v = _mm256_shuffle_ps(_mm256_castsi256_ps(row3_v), _mm256_castsi256_ps(row0_v), 0x99);

            row0_v = _mm256_castps_si256( _mm256_shuffle_ps(tmp2_v, tmp0_v, 0x88) );
            row1_v = _mm256_castps_si256( _mm256_shuffle_ps(tmp3_v, tmp1_v, 0x88) );
            row2_v = _mm256_castps_si256( _mm256_shuffle_ps(tmp3_v, tmp1_v, 0xDD) );
            row3_v = _mm256_castps_si256( _mm256_shuffle_ps(tmp0_v, tmp2_v, 0xDD) );

            t0_v = _mm256_slli_epi32(row1_v, 2);

            mu_v = _mm256_add_epi32(mu_v, row1_v);
            mc_v = _mm256_add_epi32(mc_v, row0_v);
            mc_v = _mm256_add_epi32(mc_v, row3_v);
            mc_v = _mm256_sub_epi32(mc_v, t0_v);
            md_v = _mm256_add_epi32(md_v, row2_v);

            _mm256_storeu_si256((__m256i*)(m_tmp_.data() + (start_row-1)*N_ + j - 1), mu_v);
            _mm256_storeu_si256((__m256i*)(m_tmp_.data() + (start_row  )*N_ + j - 1), mc_v);
            _mm256_storeu_si256((__m256i*)(m_tmp_.data() + (start_row+1)*N_ + j - 1), md_v);
        } // main column loop

        for(; j < N_-1; ++j) {
            if ( m_[start_row*N_ + j] > 0 ) {
                viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 4, r, m_[start_row*N_ + j], lambda_);

                m_tmp_[(start_row  )*N_ + j - 1] += r[0];
                m_tmp_[(start_row  )*N_ + j + 1] += r[1];
                m_tmp_[(start_row-1)*N_ + j    ] += r[2];
                m_tmp_[(start_row+1)*N_ + j    ] += r[3];
                m_tmp_[(start_row  )*N_ + j    ] -= (r[0] + r[1] + r[2] + r[3]);
            }
        } //remaining column loop

        if ( omp_get_thread_num() != 0 ) {
            locks_[omp_get_thread_num()-1].unlock();
        } // End of first row in thread's section

        /// The middle rows in the thread's section
        for(size_type i = start_row+1; i < end_row; ++i) {
            size_type j;
            for(j = 1; j < N_-6; j += 6) {
                __m256i mc_v, mu_v, md_v, t0_v;
                __m256i row0_v, row1_v, row2_v, row3_v;
                __m256  tmp0_v, tmp1_v, tmp2_v, tmp3_v;

                mu_v = _mm256_loadu_si256((__m256i*)(m_tmp_.data() + (i-1)*N_ + j - 1));
                mc_v = _mm256_loadu_si256((__m256i*)(m_tmp_.data() + (i  )*N_ + j - 1));
                md_v = _mm256_loadu_si256((__m256i*)(m_tmp_.data() + (i+1)*N_ + j - 1));

                if ( m_[i*N_ + j    ] > 0 ) {
                    viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 4, r     , m_[i*N_ + j    ], lambda_);
                } else {
                    _mm_storeu_si128((__m128i*)(r   ), zero_v);
                }

                if ( m_[i*N_ + j + 1] > 0 ) {
                    viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 4, r + 8 , m_[i*N_ + j + 1], lambda_);
                } else {
                    _mm_storeu_si128((__m128i*)(r+8 ), zero_v);
                }

                if ( m_[i*N_ + j + 2] > 0 ) {
                    viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 4, r + 17, m_[i*N_ + j + 2], lambda_);
                } else {
                    _mm_storeu_si128((__m128i*)(r+17), zero_v);
                }

                if ( m_[i*N_ + j + 3] > 0 ) {
                    viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 4, r + 27, m_[i*N_ + j + 3], lambda_);
                } else {
                    _mm_storeu_si128((__m128i*)(r+27), zero_v);
                }

                if ( m_[i*N_ + j + 4] > 0 ) {
                    viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 4, r + 4 , m_[i*N_ + j + 4], lambda_);
                } else {
                    _mm_storeu_si128((__m128i*)(r+4 ), zero_v);
                }

                if ( m_[i*N_ + j + 5] > 0 ) {
                    viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 4, r + 12, m_[i*N_ + j + 5], lambda_);
                } else {
                    _mm_storeu_si128((__m128i*)(r+12), zero_v);
                }

                row0_v = _mm256_loadu_si256((__m256i*)(r   ));
                row1_v = _mm256_loadu_si256((__m256i*)(r+8 ));
                row2_v = _mm256_loadu_si256((__m256i*)(r+16));
                row3_v = _mm256_loadu_si256((__m256i*)(r+24));

                tmp0_v = _mm256_shuffle_ps(_mm256_castsi256_ps(row0_v), _mm256_castsi256_ps(row1_v), 0xCC);
                tmp1_v = _mm256_shuffle_ps(_mm256_castsi256_ps(row1_v), _mm256_castsi256_ps(row2_v), 0x99);
                tmp2_v = _mm256_shuffle_ps(_mm256_castsi256_ps(row2_v), _mm256_castsi256_ps(row3_v), 0xCC);
                tmp3_v = _mm256_shuffle_ps(_mm256_castsi256_ps(row3_v), _mm256_castsi256_ps(row0_v), 0x99);

                row0_v = _mm256_castps_si256( _mm256_shuffle_ps(tmp2_v, tmp0_v, 0x88) );
                row1_v = _mm256_castps_si256( _mm256_shuffle_ps(tmp3_v, tmp1_v, 0x88) );
                row2_v = _mm256_castps_si256( _mm256_shuffle_ps(tmp3_v, tmp1_v, 0xDD) );
                row3_v = _mm256_castps_si256( _mm256_shuffle_ps(tmp0_v, tmp2_v, 0xDD) );

                t0_v = _mm256_slli_epi32(row1_v, 2);

                mu_v = _mm256_add_epi32(mu_v, row1_v);
                mc_v = _mm256_add_epi32(mc_v, row0_v);
                mc_v = _mm256_add_epi32(mc_v, row3_v);
                mc_v = _mm256_sub_epi32(mc_v, t0_v);
                md_v = _mm256_add_epi32(md_v, row2_v);

                _mm256_storeu_si256((__m256i*)(m_tmp_.data() + (i-1)*N_ + j - 1), mu_v);
                _mm256_storeu_si256((__m256i*)(m_tmp_.data() + (i  )*N_ + j - 1), mc_v);
                _mm256_storeu_si256((__m256i*)(m_tmp_.data() + (i+1)*N_ + j - 1), md_v);
            } // main column loop

            for(; j < N_-1; ++j) {
                if ( m_[i*N_ + j] > 0 ) {
                    viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 4, r, m_[i*N_ + j], lambda_);

                    m_tmp_[(i  )*N_ + j - 1] += r[0];
                    m_tmp_[(i  )*N_ + j + 1] += r[1];
                    m_tmp_[(i-1)*N_ + j    ] += r[2];
                    m_tmp_[(i+1)*N_ + j    ] += r[3];
                    m_tmp_[(i  )*N_ + j    ] -= (r[0] + r[1] + r[2] + r[3]);
                }
            } //remaining column loop
        } // main row loop of thread's section

        /// The final row in the thread's section
        if ( omp_get_thread_num() != omp_get_num_threads()-1 ) {
            locks_[omp_get_thread_num()].lock();
        }
        size_type j;
        for(j = 1; j < N_-6; j += 6) {
            __m256i mc_v, mu_v, md_v, t0_v;
            __m256i row0_v, row1_v, row2_v, row3_v;
            __m256  tmp0_v, tmp1_v, tmp2_v, tmp3_v;

            mu_v = _mm256_loadu_si256((__m256i*)(m_tmp_.data() + (end_row-1)*N_ + j - 1));
            mc_v = _mm256_loadu_si256((__m256i*)(m_tmp_.data() + (end_row  )*N_ + j - 1));
            md_v = _mm256_loadu_si256((__m256i*)(m_tmp_.data() + (end_row+1)*N_ + j - 1));

            if ( m_[end_row*N_ + j    ] > 0 ) {
                viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 4, r     , m_[end_row*N_ + j    ], lambda_);
            } else {
                _mm_storeu_si128((__m128i*)(r   ), zero_v);
            }

            if ( m_[end_row*N_ + j + 1] > 0 ) {
                viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 4, r + 8 , m_[end_row*N_ + j + 1], lambda_);
            } else {
                _mm_storeu_si128((__m128i*)(r+8 ), zero_v);
            }

            if ( m_[end_row*N_ + j + 2] > 0 ) {
                viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 4, r + 17, m_[end_row*N_ + j + 2], lambda_);
            } else {
                _mm_storeu_si128((__m128i*)(r+17), zero_v);
            }

            if ( m_[end_row*N_ + j + 3] > 0 ) {
                viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 4, r + 27, m_[end_row*N_ + j + 3], lambda_);
            } else {
                _mm_storeu_si128((__m128i*)(r+27), zero_v);
            }

            if ( m_[end_row*N_ + j + 4] > 0 ) {
                viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 4, r + 4 , m_[end_row*N_ + j + 4], lambda_);
            } else {
                _mm_storeu_si128((__m128i*)(r+4 ), zero_v);
            }

            if ( m_[end_row*N_ + j + 5] > 0 ) {
                viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 4, r + 12, m_[end_row*N_ + j + 5], lambda_);
            } else {
                _mm_storeu_si128((__m128i*)(r+12), zero_v);
            }

            row0_v = _mm256_loadu_si256((__m256i*)(r   ));
            row1_v = _mm256_loadu_si256((__m256i*)(r+8 ));
            row2_v = _mm256_loadu_si256((__m256i*)(r+16));
            row3_v = _mm256_loadu_si256((__m256i*)(r+24));

            tmp0_v = _mm256_shuffle_ps(_mm256_castsi256_ps(row0_v), _mm256_castsi256_ps(row1_v), 0xCC);
            tmp1_v = _mm256_shuffle_ps(_mm256_castsi256_ps(row1_v), _mm256_castsi256_ps(row2_v), 0x99);
            tmp2_v = _mm256_shuffle_ps(_mm256_castsi256_ps(row2_v), _mm256_castsi256_ps(row3_v), 0xCC);
            tmp3_v = _mm256_shuffle_ps(_mm256_castsi256_ps(row3_v), _mm256_castsi256_ps(row0_v), 0x99);

            row0_v = _mm256_castps_si256( _mm256_shuffle_ps(tmp2_v, tmp0_v, 0x88) );
            row1_v = _mm256_castps_si256( _mm256_shuffle_ps(tmp3_v, tmp1_v, 0x88) );
            row2_v = _mm256_castps_si256( _mm256_shuffle_ps(tmp3_v, tmp1_v, 0xDD) );
            row3_v = _mm256_castps_si256( _mm256_shuffle_ps(tmp0_v, tmp2_v, 0xDD) );

            t0_v = _mm256_slli_epi32(row1_v, 2);

            mu_v = _mm256_add_epi32(mu_v, row1_v);
            mc_v = _mm256_add_epi32(mc_v, row0_v);
            mc_v = _mm256_add_epi32(mc_v, row3_v);
            mc_v = _mm256_sub_epi32(mc_v, t0_v);
            md_v = _mm256_add_epi32(md_v, row2_v);

            _mm256_storeu_si256((__m256i*)(m_tmp_.data() + (end_row-1)*N_ + j - 1), mu_v);
            _mm256_storeu_si256((__m256i*)(m_tmp_.data() + (end_row  )*N_ + j - 1), mc_v);
            _mm256_storeu_si256((__m256i*)(m_tmp_.data() + (end_row+1)*N_ + j - 1), md_v);
        } // main column loop

        for(; j < N_-1; ++j) {
            if ( m_[end_row*N_ + j] > 0 ) {
                viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 4, r, m_[end_row*N_ + j], lambda_);

                m_tmp_[(end_row  )*N_ + j - 1] += r[0];
                m_tmp_[(end_row  )*N_ + j + 1] += r[1];
                m_tmp_[(end_row-1)*N_ + j    ] += r[2];
                m_tmp_[(end_row+1)*N_ + j    ] += r[3];
                m_tmp_[(end_row  )*N_ + j    ] -= (r[0] + r[1] + r[2] + r[3]);
            }
        } //remaining column loop

        if ( omp_get_thread_num() != omp_get_num_threads()-1 ) {
            locks_[omp_get_thread_num()].unlock();
        } // End of final row in thread's section

        #pragma omp single
        {
        m_ = m_tmp_;
        n_step_++;
        }

        } // while time() < t_max
        } // OMP parallel region
    }

    void compute_density()
    {
        for(size_type i = 1; i < N_-1; ++i) {
            for(size_type j = 1; j < N_-1; ++j) {
                rho_[i*N_ + j] = static_cast<value_type>(m_[i*N_ + j]) /
                                                        (fac_*dh_*dh_);
            }
        }
    }

    M_type compute_num_particles()
    {
        M_type n_particles = 0;

        for(size_type i = 0; i < N_; ++i) {
            for(size_type j = 0; j < N_; ++j) {
                n_particles += m_[i*N_ + j];
            }
        }

        return n_particles;
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

        for(size_type i = 0; i < N_; ++i) {
            for(size_type j = 0; j < N_; ++j) {
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
        return lambda_;
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

    M_type M_real() const
    {
        return M_real_;
    }

private:

    M_type initialize_density()
    {
        M_type n_particles = 0;

        /// initialize rho(x,y,t=0) = sin(pi*x)*sin(pi*y)
        /// and m(x,y,t=0) = fac_ * rho(x,y,t=0)
        for (size_type i = 1; i < N_-1; ++i) {
            for (size_type j = 1; j < N_-1; ++j) {
                rho_[i*N_ + j] = sin(M_PI*i*dh_) * sin(M_PI*j*dh_);
                m_[i*N_ + j] = lround(rho_[i*N_ + j] * fac_ * dh_*dh_);
                m_tmp_[i*N_ + j] = m_[i*N_ + j];
                n_particles += m_[i*N_ + j];
            }
        }

        return n_particles;
    }

    size_type N_, Ntot, n_step_;
    M_type M_, M_real_;

    value_type D_, dh_, dt_, lambda_, fac_, rms_error_, t_max_;

    std::vector<value_type> rho_;
    std::vector<particle_type> m_, m_tmp_;
    std::vector<omp_mutex> locks_;

    static VSLStreamStatePtr stream;
    #pragma omp threadprivate(stream)
};

VSLStreamStatePtr Diffusion2D::stream;


int main(int argc, char* argv[])
{
    timer t_total;
    t_total.start();

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " D N M dt (n_runs) (n_steps)" << std::endl;
        return 1;
    }

    const value_type D  = std::stod (argv[1]);
    const size_type  N  = std::stoul(argv[2]);
    const M_type     M  = std::stoul(argv[3]);
    const value_type dt = std::stod (argv[4]);

//    if ( N % 2 != 0 ) {
//        N++;
//        std::cout << "Warning: N must be a multiple of 2. Using N = "
//                  << N << " instead." << '\n';
//    }
//
//    if ( (N-2) % 8 != 0 ) {
//        N += 8 - ((N-2) % 8);
//        std::cout << "Warning: (N-2) must be a multiple of 8 for AVX. Using N = "
//                  << N << " instead." << '\n';
//    }

    value_type t_max;
    size_type n_runs;

    if (argc > 5) {
        n_runs = std::stoul(argv[5]);
    } else {
        n_runs = 1;
    }

    if (argc > 6) {
        t_max = std::stoul(argv[6]) * dt;
    } else {
        t_max = 0.1;
    }

    std::cout << "Running RW AVX Simulations" << '\n';
    std::cout << "N = " << N << '\t' << "dt = " << dt << '\t'
              << "M = " << M << std::endl;

    myInt64 min_cycles = 0;
    value_type final_time;
    M_type M_initial, M_final;

    std::vector<value_type> e_rms;
    e_rms.resize(n_runs, 0.0);

    srand(42);

    for(size_type i = 0; i < n_runs; i++) {
        Diffusion2D system(D, N, M, dt, rand(), t_max);

#ifdef USE_TIMER
        timer t;
        t.start();
#endif // USE_TIMER

#ifdef USE_TSC
        myInt64 start, cycles;
        start = start_tsc();
#endif // USE_TSC

        system.run_simulation();

#ifdef USE_TIMER
        t.stop();
//        std::cout << "Timing: " << N << " " << t.get_timing() << std::endl;
#endif // USE_TIMER

#ifdef USE_TSC
        cycles = stop_tsc(start);
//        std::cout << "Cycles = " << cycles << std::endl;
#endif // USE_TSC

        system.compute_density();
        e_rms[i] = system.compute_rms_error();

        if ( (e_rms[i] < 0.1) && ( (cycles < min_cycles) || (min_cycles == 0) ) ) {
            min_cycles = cycles;
        }

        if ( i == n_runs-1 ) {
            final_time = system.time();
            system.write_density("Solutions/RW_AVX.dat");
            system.write_reference("Solutions/RW_ref.dat");
            M_initial = system.M_real();
            M_final = system.compute_num_particles();
        }
    }

    size_type n = e_rms.size() / 2;
    std::nth_element(e_rms.begin(), e_rms.begin()+n, e_rms.end());
    value_type median_rms = e_rms[n];
    if( e_rms.size() % 2 == 0 )
    {
        std::nth_element(e_rms.begin(), e_rms.begin()+n-1, e_rms.end());
        median_rms = 0.5*(e_rms[n-1] + median_rms);
    }

    std::cout << "Actual initial # of particles = " << M_initial << '\n';
    std::cout << "Actual  final  # of particles = " << M_final << '\n';
    std::cout << "Median RMS Error = " << median_rms << '\n';
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
