#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <iomanip>

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
        d_.resize(8*N_, 0.0);

        n_step_ = 0;

        initialize_density();
        initialize_thomas();
    }

    void run_simulation(value_type t_max)
    {
        /// Dirichlet boundaries; central differences in space

        while( time() < t_max )
        {

        value_type c1 = -c_[1];
        size_type i;

        /// For each row, apply Thomas algorithm for implicit solution
        /// Loop unrolled by 8 for data reuse and preparation for AVX
        for(i = 1; i < N_-8; i += 8) {
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
            d_[9 ] = c1*(tmp0               + f1_*tmp1) + c1*tmp2;
            d_[10] = c1*(tmp1               + f1_*tmp2) + c1*tmp3;
            d_[11] = c1*(tmp2               + f1_*tmp3) + c1*tmp4;
            d_[12] = c1*(tmp3               + f1_*tmp4) + c1*tmp5;
            d_[13] = c1*(tmp4               + f1_*tmp5) + c1*tmp6;
            d_[14] = c1*(tmp5               + f1_*tmp6) + c1*tmp7;
            d_[15] = c1*(tmp6               + f1_*tmp7) + c1*rho_[(i+8)*N_ + 1];

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

            rho_half[(i+1)*N_ - 2] = ( rho_[i*N_ - 2]  + f1_*tmp0 + tmp1 +
                                       d_[8*N_ - 24] ) * tmpf;
            rho_half[(i+2)*N_ - 2] = ( tmp0 + f1_*tmp1 + tmp2 +
                                       d_[8*N_ - 23] ) * tmpf;
            rho_half[(i+3)*N_ - 2] = ( tmp1 + f1_*tmp2 + tmp3 +
                                       d_[8*N_ - 22] ) * tmpf;
            rho_half[(i+4)*N_ - 2] = ( tmp2 + f1_*tmp3 + tmp4 +
                                       d_[8*N_ - 21] ) * tmpf;
            rho_half[(i+5)*N_ - 2] = ( tmp3 + f1_*tmp4 + tmp5 +
                                       d_[8*N_ - 20] ) * tmpf;
            rho_half[(i+6)*N_ - 2] = ( tmp4 + f1_*tmp5 + tmp6 +
                                       d_[8*N_ - 19] ) * tmpf;
            rho_half[(i+7)*N_ - 2] = ( tmp5 + f1_*tmp6 + tmp7 +
                                       d_[8*N_ - 18] ) * tmpf;
            rho_half[(i+8)*N_ - 2] = ( tmp6 + f1_*tmp7 + rho_[(i+9)*N_ - 2] +
                                       d_[8*N_ - 17] ) * tmpf;

            for(size_type k = N_-3; k > 0; k--) {
                value_type tmpc = c_[k];

                rho_half[(i  )*N_ + k] = d_[k*8    ] -
                                         tmpc*rho_half[(i  )*N_ + k + 1];
                rho_half[(i+1)*N_ + k] = d_[k*8 + 1] -
                                         tmpc*rho_half[(i+1)*N_ + k + 1];
                rho_half[(i+2)*N_ + k] = d_[k*8 + 2] -
                                         tmpc*rho_half[(i+2)*N_ + k + 1];
                rho_half[(i+3)*N_ + k] = d_[k*8 + 3] -
                                         tmpc*rho_half[(i+3)*N_ + k + 1];
                rho_half[(i+4)*N_ + k] = d_[k*8 + 4] -
                                         tmpc*rho_half[(i+4)*N_ + k + 1];
                rho_half[(i+5)*N_ + k] = d_[k*8 + 5] -
                                         tmpc*rho_half[(i+5)*N_ + k + 1];
                rho_half[(i+6)*N_ + k] = d_[k*8 + 6] -
                                         tmpc*rho_half[(i+6)*N_ + k + 1];
                rho_half[(i+7)*N_ + k] = d_[k*8 + 7] -
                                         tmpc*rho_half[(i+7)*N_ + k + 1];
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
        /// Loop unrolled by 8 for scalar replacement and preparation for AVX
        for(j = 1; j < N_-8; j += 8) {
            /// First is the forward sweep in y direction
            value_type tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmpf;

            tmp0 = rho_half[N_ + j    ];
            tmp1 = rho_half[N_ + j + 1];
            tmp2 = rho_half[N_ + j + 2];
            tmp3 = rho_half[N_ + j + 3];
            tmp4 = rho_half[N_ + j + 4];
            tmp5 = rho_half[N_ + j + 5];
            tmp6 = rho_half[N_ + j + 6];
            tmp7 = rho_half[N_ + j + 7];

            d_[8 ] = c1*(rho_half[N_ + j - 1] + f1_*tmp0) + c1*tmp1;
            d_[9 ] = c1*(tmp0                 + f1_*tmp1) + c1*tmp2;
            d_[10] = c1*(tmp1                 + f1_*tmp2) + c1*tmp3;
            d_[11] = c1*(tmp2                 + f1_*tmp3) + c1*tmp4;
            d_[12] = c1*(tmp3                 + f1_*tmp4) + c1*tmp5;
            d_[13] = c1*(tmp4                 + f1_*tmp5) + c1*tmp6;
            d_[14] = c1*(tmp5                 + f1_*tmp6) + c1*tmp7;
            d_[15] = c1*(tmp6                 + f1_*tmp7) + c1*rho_half[N_ + j + 8];

            for(size_type k = 2; k < N_-2; k++) {
                value_type tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmpf;

                tmp0 = rho_half[k*N_ + j    ];
                tmp1 = rho_half[k*N_ + j + 1];
                tmp2 = rho_half[k*N_ + j + 2];
                tmp3 = rho_half[k*N_ + j + 3];
                tmp4 = rho_half[k*N_ + j + 4];
                tmp5 = rho_half[k*N_ + j + 5];
                tmp6 = rho_half[k*N_ + j + 6];
                tmp7 = rho_half[k*N_ + j + 7];
                tmpf = c_rcp_[k];

                d_[k*8    ] = ( rho_half[k*N_ + j - 1] + f1_*tmp0 + tmp1 +
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
                d_[k*8 + 7] = ( tmp6 + f1_*tmp7 + rho_half[k*N_ + j + 8] +
                                d_[(k-1)*8 + 7] ) * tmpf;
            }
            /// Second is the back substitution for the full time step
            tmp0 = rho_half[(N_-2)*N_ + j];
            tmp1 = rho_half[(N_-2)*N_ + j + 1];
            tmp2 = rho_half[(N_-2)*N_ + j + 2];
            tmp3 = rho_half[(N_-2)*N_ + j + 3];
            tmp4 = rho_half[(N_-2)*N_ + j + 4];
            tmp5 = rho_half[(N_-2)*N_ + j + 5];
            tmp6 = rho_half[(N_-2)*N_ + j + 6];
            tmp7 = rho_half[(N_-2)*N_ + j + 7];
            tmpf = c_rcp_[N_-2];

            rho_[(N_-2)*N_ + j    ] = ( rho_half[(N_-2)*N_ + j - 1] + f1_*tmp0 +
                                        tmp1 + d_[8*N_ - 24] ) * tmpf;
            rho_[(N_-2)*N_ + j + 1] = ( tmp0 + f1_*tmp1 +
                                        tmp2 + d_[8*N_ - 23] ) * tmpf;
            rho_[(N_-2)*N_ + j + 2] = ( tmp1 + f1_*tmp2 +
                                        tmp3 + d_[8*N_ - 22] ) * tmpf;
            rho_[(N_-2)*N_ + j + 3] = ( tmp2 + f1_*tmp3 +
                                        tmp4 + d_[8*N_ - 21] ) * tmpf;
            rho_[(N_-2)*N_ + j + 4] = ( tmp3 + f1_*tmp4 +
                                        tmp5 + d_[8*N_ - 20] ) * tmpf;
            rho_[(N_-2)*N_ + j + 5] = ( tmp4 + f1_*tmp5 +
                                        tmp6 + d_[8*N_ - 19] ) * tmpf;
            rho_[(N_-2)*N_ + j + 6] = ( tmp5 + f1_*tmp6 +
                                        tmp7 + d_[8*N_ - 18] ) * tmpf;
            rho_[(N_-2)*N_ + j + 7] = ( tmp6 + f1_*tmp7 +
                                        rho_half[(N_-2)*N_ + j + 8] + d_[8*N_ - 17] ) * tmpf;

            for(size_type k = N_-3; k > 0; k--) {
                value_type tmpc = c_[k];

                rho_[k*N_ + j    ] = d_[k*8    ] - tmpc*rho_[(k + 1)*N_ + j    ];
                rho_[k*N_ + j + 1] = d_[k*8 + 1] - tmpc*rho_[(k + 1)*N_ + j + 1];
                rho_[k*N_ + j + 2] = d_[k*8 + 2] - tmpc*rho_[(k + 1)*N_ + j + 2];
                rho_[k*N_ + j + 3] = d_[k*8 + 3] - tmpc*rho_[(k + 1)*N_ + j + 3];
                rho_[k*N_ + j + 4] = d_[k*8 + 4] - tmpc*rho_[(k + 1)*N_ + j + 4];
                rho_[k*N_ + j + 5] = d_[k*8 + 5] - tmpc*rho_[(k + 1)*N_ + j + 5];
                rho_[k*N_ + j + 6] = d_[k*8 + 6] - tmpc*rho_[(k + 1)*N_ + j + 6];
                rho_[k*N_ + j + 7] = d_[k*8 + 7] - tmpc*rho_[(k + 1)*N_ + j + 7];
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

        } // while time < t_max
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
            c_[i] = -fac_ / (f2_ + fac_*c_[i-1]);
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
    std::cout << "Beginning OVS for time step..." << std:: endl;

    const value_type D  = 1;
    const size_type  N  = 256;
    value_type tmax = 1;

    timer t;

    std::ofstream OVS_file("OVS_ADI_dt.dat", std::ios::out);

    if ( OVS_file.is_open() ) {
        // Set highest possible precision
        OVS_file << std::setprecision(std::numeric_limits<long double>::digits10 + 1);

        for (value_type dt = 0.5; dt > 0.0005; dt /= sqrt(2)) {
            std::cout << "dt = " << dt;

            Diffusion2D system(D, N, dt);

            t.start();
            system.run_simulation(tmax);
            t.stop();

    //        system.write_reference("Solutions/ADI_ref.dat");

            system.compute_rms_error();

            std::cout << "\tRMS Error = " << system.rms_error() << std::endl;
            std::cout << "Timing: " << N << " " << t.get_timing() << std::endl;
            std::cout << std::endl;

            OVS_file << dt << " " << system.rms_error() << "\n";
            OVS_file.flush();
        }

        OVS_file.close();
    } else {
        std::cout << "Unable to open OVS file." << std::endl;
    }

    return 0;
}
