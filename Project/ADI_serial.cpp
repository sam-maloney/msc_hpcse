#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>

//#define USE_TIMER
#define USE_TSC

#ifdef USE_TIMER
#include "timer.hpp"
#endif // USE_TIMER

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

        rho_.resize(Ntot, 0.0);
        rho_half.resize(Ntot, 0.0);

        n_step_ = 0;

        initialize_density();
        initialize_thomas();
    }

    void advance()
    {
        /// Dirichlet boundaries; central differences in space

        /// For each row, apply Thomas algorithm for implicit solution
        for(size_type i = 1; i < N_-1; ++i) {
            /// First is the forward sweep in x direction
            d_[1] = -c_[1]*rho_[(i-1)*N_ + 1] + (1-2*fac_)/(1+2*fac_) *
                    rho_[i*N_ + 1] - c_[1]*rho_[(i+1)*N_ + 1];
            for(size_type k = 2; k < N_-1; k++) {
                d_[k] = ( fac_*rho_[(i-1)*N_ + k] +
                          (1-2*fac_)*rho_[i*N_ + k] +
                          fac_*rho_[(i+1)*N_ + k] +
                          fac_*d_[k-1] ) /
                         ( 1 + 2*fac_ + fac_*c_[k-1] );
            }
            /// Second is the back substitution for the half time step
            rho_half[i*N_ + N_ - 2] = d_[N_ - 2];
            for(size_type k = N_-3; k > 0; k--) {
                rho_half[i*N_ + k] = d_[k] - c_[k]*rho_half[i*N_ + k + 1];
            }
        }

        /// For each column, apply Thomas algorithm for implicit solution
        for(size_type j = 1; j < N_-1; ++j) {
            /// First is the forward sweep in y direction
            d_[1] = -c_[1]*rho_half[N_ + j - 1] + (1-2*fac_)/(1+2*fac_) *
                    rho_half[N_ + j] - c_[1]*rho_half[N_ + j + 1];
            for(size_type k = 2; k < N_-1; k++) {
                d_[k] = ( fac_*rho_half[k*N_ + j - 1] +
                          (1-2*fac_)*rho_half[k*N_ + j] +
                          fac_*rho_half[k*N_ + j + 1] +
                          fac_*d_[k-1] ) /
                         ( 1 + 2*fac_ + fac_*c_[k-1] );
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
            for(size_type j = 0; j < N_; ++j)
                out_file << (i*dh_ - 0.5) << '\t' << (j*dh_ - 0.5) << '\t' << rho_[i*N_ + j] << "\n";
            out_file << "\n";
        }
        out_file.close();
    }

    void write_reference(std::string const& filename)
    {
        std::ofstream out_file(filename, std::ios::out);

        value_type ref_value, t_f;
        t_f = time();
        rms_error_ = 0.0;

        for(size_type i = 0; i < N_; ++i) {
            for(size_type j = 0; j < N_; ++j) {
                ref_value = sin(M_PI*i*dh_) * sin(M_PI*j*dh_) *
                            exp(-2*D_*t_f*M_PI*M_PI);
                rms_error_ += pow(rho_[i*N_ + j] - ref_value, 2);
                out_file << (i*dh_ - 0.5) << '\t' << (j*dh_ - 0.5) << '\t'
                         << ref_value << "\n";
            }
            out_file << "\n";
        }
        rms_error_ = sqrt(rms_error_/(N_*N_));
        out_file.close();
    }

    value_type rms_error()
    {
        return rms_error_;
    }

    value_type CFL()
    {
        return fac_;
    }

    value_type time()
    {
        return n_step_ * dt_;
    }

    size_type time_step()
    {
        return n_step_;
    }

    value_type dt()
    {
        return dt_;
    }

private:

    void initialize_density()
    {
        /// initialize rho(x,y,t=0) = sin(pi*x)*sin(pi*y)
        for (size_type i = 0; i < N_; ++i) {
            for (size_type j = 0; j < N_; ++j) {
                rho_[i*N_ + j] = sin(M_PI*i*dh_) * sin(M_PI*j*dh_);
            }
        }
    }

    void initialize_thomas()
    {
        c_.resize(N_, 0.0);
        d_.resize(N_, 0.0);

        c_[1] = -fac_ / (1.0 + 2.0*fac_);
        for(size_type i = 2; i < N_-2; i++) {
            c_[i] = -fac_ / (1.0 + 2.0*fac_ - fac_*c_[i-1]);
        }
    }

    value_type D_;
    size_type N_, Ntot, n_step_;

    value_type dh_, dt_, fac_, rms_error_;

    std::vector<value_type> rho_, rho_half, c_, d_;
};

int main(int argc, char* argv[])
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " D N dt (tmax)" << std::endl;
        return 1;
    }

    const value_type D  = std::stod(argv[1]);
    const size_type  N  = std::stoul(argv[2]);
    const value_type dt = std::stod(argv[3]);


    Diffusion2D system(D, N, dt);
    system.write_density("Solutions/ADI_000.dat");

    value_type tmax ;

    if (argc > 4) {
        tmax = std::stoul(argv[5]);
    } else {
        tmax = 0.1;
    }

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
    std::cout << "Timing: " << N << " " << t.get_timing() << std::endl;
#endif // USE_TIMER

#ifdef USE_TSC
    cycles = stop_tsc(start);
    std::cout << "Cycles = " << cycles << std::endl;
#endif // USE_TSC

    std::cout << "CFL # = " << system.CFL() << std::endl;

    system.write_density("Solutions/ADI_serial.dat");
    system.write_reference("Solutions/ADI_ref.dat");

    std::cout << "RMS Error = " << system.rms_error() << std::endl;

    return 0;
}
