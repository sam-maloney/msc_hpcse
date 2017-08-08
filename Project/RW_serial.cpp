#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>

#include "timer.hpp"
#include "prng_engine.hpp" // Sitmo PRNG

/// Select which runtime measure to use
//#define USE_TIMER
#define USE_TSC

#ifdef USE_TSC
#include "tsc_x86.hpp"
#endif // USE_TSC

typedef double value_type;
typedef std::size_t size_type;
typedef int32_t particle_type;
typedef uint64_t M_type;

#ifndef M_PI
    constexpr value_type M_PI = 3.14159265358979323846;
#endif // M_PI

class Diffusion2D {
public:
    Diffusion2D(const value_type D,
                const size_type N,
                const M_type M,
                const value_type dt)
    : D_(D), N_(N), Ntot(N_*N_), M_(M), dt_(dt)
    {
        /// real space grid spacing
        dh_ = 1.0 / (N_ - 1);

        /// lambda factor
        lambda_ = dt_*D_ / (dh_*dh_);
        if ( lambda_ >= 0.25 ) {
            lambda_ = 0.2;
            dt_ = lambda_*dh_*dh_ / D_;
            std::cout << "Time-step too large for given n, instead using dt = "
                      << dt_ << std::endl;
        }

        /// probability of particle staying in same position
        p_stay_ = 1 - 4.0*lambda_;

        /// conversion factor; m_ij = fac_ * rho_ij
        fac_ = M_*M_PI*M_PI/4.0;

        rho_.resize(Ntot, 0.0);
        m_.resize(Ntot, 0);
        m_tmp_.resize(Ntot, 0);

        eng0.seed(0);
        n_step_ = 0;

        M_real_ = initialize_density();
    }

    void advance()
    {
        m_tmp_ = m_;

        /// Dirichlet boundaries
        for(size_type i = 1; i < N_-1; ++i) {
            for(size_type j = 1; j < N_-1; ++j) {
                for(particle_type k = 0; k < m_[i*N_ + j]; k++) {
                    if ( static_cast<value_type>(eng0()) /
                         static_cast<value_type>(eng0.max()) <= p_stay_ ) {
                        continue; // does not move
                    }

                    --m_tmp_[i*N_ + j];
                    value_type direction = static_cast<value_type>(eng0()) /
                                           static_cast<value_type>(eng0.max());

                    if ( direction < 0.25 ) {
                        ++m_tmp_[i*N_ + j - 1]; // moves left
                    } else if ( direction < 0.5 ) {
                        ++m_tmp_[i*N_ + j + 1]; // moves right
                    } else if ( direction < 0.75 ) {
                        ++m_tmp_[(i-1)*N_ + j]; // moves up
                    } else {
                        ++m_tmp_[(i+1)*N_ + j]; // moves down
                    }
                }
            }
        }

        m_.swap(m_tmp_);

        for(size_type i = 1; i < N_-1; ++i) {
            for(size_type j = 1; j < N_-1; ++j) {
                rho_[i*N_ + j] = static_cast<value_type>(m_[i*N_ + j]) / (fac_*dh_*dh_);
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
                m_[i*N_ + j] = static_cast<size_type>(rho_[i*N_ + j]*fac_*dh_*dh_);
				n_particles += m_[i*N_ + j];
            }
        }
		
		return n_particles;
    }

    value_type D_;
    size_type N_, Ntot, n_step_;
	M_type M_, M_real_;

    value_type dh_, dt_, lambda_, fac_, p_stay_, rms_error_;

    std::vector<value_type> rho_;
    std::vector<particle_type> m_, m_tmp_;

    sitmo::prng_engine eng0;
};

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

//    if ( N % 8 != 0 ) {
//        N += 8 - (N % 8);
//        std::cout << "N must be a multiple of 8 to prevent false sharing. Using N = "
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

    std::cout << "Running RW Serial Simulations" << '\n';
    std::cout << "N = " << N << '\t' << "dt = " << dt << '\t'
              << "M = " << M << std::endl;

    myInt64 min_cycles = 0;
    value_type e_rms, final_time;
	M_type M_initial = 0;

    for(size_type i = 0; i < n_runs; i++) {
        Diffusion2D system(D, N, M, dt);

#ifdef USE_TIMER
        timer t;
        t.start();
#endif // USE_TIMER

#ifdef USE_TSC
        myInt64 start, cycles;
        start = start_tsc();
#endif // USE_TSC

        while ( system.time() < t_max ) {
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

        if ( (e_rms < 1) && ( (cycles < min_cycles) || (min_cycles == 0) ) ) {
            min_cycles = cycles;
        }

        if ( i == n_runs-1 ) {
            final_time = system.time();
			M_initial = system.M_real();
            system.write_density("Solutions/RW_scalar.dat");
            system.write_reference("Solutions/RW_ref.dat");
        }
    }
	
	std::cout << "Actual # of particles = " << M_initial << std::endl;

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

