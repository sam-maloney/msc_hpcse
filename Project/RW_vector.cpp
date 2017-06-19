
#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>
#include <random>

#include "prng_engine.hpp" // Sitmo PRNG
#include "timer.hpp"

typedef double value_type;
typedef std::size_t size_type;

#ifndef M_PI
    constexpr value_type M_PI = 3.14159265358979323846;
#endif // M_PI

class Diffusion2D {
public:
    Diffusion2D(const value_type D,
                const size_type N,
                const size_type M,
                const value_type dt)
    : D_(D), N_(N), Ntot(N_*N_), M_(M), dt_(dt)
    {
        prng_eng.seed(0);

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
        m_tmp.resize(Ntot, 0);

        n_step_ = 0;

        initialize_density();
    }

    void advance()
    {
        m_tmp = m_;

        /// Dirichlet boundaries
        for(size_type i = 1; i < N_-1; ++i) {
            for(size_type j = 1; j < N_-1; ++j) {
                std::binomial_distribution<> d(m_[i*N_ + j], lambda_);
                size_type n_left  = d(prng_eng);
                size_type n_right = d(prng_eng);
                size_type n_up    = d(prng_eng);
                size_type n_down  = d(prng_eng);

                m_tmp[i*N_ + j - 1] += n_left;
                m_tmp[i*N_ + j + 1] += n_right;
                m_tmp[(i-1)*N_ + j] += n_up;
                m_tmp[(i+1)*N_ + j] += n_down;
                m_tmp[i*N_ + j] -= (n_left + n_right + n_up + n_down);
            }
        }

        m_.swap(m_tmp);
        n_step_++;
    }

    void compute_density()
    {
        for(size_type i = 1; i < N_-1; ++i) {
            for(size_type j = 1; j < N_-1; ++j) {
                rho_[i*N_ + j] = static_cast<value_type>(m_[i*N_ + j]) / fac_;
            }
        }
    }

    void write_density(std::string const& filename) const
    {
        std::ofstream out_file(filename, std::ios::out);

        for(size_type i = 0; i < N_; ++i) {
            for(size_type j = 0; j < N_; ++j) {
                out_file << (i*dh_ - 0.5) << '\t' << (j*dh_ - 0.5) << '\t' << rho_[i*N_ + j] << "\n";
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

private:

    void initialize_density()
    {
        /// initialize rho(x,y,t=0) = sin(pi*x)*sin(pi*y)
        /// and m(x,y,t=0) = fac_ * rho(x,y,t=0)
        for (size_type i = 0; i < N_; ++i) {
            for (size_type j = 0; j < N_; ++j) {
                rho_[i*N_ + j] = sin(M_PI*i*dh_) * sin(M_PI*j*dh_);
                m_[i*N_ + j] = static_cast<size_type>(rho_[i*N_ + j] * fac_);
            }
        }
    }

    value_type D_;
    size_type N_, Ntot, M_, n_step_;

    value_type dh_, dt_, lambda_, fac_, rms_error_;

    std::vector<value_type> rho_;
    std::vector<size_type> m_, m_tmp;

    sitmo::prng_engine prng_eng;
};

int main(int argc, char* argv[])
{
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " D N M dt (tmax)" << std::endl;
        return 1;
    }

    const value_type D  = std::stod(argv[1]);
    const size_type  N  = std::stoul(argv[2]);
    const size_type  M  = std::stoul(argv[3]);
    const value_type dt = std::stod(argv[4]);


    Diffusion2D system(D, N, M, dt);
    system.write_density("Solutions/RW_000.dat");

    value_type tmax;

    if (argc > 5) {
        tmax = std::stoul(argv[5]);
    } else {
        tmax = 0.1;
    }

    timer t;

    t.start();
    while (system.time() < tmax) {
        system.advance();
    }
    t.stop();

    std::cout << "Timing : " << N << " " << 1 << " " << t.get_timing() << std::endl;
    std::cout << "CFL # = " << system.CFL() << std::endl;

    system.compute_density(); // for scalar and later code
    system.write_density("Solutions/RW_serial.dat");
    system.write_reference("Solutions/RW_ref.dat");

    std::cout << "RMS Error = " << system.rms_error() << std::endl;

    return 0;
}
