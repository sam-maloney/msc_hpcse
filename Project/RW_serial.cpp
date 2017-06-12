
#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>
#include <random>

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
        /// real space grid spacing
        dh_ = 1.0 / (N_ - 1);

        /// lambda factor and probability of particle staying in same position
        lambda_ = dt_*D_ / (dh_*dh_);
        p_stay_ = 1 - 4.0*lambda_;

        /// conversion factor; m_ij = fac_ * rho_ij
        fac_ = M_*M_PI*M_PI/4.0;

        rho_.resize(Ntot, 0.0);

        initialize_density();
    }

    void advance()
    {
        /// Dirichlet boundaries

        for(size_type i = 1; i < N_-1; ++i) {
            for(size_type j = 1; j < N_-1; ++j) {

            }
        }

//        /// use swap instead of rho_=rho_half. this is much more efficient,
//        /// because it does not have to copy element by element.
//        std::swap(rho_half, rho_);
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

    void write_reference(std::string const& filename, value_type t_f) const
    {
        std::ofstream out_file(filename, std::ios::out);

        for(size_type i = 0; i < N_; ++i) {
            for(size_type j = 0; j < N_; ++j)
                out_file << (i*dh_ - 0.5) << '\t' << (j*dh_ - 0.5) << '\t'
                         << sin(M_PI*i*dh_) * sin(M_PI*j*dh_) *
                            exp(-2*D_*t_f*M_PI*M_PI)
                         << "\n";
            out_file << "\n";
        }
        out_file.close();
    }

private:

    void initialize_density()
    {
        /// initialize rho(x,y,t=0) = sin(pi*x)*sin(pi*y)
        /// and m(x,y,t=0) = fac_ * rho(x,y,t=0)
        for (size_type i = 0; i < N_; ++i) {
            for (size_type j = 0; j < N_; ++j) {
                rho_[i*N_ + j] = sin(M_PI*i*dh_) * sin(M_PI*j*dh_);
                m_[i*N_ + j] = rho_[i*N_ + j] * fac_;
            }
        }
    }

    value_type D_;
    size_type N_, Ntot, M_;

    value_type dh_, dt_, lambda_, fac_, p_stay_;

    std::vector<value_type> rho_, m_;
};


int main(int argc, char* argv[])
{
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " D N M dt (n_steps)" << std::endl;
        return 1;
    }

    const value_type D  = std::stod(argv[1]);
    const size_type  N  = std::stoul(argv[2]);
    const size_type  M  = std::stoul(argv[3]);
    const value_type dt = std::stod(argv[4]);


    Diffusion2D system(D, N, M, dt);
    system.write_density("Solutions/RW_000.dat");

    value_type time = 0;
    value_type tmax = 10000 * dt;

    if (argc > 5) {
        tmax = std::stoul(argv[5]) * dt;
    }

    timer t;

    t.start();
    while (time < tmax) {
        system.advance();
        time += dt;
    }
    t.stop();

    std::cout << "Timing : " << N << " " << 1 << " " << t.get_timing() << std::endl;

    system.write_density("Solutions/RW_serial.dat");
    system.write_reference("Solutions/RW_ref.dat", time);

    return 0;
}
