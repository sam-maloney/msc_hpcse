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

//    system.compute_density(); // for scalar and later code
    system.write_density("Solutions/RW_serial.dat");
    system.write_reference("Solutions/RW_ref.dat");

    std::cout << "RMS Error = " << system.rms_error() << std::endl;

    return 0;
}
