int main(int argc, char* argv[])
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " D N dt (tmax)" << std::endl;
        return 1;
    }

    const value_type D  = std::stod (argv[1]);
    const size_type  N  = std::stoul(argv[2]);
    const value_type dt = std::stod (argv[3]);


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
