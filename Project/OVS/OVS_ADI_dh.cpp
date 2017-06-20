int main(int argc, char* argv[])
{
    std::cout << "Beginning OVS for grid spacing..." << std:: endl;

    const value_type D  = 1;
    const value_type dt = 0.00000001;
    const value_type tmax = 0.1;
    const size_type  N[7] = {6,10,22,42,82,162,322};

    std::ofstream OVS_file("OVS/OVS_ADI_dh.dat", std::ios::out);

    for (size_type i = 0; i <= sizeof(N); i++) {
        std::cout << "N = " << N[i] << std::endl;

        Diffusion2D system(D, N[i], dt);

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
    std::cout << "Timing: " << " " << t.get_timing() << std::endl;
#endif // USE_TIMER

#ifdef USE_TSC
    cycles = stop_tsc(start);
    std::cout << "Cycles = " << cycles << std::endl;
#endif // USE_TSC

        system.write_reference("Solutions/ADI_ref.dat");

        std::cout << "RMS Error = " << system.rms_error() << std::endl;
        std::cout << "CFL = " << system.CFL() << '\n' << std::endl;

        OVS_file << N[i] << '\t' << system.rms_error() << '\n';
    }

    OVS_file.close();

    return 0;
}
