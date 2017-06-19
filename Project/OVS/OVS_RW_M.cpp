int main(int argc, char* argv[])
{
    std::cout << "Beginning OVS for number of particles..." << std:: endl;

    const value_type D  = 1;
    const value_type dt = 0.0002;
    const size_type  N  = 32;
    value_type tmax = 0.1;

    timer t;

    std::ofstream OVS_file("OVS/OVS_RW_M.dat", std::ios::out);

    for (size_type M = 10000; M < 10000000; M *= 2) {
        std::cout << "M = " << M << std::flush;

        Diffusion2D system(D, N, M, dt);

        t.start();
        while (system.time() < tmax) {
            system.advance();
        }
        t.stop();

        system.compute_density();
        system.write_reference("Solutions/RW_ref.dat");

        std::cout << "\tRMS Error = " << system.rms_error() << std::endl;
        std::cout << "Timing: " << N << " " << t.get_timing() << std::endl;
        std::cout << std::endl;

        OVS_file << M << " " << system.rms_error() << "\n";
    }

    OVS_file.close();

    return 0;
}
