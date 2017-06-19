int main(int argc, char* argv[])
{
    std::cout << "Beginning OVS for grid spacing..." << std:: endl;

    const value_type D  = 1;
    value_type dt = 0.00000005;
    value_type tmax = 0.1;

    timer t;

    std::ofstream OVS_file("OVS/OVS_ADI_dh.dat", std::ios::out);

    for (size_type N = 4; N <= 256; N *= 2) {
        std::cout << "N = " << N;

        Diffusion2D system(D, N, dt);

        t.start();
        while (system.time() < tmax) {
            system.advance();
        }
        t.stop();

        system.write_reference("Solutions/ADI_ref.dat");

        std::cout << "\tRMS Error = " << system.rms_error() << std::endl;
        std::cout << "Timing: " << " " << t.get_timing() << "\t";
        std::cout << "CFL = " << system.CFL() << std::endl;
        std::cout << std::endl;

        OVS_file << N << " " << system.rms_error() << "\n";
    }

    OVS_file.close();

    return 0;
}
