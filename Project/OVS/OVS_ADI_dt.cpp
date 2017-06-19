int main(int argc, char* argv[])
{
    std::cout << "Beginning OVS for time step..." << std:: endl;

    const value_type D  = 1;
    const size_type  N  = 32;
    value_type tmax = 0.1;

    timer t;

    std::ofstream OVS_file("OVS/OVS_ADI_dt.dat", std::ios::out);

    for (value_type dt = 0.001; dt > 0.000001; dt /= 1.2) {
        std::cout << "dt = " << dt;

        Diffusion2D system(D, N, dt);

        t.start();
        while (system.time() < tmax) {
            system.advance();
        }
        t.stop();

        system.write_reference("Solutions/ADI_ref.dat");

        std::cout << "\tRMS Error = " << system.rms_error() << std::endl;
        std::cout << "Timing: " << N << " " << t.get_timing() << std::endl;
        std::cout << std::endl;

        OVS_file << dt << " " << system.rms_error() << "\n";
    }

    OVS_file.close();

    return 0;
}
