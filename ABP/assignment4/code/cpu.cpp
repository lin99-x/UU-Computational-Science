#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>

// template <typename Number>;

int main(int argc, char* argv[])
{
    int nrepeat = -1;
    int num_elements = -1;

    for (int i=1; i<argc; i++)
    {
        if(strcmp(argv[i], "-N") == 0) {
            num_elements = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-repeat") == 0) {
            nrepeat = atoi(argv[++i]);
        }
    }

    Kokkos::initialize(argc, argv);
    {
    #ifdef KOKKOS_ENABLE_CUDA
    #define MemSpace Kokkos::CudaSpace
    #else
    #define MemSpace Kokkos::HostSpace
    #endif

    using Number = float;
    using LayoutType = Kokkos::LayoutLeft;
    using ExecutionSpace = MemSpace::execution_space;
    using range_policy = Kokkos::RangePolicy<ExecutionSpace>;
    using namespace Kokkos;

    // Allocate J, A on device
    Kokkos::View<Number*[3][3], LayoutType, MemSpace> J("J", num_elements);
    Kokkos::View<Number*[4][4], LayoutType, MemSpace> A("A", num_elements);

    // create host mirrors of device views
    Kokkos::View<Number*[3][3], LayoutType, MemSpace>::HostMirror h_J = Kokkos::create_mirror_view(J);
    Kokkos::View<Number*[4][4], LayoutType, MemSpace>::HostMirror h_A = Kokkos::create_mirror_view(A);  

    // Populate J_i with the given J_iacobian matrix for each element on host.
    Kokkos::parallel_for("Populate J_iacobian matrix", Kokkos::RangePolicy<Kokkos::OpenMP>(0, num_elements), KOKKOS_LAMBDA (int i)
    {
        h_J(i, 0, 0) = 3;
        h_J(i, 0, 1) = 1;
        h_J(i, 0, 2) = 1;
        h_J(i, 1, 0) = 1;
        h_J(i, 1, 1) = 3;
        h_J(i, 1, 2) = 1;
        h_J(i, 2, 0) = 1;
        h_J(i, 2, 1) = 1;
        h_J(i, 2, 2) = 3;
    });
    Kokkos::fence();
    
    // Timer for transfer time host to device
    Kokkos::Timer timer;
    // Deep copy host views to device views
    Kokkos::deep_copy(J, h_J);
    // Compute transfer time host to device
    double host_to_dev = timer.seconds();
    // printf("Host to device transfer time: %f\n", host_to_dev);
    // Timer for computation time
    Kokkos::Timer timer2;

    for (int repeat = 0; repeat < nrepeat; repeat++) {
        // Compute element matrix
        Kokkos::parallel_for("compute element matrix", range_policy(0, num_elements), KOKKOS_LAMBDA (int i)
        {
            auto J_i = Kokkos::subview(J, i, Kokkos::ALL, Kokkos::ALL);
            auto A_i = Kokkos::subview(A, i, Kokkos::ALL, Kokkos::ALL);

            Number C0 = J_i(1, 1) * J_i(2, 2) - J_i(1, 2) * J_i(2, 1);
            Number C1 = J_i(1, 2) * J_i(2, 0) - J_i(1, 0) * J_i(2, 2);
            Number C2 = J_i(1, 0) * J_i(2, 1) - J_i(1, 1) * J_i(2, 0);
            Number inv_J_i_det = J_i(0, 0) * C0 + J_i(0, 1) * C1 + J_i(0, 2) * C2;
            Number d = (1. / 6.) / inv_J_i_det;
            Number G0 = d * (J_i(0, 0) * J_i(0, 0) + J_i(1, 0) * J_i(1, 0) + J_i(2, 0) * J_i(2, 0));
            Number G1 = d * (J_i(0, 0) * J_i(0, 1) + J_i(1, 0) * J_i(1, 1) + J_i(2, 0) * J_i(2, 1));
            Number G2 = d * (J_i(0, 0) * J_i(0, 2) + J_i(1, 0) * J_i(1, 2) + J_i(2, 0) * J_i(2, 2));
            Number G3 = d * (J_i(0, 1) * J_i(0, 1) + J_i(1, 1) * J_i(1, 1) + J_i(2, 1) * J_i(2, 1));
            Number G4 = d * (J_i(0, 1) * J_i(0, 2) + J_i(1, 1) * J_i(1, 2) + J_i(2, 1) * J_i(2, 2));
            Number G5 = d * (J_i(0, 2) * J_i(0, 2) + J_i(1, 2) * J_i(1, 2) + J_i(2, 2) * J_i(2, 2));

            A_i(0, 0) = G0;
            A_i(0, 1) = A_i(1, 0) = G1;
            A_i(0, 2) = A_i(2, 0) = G2;
            A_i(0, 3) = A_i(3, 0) = -G0 - G1 - G2;
            A_i(1, 1) = G3;
            A_i(1, 2) = A_i(2, 1) = G4;
            A_i(1, 3) = A_i(3, 1) = -G1 - G3 - G4;
            A_i(2, 2) = G5;
            A_i(2, 3) = A_i(3, 2) = -G2 - G4 - G5;
            A_i(3, 3) = G0 + 2 * G1 + 2 * G2 + G3 + 2 * G4 + G5;
        });
        #ifdef KOKKOS_ENABLE_CUDA
        cudaDeviceSynchronize();
        #endif
    //     // if (repeat == nrepeat - 1) {
    //     //     // Timer for transfer time device to host
    //     //     Kokkos::Timer timer3;
    //     //     // Deep copy device views to host views
    //     //     Kokkos::deep_copy(h_A, A);
    //     //     // Compute transfer time device to host
    //     //     double dev_to_host = timer3.seconds();
    //     //     // print the result
    //     //     // for (int i = 0; i < num_elements; i++) {
    //     //     //     printf("Element %d:\n", i);
    //     //     //     for (int j = 0; j < 4; j++) {
    //     //     //         printf("%f %f %f %f\n", h_A(i, j, 0), h_A(i, j, 1), h_A(i, j, 2), h_A(i, j, 3));
    //     //     //     }
    //     //     // }
    //     // }

    }
    Kokkos::fence();
    // Compute computation time
    double computation = timer2.seconds();
    // printf("Computation time: %f\n", computation / nrepeat);

    // Timer for transfer time device to host
    Kokkos::Timer timer3;
    // Deep copy device views to host views
    Kokkos::deep_copy(h_A, A);
    // Compute transfer time device to host
    double dev_to_host = timer3.seconds();

    double Gbytes = 1.0e-9 * double(num_elements * (3*3 + 4*4) * sizeof(Number));
    double GFlop = 1.0e-9 * 84.0 * double(num_elements);
    double mecps = 1e-6 * (static_cast<double>(num_elements) / (computation / nrepeat));
    std::cout << "Number of elements;" << num_elements << ";" 
              << "Transfer time host to device;" << host_to_dev << ";"
              << "Average calculation time;" << computation / nrepeat << ";"
              << "Transfer time device to host;" << dev_to_host << ";"
              << "Million element computed per second;" << mecps << ";"
              << "Floating point operations;" << GFlop / (computation / nrepeat) << ";"
              << "Bandwidth achieved;" << Gbytes / (computation / nrepeat) << std::endl;
    }

    Kokkos::finalize();
    return 0;
}