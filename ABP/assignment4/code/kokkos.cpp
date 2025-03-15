#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>

template <typename Number, typename LayoutType, typename MemSpace, typename ExecutionSpace, typename range_policy>
void compute_element_matrix(const Kokkos::View<Number*[3][3], LayoutType, MemSpace> &J, Kokkos::View<Number*[4][4], LayoutType, MemSpace> &A, int num_elements)
{
    Kokkos::parallel_for("compute element matrix", range_policy(0, num_elements), KOKKOS_LAMBDA (int i)
    {
        Number C0 = J(i, 1, 1) * J(i, 2, 2) - J(i, 1, 2) * J(i, 2, 1);
        Number C1 = J(i, 1, 2) * J(i, 2, 0) - J(i, 1, 0) * J(i, 2, 2);
        Number C2 = J(i, 1, 0) * J(i, 2, 1) - J(i, 1, 1) * J(i, 2, 0);
        Number inv_J_det = J(i, 0, 0) * C0 + J(i, 0, 1) * C1 + J(i, 0, 2) * C2;
        Number d = (1. / 6.) / inv_J_det;
        Number G0 = d * (J(i, 0, 0) * J(i, 0, 0) + J(i, 1, 0) * J(i, 1, 0) + J(i, 2, 0) * J(i, 2, 0));
        Number G1 = d * (J(i, 0, 0) * J(i, 0, 1) + J(i, 1, 0) * J(i, 1, 1) + J(i, 2, 0) * J(i, 2, 1));
        Number G2 = d * (J(i, 0, 0) * J(i, 0, 2) + J(i, 1, 0) * J(i, 1, 2) + J(i, 2, 0) * J(i, 2, 2));
        Number G3 = d * (J(i, 0, 1) * J(i, 0, 1) + J(i, 1, 1) * J(i, 1, 1) + J(i, 2, 1) * J(i, 2, 1));
        Number G4 = d * (J(i, 0, 1) * J(i, 0, 2) + J(i, 1, 1) * J(i, 1, 2) + J(i, 2, 1) * J(i, 2, 2));
        Number G5 = d * (J(i, 0, 2) * J(i, 0, 2) + J(i, 1, 2) * J(i, 1, 2) + J(i, 2, 2) * J(i, 2, 2));

        A(i, 0, 0) = G0;
        A(i, 0, 1) = A(i, 1, 0) = G1;
        A(i, 0, 2) = A(i, 2, 0) = G2;
        A(i, 0, 3) = A(i, 3, 0) = -G0 - G1 - G2;
        A(i, 1, 1) = G3;
        A(i, 1, 2) = A(i, 2, 1) = G4;
        A(i, 1, 3) = A(i, 3, 1) = -G1 - G3 - G4;
        A(i, 2, 2) = G5;
        A(i, 2, 3) = A(i, 3, 2) = -G2 - G4 - G5;
        A(i, 3, 3) = G0 + 2 * G1 + 2 * G2 + G3 + 2 * G4 + G5;
    });
}

template <typename Number, typename LayoutType, typename MemSpace, typename ExecutionSpace, typename range_policy>
void benchmark(int num_elements, int nrepeat)
{
    typedef Kokkos::View<Number*[3][3], LayoutType, MemSpace> TypeJ;
    typedef Kokkos::View<Number*[4][4], LayoutType, MemSpace> TypeA;

    TypeJ J("J", num_elements);
    TypeA A("A", num_elements);
    typename TypeJ::HostMirror h_J = Kokkos::create_mirror_view(J);
    typename TypeA::HostMirror h_A = Kokkos::create_mirror_view(A);

    for (int i=0; i<num_elements; i++)
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
    }

    Kokkos::Timer timer;
    Kokkos::deep_copy(J, h_J);
    Kokkos::fence();
    double host_to_dev = timer.seconds();

    timer.reset();
    Kokkos::Timer timer2;
    for (int i=0; i<nrepeat; i++)
    {
        compute_element_matrix<Number, LayoutType, MemSpace, ExecutionSpace, range_policy>(J, A, num_elements);
    }
    Kokkos::fence();
    double compute_time = timer2.seconds();

    timer.reset();
    Kokkos::Timer timer3;
    Kokkos::deep_copy(h_A, A);
    Kokkos::fence();
    double dev_to_host = timer3.seconds();

    // for (int i = 0; i < num_elements; i++) {
    //     printf("Element %d:\n", i);
    //     for (int j = 0; j < 4; j++) {
    //         printf("%f %f %f %f\n", h_A(i, j, 0), h_A(i, j, 1), h_A(i, j, 2), h_A(i, j, 3));
    //     }
    // }

    double Gbytes = 1.0e-9 * double(num_elements * (3*3 + 4*4) * sizeof(Number));
    double GFlop = 1.0e-9 * 84.0 * double(num_elements);
    double mecps = 1e-6 * (static_cast<double>(num_elements) / (compute_time / nrepeat));

    std::cout << "Number of elements;" << num_elements << ";" 
            << "Transfer time host to device;" << host_to_dev << ";"
            << "Average calculation time;" << compute_time / nrepeat << ";"
            << "Transfer time device to host;" << dev_to_host << ";"
            << "Million element computed per second;" << mecps << ";"
            << "Floating point operations;" << GFlop / (compute_time / nrepeat) << ";"
            << "Bandwidth achieved;" << Gbytes / (compute_time / nrepeat) << std::endl;
    
}

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
        #define ExecutionSpace Kokkos::Cuda
        #define range_policy Kokkos::RangePolicy<ExecutionSpace>
        #else
        #define MemSpace Kokkos::HostSpace
        #define ExecutionSpace Kokkos::OpenMP
        #define range_policy Kokkos::RangePolicy<ExecutionSpace>
        #endif

        // using ExecutionSpace = MemSpace::execution_space;
        // using range_policy = Kokkos::RangePolicy<ExecutionSpace>;

        benchmark<float, Kokkos::LayoutRight, MemSpace, ExecutionSpace, range_policy>(num_elements, nrepeat);
    }

    Kokkos::finalize();
    return 0;
}