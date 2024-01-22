//To compile: !nvcc 1.Matrix_Multiplication.cu
// To run: !./a.out

#include <bits/stdc++.h>
using namespace std;

__global__ void matrix_multiplication(int *A, int *B, int *C, int N, int M, int P, int num_of_matrix, int threads_per_block)
{
    int thread_id = threadIdx.x;
    int start_index = thread_id * (num_of_matrix / threads_per_block);
    int end_index = (thread_id + 1) * (num_of_matrix / threads_per_block);

    if (thread_id == threads_per_block - 1)
        end_index = num_of_matrix;

    for (int x = start_index; x < end_index; x++)
    {
        int add_a = x * N * M;
        int add_b = x * M * P;
        int add_c = x * N * P;

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < P; j++)
            {
                int sum = 0;
                for (int k = 0; k < M; k++)
                    sum += A[add_a + i * M + k] * B[add_b + k * P + j];
                C[add_c + i * P + j] = sum;
            }
        }
    }
}

int main(int argc, char *argv[])
{
    int N, M, P, num_of_matrix;
    cout << "Enter the dimension of metrices: ";
    cin >> N >> M >> P;

    cout << "Enter the number of matrices: ";
    cin >> num_of_matrix;

    int *A, *B, *C;
    int element_of_a = num_of_matrix * N * M;
    int element_of_b = num_of_matrix * M * P;
    int element_of_c = num_of_matrix * N * P;
    A = new int[element_of_a];
    B = new int[element_of_b];
    C = new int[element_of_c];
    srand(time(nullptr));

    for (int i = 0; i < element_of_a; i++)
        A[i] = rand() % 10;

    for (int i = 0; i < element_of_b; i++)
        B[i] = rand() % 10;

    int *device_A, *device_B, *device_C;
    cudaMalloc(&device_A, element_of_a * sizeof(int));
    cudaMalloc(&device_B, element_of_b * sizeof(int));
    cudaMalloc(&device_C, element_of_c * sizeof(int));

    cudaMemcpy(device_A, A, element_of_a * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, B, element_of_b * sizeof(int), cudaMemcpyHostToDevice);

    cout << "Enter the number of threads per block: ";
    int threads_per_block;
    cin >> threads_per_block;

    cudaEvent_t start_time, end_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&end_time);
    cudaEventRecord(start_time); // record the start time

    // functionName<<<block,threads_per_block>>>(....);
    matrix_multiplication<<<1, threads_per_block>>>(device_A, device_B, device_C, N, M, P, num_of_matrix, threads_per_block);

    cudaDeviceSynchronize();        // wait for all the threads to complete
    cudaEventRecord(end_time);      // record the end time
    cudaEventSynchronize(end_time); // wait for the end time to be recorded
    cudaMemcpy(C, device_C, element_of_c * sizeof(int), cudaMemcpyDeviceToHost);

    float time_taken;
    cudaEventElapsedTime(&time_taken, start_time, end_time);
    cout << "Time taken for multiplication: " << time_taken << "ms"
         << "\n";

    // print results
    //  for(int x = 0;x<num_of_matrix;x++)
    //  {
    //      cout<<"Matrix ["<<x+1<<"]"<<"\n";
    //      for(int i=0;i<N;i++)
    //      {
    //          for(int j=0;j<P;j++)
    //              cout<<C[x*N*P + i*P + j]<<" ";
    //          cout<<"\n";
    //      }
    //      cout<<"\n";
    //  }

    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);

    return 0;
}