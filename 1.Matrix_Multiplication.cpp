//To compile: mpic++ 1.Matrix_Multiplication.cpp
//To run: mpiexec -n 4 ./a.out

#include <bits/stdc++.h>
using namespace std;
#include <mpi.h>

void send_a_number(int num, int to)
{
    MPI_Send(&num, 1, MPI_INT, to, 0, MPI_COMM_WORLD);
}
void send_an_array(int *arr, int sz, int to)
{
    send_a_number(sz, to);
    MPI_Send(arr, sz, MPI_INT, to, 0, MPI_COMM_WORLD);
}

int receive_a_number(int from)
{
    int length;
    MPI_Recv(&length, 1, MPI_INT, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return length;
}

vector<int> receive_an_array(int from)
{
    int length = receive_a_number(from);
    vector<int> ans(length);
    MPI_Recv(&ans[0], length, MPI_INT, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return ans;
}

vector<int> multiplication(int *A, int *B, int N, int M, int P, int number_of_matrix)
{
    vector<int> C(N * P * number_of_matrix);

    for (int x = 0; x < number_of_matrix; x++)
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

    return C;
}

int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);
    int rank, num_of_process;
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        int N, M, P, number_of_matrix;
        cout << "Enter the dimenssion of matrices: ";
        cin >> N >> M >> P;

        cout << "Enter the number of matrix: ";
        cin >> number_of_matrix;

        int *A, *B;
        int a_matrix_len = number_of_matrix * N * M;
        int b_matrix_len = number_of_matrix * M * P;

        A = new int[a_matrix_len];
        B = new int[b_matrix_len];
        srand(time(NULL));

        for (int i = 0; i < a_matrix_len; i++)
            A[i] = rand() % 10;

        for (int i = 0; i < b_matrix_len; i++)
            B[i] = rand() % 10;

        clock_t start_time, end_time;
        start_time = clock();

        for (int i = 1; i < num_of_process; i++)
        {

            int start_position = i * (number_of_matrix / num_of_process);
            int end_position = (i + 1) * (number_of_matrix / num_of_process);
            if (i == num_of_process - 1)
                end_position = number_of_matrix;

            send_a_number(N, i);
            send_a_number(M, i);
            send_a_number(P, i);
            send_a_number(end_position - start_position, i);

            int start_index_for_A = start_position * N * M;
            int end_index_for_A = end_position * N * M;

            send_an_array(A + start_index_for_A, end_index_for_A - start_index_for_A, i);

            int start_index_for_B = start_position * M * P;
            int end_index_for_B = end_position * M * P;
            send_an_array(B + start_index_for_B, end_index_for_B - start_index_for_B, i);
        }

        vector<int> C = multiplication(&A[0], &B[0], N, M, P, number_of_matrix / num_of_process);
        for (int i = 1; i < num_of_process; i++)
        {
            vector<int> temp = receive_an_array(i);
            int num_of_element = temp.size();

            for (int j = 0; j < num_of_element; j++)
                C.push_back(temp[j]);
        }

        // Print the result
        // for (int x = 0; x < number_of_matrix; x++)
        // {
        //     cout << "Matrix C[" << x << "]:\n";
        //     for (int i = 0; i < N; i++)
        //     {
        //         for (int j = 0; j < P; j++)
        //             cout << C[x * N * P + i * P + j] << " ";
        //         cout << "\n";
        //     }
        //     cout << "\n";
        // }

        end_time = clock();
        double time_taken = double(end_time - start_time) / double(CLOCKS_PER_SEC);
        cout << "Process " << rank << " took " << time_taken << "secconds."<< "\n";
    }
    else
    {
        clock_t start_time, end_time;
        start_time = clock();

        int N = receive_a_number(0);
        int M = receive_a_number(0);
        int P = receive_a_number(0);
        int number_of_matrix = receive_a_number(0);

        vector<int> A = receive_an_array(0);
        vector<int> B = receive_an_array(0);
        vector<int> C = multiplication(&A[0], &B[0], N, M, P, number_of_matrix);
        send_an_array(&C[0], C.size(), 0);

        end_time = clock();
        double time_taken = double(end_time - start_time) / double(CLOCKS_PER_SEC);
        cout << "Process " << rank << " took " << time_taken << "seconds."<< "\n";
    }

    MPI_Finalize();
    return 0;
}