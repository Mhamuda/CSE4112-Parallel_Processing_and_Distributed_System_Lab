//To compile: !nvcc 2.Phonebook.cu
//To run : !./a.out "name" file1.txt file2.txt file3.txt 

#include <bits/stdc++.h>
using namespace std;

__global__ void find_string_matchings(char *text, char *pattern, int *index, int pattern_len, int text_len, int num_of_core)
{
    int thread_id = threadIdx.x;

    int start_index = thread_id * (text_len / num_of_core);
    int end_index = (thread_id + 1) * (text_len / num_of_core);

    if (thread_id == num_of_core - 1)
        end_index = text_len;

    for (int i = start_index; i < end_index; i++)
    {
        int cnt = 0;
        for (int j = index[i]; text[j] != '\n'; j++)
        {
            cnt = 0;
            for (int k = 0; k < pattern_len; k++)
            {
                if (pattern[k] == text[j + k])
                    cnt++;
                else if ((pattern[k] - 32) == text[j + k])
                    cnt++;
                else
                    break;
            }

            if (cnt == pattern_len)
            {
                index[i] = -1;
                break;
            }
        }
    }
}

int main(int argc, char *argv[])
{
    vector<string> contact_list;
    string search_name = argv[1];
    int name_size = search_name.size();

    for (int i = 0; i < name_size; i++)
    {
        search_name[i] = tolower(search_name[i]);
    }

    for (int i = 2; i < argc; i++)
    {
        ifstream fin(argv[i]); // reading input from file
        string line;
        while (getline(fin, line))
        {
            contact_list.push_back(line);
        }
    }

    int contact_list_size = contact_list.size();
    vector<int> index(contact_list_size);
    string text = "";

    for (int i = 0; i < contact_list_size; i++)
    {
        index[i] = text.size();
        text += contact_list[i] + "\n";
    }

    char *DeviceText, *DevicePattern;
    cudaMalloc(&DeviceText, (text.size() + 1) * sizeof(char));
    cudaMemcpy(DeviceText, &text[0], (text.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
    // cudaMemcpy(dest,src,num of byte to copy,cudaMemcpyHostToDevice);

    cudaMalloc(&DevicePattern, (search_name.size() + 1) * sizeof(char));
    cudaMemcpy(DevicePattern, &search_name[0], (search_name.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);

    int *DeviceIndex;
    cudaMalloc(&DeviceIndex, contact_list_size * sizeof(int));
    cudaMemcpy(DeviceIndex, &index[0], contact_list_size * sizeof(int), cudaMemcpyHostToDevice);

    int num_of_core;
    cout << "Enter number of cores: ";
    cin >> num_of_core;

    cudaEvent_t start, end;  // cudaEvent_t is a predefined data type for cuda events to measure time
    cudaEventCreate(&start); // creating event
    cudaEventCreate(&end);   // creating event
    cudaEventRecord(start);  // start recording time

    find_string_matchings<<<1, num_of_core>>>(DeviceText, DevicePattern, DeviceIndex, search_name.size(), contact_list_size, num_of_core);
    cudaDeviceSynchronize();   // wait for the device to finish its task
    cudaEventRecord(end);      // end recording time
    cudaEventSynchronize(end); // wait for the event to complete

    cudaMemcpy(&index[0], DeviceIndex, contact_list_size * sizeof(int), cudaMemcpyDeviceToHost);

    float time_taken;
    cudaEventElapsedTime(&time_taken, start, end); // calculating time taken
    cout << "Time taken: " << time_taken << " ms"
         << "\n";

    set<string> result;
    for (int i = 0; i < contact_list_size; i++)
    {
        if (index[i] == -1)
            result.insert(contact_list[i]);
    }

    for (set<string>::iterator it = result.begin(); it != result.end(); it++)
    {
        cout << *it << "\n";
    }
}