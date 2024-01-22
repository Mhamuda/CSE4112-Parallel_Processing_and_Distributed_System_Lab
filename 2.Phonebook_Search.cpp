//To compile: mpic++ 2.Phonebook_Search.cpp 
//To run: mpiexec -n 4 ./a.out "name" file1.txt file2.txt file3.txt

#include<bits/stdc++.h>
#include<mpi.h>
using namespace std;

void send_a_number(int num, int to)
{
    MPI_Send(&num, 1, MPI_INT, to, 0, MPI_COMM_WORLD);
}

void send_a_string(string str, int to)
{
    int length = str.size()+1;
    send_a_number(length, to);
    MPI_Send(&str[0], length, MPI_CHAR, to, 0, MPI_COMM_WORLD);    
}

int receive_a_number(int from)
{
    int length;
    MPI_Recv(&length,1,MPI_INT,from,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    return length;
}

string receive_a_string(int from)
{
    int length = receive_a_number(from);
    if(length==0)
        return "";

    char *st = new char[length];
    MPI_Recv(st,length,MPI_CHAR,from,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    return string(st);
}

string generate_string(vector<string>&str, int start_position, int end_position)
{
    string result = "";
    for(int i=start_position;i<end_position;i++)
        result += str[i] + "\n";
    return result;
}

vector<string>get_string(string str)
{
    stringstream ss(str);   //Used to break string into tokens
    string line;
    vector<string>result;
    while(getline(ss,line))
        result.push_back(line);

    return result;
}

bool check(string text, string pattern)
{
    int n = text.size();
    int m = pattern.size();
    int cnt= 0;

    for(int i=0;i<n;i++)
    {
        cnt=0;
        for(int j=0;j<m;j++)
        {
            if(text[i+j]==pattern[j] || text[i+j]==(pattern[j] xor 32) )
                cnt++;
            else
                break;
        }
        if(cnt==m)
            return true;
    }

    return false;
}


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_of_process;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&num_of_process);

    if(rank==0)
    {
        vector<string>contact_list;
        string search_name = argv[1];

        for(int i=2; i<argc;i++)
        {
            ifstream fin(argv[i]);  //Reading input from a file
            string line;
            while(getline(fin,line))
                contact_list.push_back(line);   
        }

        int contact_list_size = contact_list.size();
        clock_t start_time, end_time;
        start_time = clock();

        for(int i=1;i<num_of_process;i++)
        {
            int start_position = i*(contact_list_size/num_of_process);
            int end_position = (i+1)*(contact_list_size/num_of_process);
            
            if(i==num_of_process-1)
                end_position = contact_list_size;

            string for_send_name = generate_string(contact_list, start_position, end_position);
            send_a_string(for_send_name, i);
            send_a_string(search_name, i);
        }

        vector<string>temp;
        set<string>ans;

        for(int i=0;i<contact_list_size/num_of_process;i++)
        {
            if(check(contact_list[i], search_name))
                temp.push_back(contact_list[i]);
        }

        // string work_for = generate_string(temp,0,temp.size());
        // vector<string> work_list = get_string(work_for);

        for(int i=0;i<temp.size();i++)
            ans.insert(temp[i]);

        for(int i=1;i<num_of_process;i++)
        {
            string receive_string = receive_a_string(i);
            vector<string>rec_list = get_string(receive_string);
            int rec_list_size = rec_list.size();

            for(int j=0;j<rec_list_size;j++)
                ans.insert(rec_list[j]);
        }

        for(set<string>::iterator it=ans.begin();it!=ans.end();it++)
            cout<<*it<<"\n";

        end_time = clock();
        double total_time = (double)(end_time-start_time)/double(CLOCKS_PER_SEC);
        cout<<"Total time taken: "<<total_time<<"\n";
    }
    else
    {
        string name = receive_a_string(0);
        string search_name = receive_a_string(0);
        vector<string>contact_list = get_string(name);
        vector<string>result;
        int contact_list_size = contact_list.size();

        for(int i=0;i<contact_list_size;i++)
        {
            if(check(contact_list[i], search_name))
                result.push_back(contact_list[i]);
        }

        string for_send_name = generate_string(result, 0, result.size());
        send_a_string(for_send_name, 0);
    }

    MPI_Finalize();
    return 0;
}