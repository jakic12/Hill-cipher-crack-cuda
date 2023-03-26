#include <iostream>
#include <math.h>
#include <string>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__
void to_base_N(int N, int num, int* out, int size){
    int i = 0;
    while(num > 0){
        if (i >= size){
            break;
        }
        out[i] = num % N;
        num = num / N;
        i++;
    }
    // fill with zeroes
    for(; i < size; i++){
        out[i] = 0;
    }
}

void to_base_N_cpu(int N, int num, int* out, int size){
    int i = 0;
    while(num > 0){
        if (i >= size){
            break;
        }
        out[i] = num % N;
        num = num / N;
        i++;
    }
    // fill with zeroes
    for(; i < size; i++){
        out[i] = 0;
    }
}

void from_base_factorial(int num, int* out, int size){
    int i = 0;
    while(num > 0){
        if (i >= size){
            return;
        }
        out[i] = num % (size - i);
        num = num / (size - i);
        i++;
    }
}

__device__
void base_factorial_to_perm(
    int perm_i, // the index of the permutation in the factorial base
    int* out, // the output array
    int size // the size of the arrays
){
    // pool of numbers to chose
    int* temp = new int[size];
    for(int i = 0; i < size; i++){
        temp[i] = i;
    }

    // this operation can be slower, because size is small
    for(int i = 0; i < size; i++){
        // convert from base factiorial to base 10
        // (each digit is the index of the number to chose from the pool)
        int j = perm_i % (size - i);
        perm_i = perm_i / (size - i);

        // chose the number
        out[i] = temp[j];

        // remove the number from the pool
        for(int l = j; l < size - i - 1; l++){
            temp[l] = temp[l+1];
        }
    }
}

int get_num_blocks(int block_size, int N){
    return (N + block_size - 1) / block_size;
}

__global__
void get_chi_squared_for_all(
    const char* cypher, // the cypher
    int N, // length of the cypher
    int k, // size of the bruteforced vector
    int num_of_vecs, // number of vectors to test
    float* chi_squared // the out chi squared values
){
    // its fine to have this on the device because it is constant
    static const float dist[26] = {8.12f, 1.49f, 2.71f, 4.32f, 12.02f, 2.30f, 2.03f, 5.92f, 7.31f, 0.10f, 0.69f, 3.98f, 2.61f, 6.95f, 7.68f, 1.82f, 0.11f, 6.02f, 6.28f, 9.10f, 2.88f, 1.11f, 2.09f, 0.17f, 2.11f, 0.07f};

    int i = blockIdx.x*blockDim.x+threadIdx.x;

    // cuda will run more threads than we need
    if (i < num_of_vecs) {
        // get vector that is being tested
        int* vec = new int[k];
        to_base_N(26, i, vec, k);

        chi_squared[i] = 0.0f;
        int* freq = new int[26];

        for(int j = 0; j < N; j += k){
            int sum = 0;

            // multiply the vector with the block
            for(int l = 0; l < k; l++){
                //printf("c:%d, converted:%d\n, vec[l]: %d\n", cypher[j+l], (cypher[j+l] - 'a'), vec[l]);
                sum += (cypher[j+l] - 'a') * vec[l];
            }
            sum %= 26;

            // add to the frequency
            freq[sum]++;
        }
        
        // print freq
        /*if(vec[0] == 15 && vec[1] == 18){
            for(int j = 0; j < 26; j++){
                printf("%d-%d:%d\n",i, j, freq[j]);
            }
        }*/

        // calculate chi squared
        for(int j = 0; j < 26; j++){
            float dst = dist[j]/100.0f * N;
            chi_squared[i] += (freq[j] - dst)*(freq[j] - dst)/dst;
        }

        //printf("[%d] chi squared done\n", i);

        delete[] vec;
        delete[] freq;
    }

}

// function that calculates the determinant of a matrix with LU
float det(int N, float *A){
    return -1;
}

__global__
void get_decrypted_permutations(
    const char* cypher, // the cypher
    int N, // length of the cypher
    int k, // size of the matrix
    int num_of_perms, // number of permutations
    int* vectors, // the vectors to test permutations of (vector of numbers)
    char* decrypted // the out decrypted cypher (flattened matrix)
){
    int i = blockIdx.x*blockDim.x+threadIdx.x; // permutation id
    int j = blockIdx.y*blockDim.y+threadIdx.y; // block id (block in text)

    if (i < num_of_perms && j+k < N) {

        // get how the vectors are permuted in the permutation encoded as i
        // (array of indexes)
        int* idx_permutation = new int[k];
        base_factorial_to_perm(i, idx_permutation, k);

        // for each letter in the current block to decypher
        int* vector = new int[k];
        for(int m = 0; m < k; m++){
            // convert from number to vector
            to_base_N(26, vectors[idx_permutation[m]], vector, k);

            // multiply block with vector
            int chr = 0;
            for(int l = 0; l < k; l++){
                chr += (cypher[j*k+l] - 'a') * vector[l];
            }
            chr = chr % 26;
            decrypted[i*N + (j*k+m)] = chr + 'a';
        }

        delete[] idx_permutation;
        delete[] vector;
    }
}

int argmin(int* a, int size){
    int min = 0;
    for(int i = 1; i < size; i++){
        if(a[i] < a[min]){
            min = i;
        }
    }
    return min;
}

// search for the index of the max value in a
// (ignore indexes not in indexes)
int argmax_from_list_of_indexes(float* a, int* indexes, int idx_size){
    int max = 0;
    for(int i = 0; i < idx_size; i++){
        int idx = indexes[i];
        if(a[idx] > a[indexes[max]]){
            max = i;
        }
    }
    return max;
}

void print_arr(int* a, int size){
    printf("[");
    for(int i = 0; i < size; i++){
        printf("%d ", a[i]);
    }
    printf("]\n");
}

void print_arrf(float* a, int size){
    printf("[");
    for(int i = 0; i < size; i++){
        printf("%.2f ", a[i]);
    }
    printf("]\n");
}

void arg_smallest_k(float* a, int* out, int k, int a_size){
    for(int i = 0; i < k; i++){
        out[i] = i;
    }

    int max_idx = argmax_from_list_of_indexes(a, out, k);
    for(int i = k; i < a_size; i++){
        if(a[i] < a[out[max_idx]]){
            out[max_idx] = i;
            max_idx = argmax_from_list_of_indexes(a, out, k);
        }
    }
}

int factorial(int n){
    int out = 1;
    for(int i = 2; i <= n; i++){
        out *= i;
    }
    return out;
}

int main(void)
{
    std::string line;
    std::getline(std::cin, line);

    // cypher encrypted with hill cipher
    char* cypher;
    int N = line.length();
    cudaMallocManaged(&cypher, N*sizeof(float));

    // copy the cypher to the gpu
    for(int i = 0; i < N; i++){
        cypher[i] = line[i];
    }
    std::cout << "N:" << N << std::endl;

    int blockSize = 32;

    // k is the block size
    for(int k = 2; k <= 2; k++){
        // it doesn't make sense to parallelize this because you have to manually confirm
        // each decryption anyways
        if(N % k == 0){
            int num_of_vecs = pow(26,k);
            int numBlocks = get_num_blocks(blockSize, num_of_vecs);

            std::cout << "k:" << k << std::endl << "number of vectors to check: " << num_of_vecs << std::endl;

            // allocate memory for the chi squared values
            if (num_of_vecs < 0) {
                std::cout << "Overflow in number of vectors" << std::endl;
                exit(0);
            }

            float* chi_squared;
            gpuErrchk( cudaMallocManaged(&chi_squared, num_of_vecs*sizeof(float)) );
            get_chi_squared_for_all<<<numBlocks, blockSize>>>(
                cypher,
                N,
                k, 
                num_of_vecs,
                chi_squared
            );

            // wait for the kernel to finish
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            std::cout << "(" << num_of_vecs << "):";

            int* vec = new int[k];
            // each number represents a row vector, so this is the decoder matrix

            for(int i = 0; i < num_of_vecs; i++){
                to_base_N_cpu(26, i, vec, k);
                std::cout << i << '[';
                for(int j = 0; j < k; j++){
                    std::cout << vec[j] << ", ";
                }
                std::cout<< "]:";
                std::cout << chi_squared[i] << std::endl;
            }
            std::cout << std::endl;

            std::cout << "done calculating chi" << std::endl;

            int* smallest_k_cpu = new int[k];
            arg_smallest_k(chi_squared, smallest_k_cpu, k, num_of_vecs);

            std::cout << "smallest k:[";
            for(int i = 0; i < k; i++){
                std::cout << smallest_k_cpu[i] << " ";
            }
            std::cout << "]" << std::endl;

            for(int i = 0; i < k; i++){
                int vec_i = smallest_k_cpu[i];
                to_base_N_cpu(26, vec_i, vec, k);
                std::cout << vec_i << '[';
                for(int j = 0; j < k; j++){
                    std::cout << vec[j] << ", ";
                }
                std::cout << "]:";
                std::cout << chi_squared[vec_i] << std::endl;
            }

            int* smallest_k;
            cudaMallocManaged(&smallest_k, k*sizeof(int));

            for(int i = 0; i < k; i++){
                smallest_k[i] = smallest_k_cpu[i];
            }

            int num_of_perms = factorial(k);
            char* deciphered;
            cudaMallocManaged(&deciphered, num_of_perms*N*sizeof(char));

            // calculate how many cuda threads and blocks to spawn
            dim3 dimBlock(num_of_perms, N/k); // so your threads are BLOCK_SIZE*BLOCK_SIZE, 256 in this case
            dim3 dimGrid(1, 1); // 1*1 blocks in a grid

            get_decrypted_permutations<<<dimGrid, dimBlock>>>(
                cypher,
                N,
                k,
                num_of_perms,
                smallest_k,
                deciphered
            );

            cudaDeviceSynchronize();
            std::cout << std::endl;
            for(int o = 0; o < num_of_perms; o++){
                std::cout << o << ": ";
                for(int r = 0; r < N; r++){
                    std::cout << deciphered[o*N + r];
                }
                std::cout << std::endl;
            }

            cudaFree(deciphered);
            cudaFree(chi_squared);
        }
    }
}
