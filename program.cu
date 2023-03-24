#include <iostream>
#include <math.h>
#include <string>

__device__
void to_base_N(int N, int num, int* out, int size){
    int i = 0;
    while(num > 0){
        if (i >= size){ // TODO: fill with zeroes
            return;
        }
        out[i] = num % N;
        num = num / N;
        i++;
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
                sum += (cypher[j+l] - 'a') * vec[l];
            }
            sum %= 26;

            // add to the frequency
            ++freq[sum];
        }

        // calculate chi squared
        for(int j = 0; j < 26; j++){
            chi_squared[i] += (freq[j] - dist[j])*(freq[j] - dist[j])/dist[j];
        }

        delete[] vec;
        delete[] freq;

        chi_squared[i] = 1;
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
            to_base_N(26, idx_permutation[m], vector, k);

            // multiply block with vector
            int chr = 0;
            for(int l = 0; l < k; l++){
                chr += cypher[j+l] * vector[l];
            }
            chr = chr % 26;
            decrypted[i*N + (j+m)] = chr;
        }

        delete[] idx_permutation;
        delete[] vector;
    }
}

int argmin(int* a, int size){
    int min = 0;
    for(int i = 0; i < size; i++){
        if(a[i] < a[min]){
            min = i;
        }
    }
    return min;
}

int argmax(int* a, int size){
    int min = 0;
    for(int i = 0; i < size; i++){
        if(a[i] > a[min]){
            min = i;
        }
    }
    return min;
}

void arg_smallest_k(float* a, int* out, int k, int a_size){
    for(int i = 0; i < k; i++){
        out[i] = a[i];
    }

    int max = argmax(out,k);

    for(int i = k; i < a_size; i++){
        if(a[i] < a[max]){
            out[max] = i;
            max = argmax(out,k);
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
    const char* cypher = line.c_str();
    int N = line.length();
    cudaMallocManaged(&cypher, N*sizeof(float));

    std::cout << "N:" << N << std::endl;

    int blockSize = 32;

    // k is the block size
    for(int k = 2; k <= N; k++){
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

            float* chi_squared = new float[num_of_vecs];
            cudaMallocManaged(&chi_squared, num_of_vecs*sizeof(float));
            get_chi_squared_for_all<<<numBlocks, blockSize>>>(
                cypher,
                N,
                k, 
                num_of_vecs,
                chi_squared
            );

            // wait for the kernel to finish
            cudaDeviceSynchronize();

            std::cout << "(" << num_of_vecs << "):";
            for(int i = 0; i < num_of_vecs; i++){
                std::cout << chi_squared[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "done calculating chi" << std::endl;

            // each number represents a row vector, so this is the decoder matrix
            int* smallest_k = new int[k];
            arg_smallest_k(chi_squared, smallest_k, k, num_of_vecs);
            cudaMallocManaged(&smallest_k, k*sizeof(int));


            int num_of_perms = factorial(k);
            char* decyphered = new char[num_of_perms*N];
            cudaMallocManaged(&decyphered, k*sizeof(char));

            // calculate how many cuda threads and blocks to spawn
            dim3 dimBlock(num_of_perms, N/k); // so your threads are BLOCK_SIZE*BLOCK_SIZE, 256 in this case
            dim3 dimGrid(1, 1); // 1*1 blocks in a grid

            get_decrypted_permutations<<<dimGrid, dimBlock>>>(
                cypher,
                N,
                k,
                num_of_perms,
                smallest_k,
                decyphered
            );

            cudaDeviceSynchronize();
            std::cout << std::endl;
            for(int o = 0; o < num_of_perms; o++){
                std::cout << o << ": ";
                for(int r = 0; r < N; r++){
                    std::cout << decyphered[o*N + r];
                }
                std::cout << std::endl;
            }

            cudaFree(decyphered);
            cudaFree(chi_squared);
        }
    }
}
