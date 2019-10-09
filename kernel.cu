#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <mkl_spblas.h>
#include <malloc.h>
#include <time.h>
#include <cuda_runtime.h>


MKL_INT m = 5, nnz = 13;

__global__ void hadamard(int *result, MKL_INT *rows_start_A2, MKL_INT *rows_end_A2, MKL_INT *col_index_A2, MKL_INT no_rows_A, MKL_INT * rows_start_A, MKL_INT * rows_end_A, MKL_INT * col_index_A,
float* values_A2, float * values_A) 
{
	int sum = 0;
	//#pragma omp parallel for shared(sum) reduction(+: sum)
	for (int r_index = 0; r_index < no_rows_A; r_index++)// Processing each rows of the matrices
	{
		int A_lower_bound = rows_start_A[r_index] - 1;
		int A_upper_bound = rows_end_A[r_index] - 2;
		int A2_lower_bound = rows_start_A2[r_index] - 1;
		int A2_upper_bound = rows_end_A2[r_index] - 2;

		int A_c_index = A_lower_bound;
		int A2_c_index = A2_lower_bound;

		while (A_c_index >= A_lower_bound && A_c_index <= A_upper_bound && A2_c_index >= A2_lower_bound && A2_c_index <= A2_upper_bound) {
			if (col_index_A[A_c_index] == col_index_A2[A2_c_index]) {
				sum += (int)(values_A[A_c_index] * values_A2[A2_c_index]);
				A_c_index++;
				A2_c_index++;
			}
			else if (col_index_A[A_c_index] < col_index_A2[A2_c_index]) {
				A_c_index++;
			}
			else {
				A2_c_index++;
			}
		}

	}
	//printf("CUDA sum = %d\n", sum);
	result[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

int main()
{
	bool debugging = false;

	int NNZ = 6629222;	// auto
	MKL_INT sizeOfMatrix = 448695;
	//int NNZ = 16313034; // britain
	//MKL_INT sizeOfMatrix = 7733822;
	//int NNZ = 25165738;	// delaunay
	//MKL_INT sizeOfMatrix = 4194304;

	MKL_INT* row = (MKL_INT*)malloc(NNZ * sizeof(MKL_INT));
	MKL_INT* col = (MKL_INT*)malloc(NNZ * sizeof(MKL_INT));
	float* val = (float*)malloc(NNZ * sizeof(float));
	sparse_matrix_t A_COO, A, A2;
	sparse_status_t status;
	clock_t start, end;
	double time_taken, total_time_taken = 0;

	FILE* fp;
	//char buff[255];
	int buff_int = 0;

	fp = fopen("auto_A.txt", "r");	//auto
	//fp = fopen("britain_A.txt", "r");	//britain
	//fp = fopen("delaunay_A.txt", "r");	//delaunay

	for (int i = 0; i < NNZ; i++)
	{
		fscanf(fp, "%d", &row[i]);
		fscanf(fp, "%d", &col[i]);
		fscanf(fp, "%f", &val[i]);
	}
	fclose(fp);

	start = clock();

	// Creating the Sparse Matrix A in COO format
	status = mkl_sparse_s_create_coo(&A_COO, SPARSE_INDEX_BASE_ONE, sizeOfMatrix, sizeOfMatrix, NNZ, row, col, val);
	if (status == SPARSE_STATUS_SUCCESS && debugging)
		printf("Matrix A created with SUCCESS.\n");

	// Convert Sparse Matrix A to CSR format
	status = mkl_sparse_convert_csr(A_COO, SPARSE_OPERATION_NON_TRANSPOSE, &A);
	if (status == SPARSE_STATUS_SUCCESS && debugging)
		printf("Matrix A converted to CSR with SUCCESS.\n");
	/*status = mkl_sparse_order(A);
	if (status == SPARSE_STATUS_SUCCESS && debugging)
		printf("A ORDER done with SUCCESS.\n");*/

	struct matrix_descr generalDesc;
	generalDesc.type = SPARSE_MATRIX_TYPE_GENERAL;
	status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, A, A, &A2);
	if (status == SPARSE_STATUS_SUCCESS && debugging)
		printf("\nA^2 multiplication done with SUCCESS.\n");

	status = mkl_sparse_order(A2);
	if (status == SPARSE_STATUS_SUCCESS && debugging)
		printf("A^2 ORDER done with SUCCESS.\n");


	MKL_INT no_rows_A2, no_cols_A2, * rows_start_A2, * rows_end_A2, * col_index_A2;
	MKL_INT no_rows_A, no_cols_A, * rows_start_A, * rows_end_A, * col_index_A;
	float* values_A2, * values_A;
	sparse_index_base_t sparse_index_A2, sparse_index_A;

	status = mkl_sparse_s_export_csr(A2, &sparse_index_A2, &no_rows_A2, &no_cols_A2, &rows_start_A2, &rows_end_A2, &col_index_A2, &values_A2);
	if (status == SPARSE_STATUS_SUCCESS && debugging)
		printf("\nA^2 export done with SUCCESS.\n");
	//printf("%d\t%d\t%d\t%p\t%p\t%p\n", sparse_index_A2, no_rows_A2, no_cols_A2, &rows_start_A2[1], rows_end_A2, col_index_A2);

	status = mkl_sparse_s_export_csr(A, &sparse_index_A, &no_rows_A, &no_cols_A, &rows_start_A, &rows_end_A, &col_index_A, &values_A);
	if (status == SPARSE_STATUS_SUCCESS && debugging)
		printf("A export done with SUCCESS.\n");

	MKL_INT * c_rows_start_A2, * c_rows_end_A2, * c_col_index_A2;
	MKL_INT * c_rows_start_A, * c_rows_end_A, * c_col_index_A;
	float* c_values_A2, * c_values_A;

	cudaMallocManaged(&c_rows_start_A2, no_rows_A2 *sizeof(MKL_INT));
	cudaMallocManaged(&c_rows_end_A2, no_rows_A2 * sizeof(MKL_INT));
	cudaMallocManaged(&c_rows_start_A, no_rows_A * sizeof(MKL_INT));
	cudaMallocManaged(&c_rows_end_A, no_rows_A * sizeof(MKL_INT));
	cudaMallocManaged(&c_col_index_A, rows_end_A[no_rows_A - 1] * sizeof(MKL_INT));
	cudaMallocManaged(&c_col_index_A2, rows_end_A2[no_rows_A2 - 1] * sizeof(MKL_INT));
	cudaMallocManaged(&c_values_A, rows_end_A[no_rows_A - 1] * sizeof(float));
	cudaMallocManaged(&c_values_A2, rows_end_A2[no_rows_A2 - 1] * sizeof(float));

	cudaDeviceSynchronize();

	c_rows_start_A[0] = 1;

	printf("NNZ of A = %d\n", rows_end_A[no_rows_A - 1]);
	printf("NNZ of A2 = %d\n", rows_end_A2[no_rows_A2 - 1]);

	for (int i = 0; i < no_rows_A; i++)
	{
		c_rows_start_A[i] = rows_start_A[i];
		c_rows_end_A[i] = rows_end_A[i];
		c_rows_start_A2[i] = rows_start_A2[i];
		c_rows_end_A2[i] = rows_end_A2[i];
	}
	for (int i = 0; i < rows_end_A[no_rows_A - 1]; i++)
	{
		c_col_index_A[i] = col_index_A[i];
		c_values_A[i] = values_A[i];
	}
	for (int i = 0; i < rows_end_A2[no_rows_A2 - 1]; i++)
	{
		c_col_index_A2[i] = col_index_A2[i];
		c_values_A2[i] = values_A2[i];
	}

	/*printf("\nNNZ of A = %d\n", rows_start_A[no_rows_A] - sparse_index_A);
	printf("NNZ of A^2 = %d\n", rows_start_A2[no_rows_A2]-sparse_index_A2);*/

	end = clock();
	// Calculating total time taken by the program. 
	time_taken = double(end - start) / double(CLOCKS_PER_SEC);
	printf("A^2 time = %f\n", time_taken);
	total_time_taken += time_taken;

	start = clock();

	// Hadamard product and sum together
	int* results;
	cudaMallocManaged(&results, no_rows_A * sizeof(int));
	
	hadamard<<<1,1>>>(results,c_rows_start_A2, c_rows_end_A2, c_col_index_A2, no_rows_A, c_rows_start_A, c_rows_end_A, c_col_index_A, c_values_A2, c_values_A);
	cudaDeviceSynchronize();

	end = clock();
	// Calculating total time taken by the program. 
	time_taken = double(end - start) / double(CLOCKS_PER_SEC);
	printf("Hadamard time = %f\n", time_taken);
	total_time_taken += time_taken;
	printf("Wall time = %f\n", total_time_taken);

	printf("\nsum = %d\n", results[0]);

	float nT = results[0] / 6;
	printf("nT = %.0f\n", nT);


	mkl_sparse_destroy(A_COO);
	mkl_sparse_destroy(A);
	mkl_sparse_destroy(A2);
	free(row);
	free(col);
	free(val);
	cudaFree(c_rows_start_A2);
	cudaFree(c_rows_end_A2);
	cudaFree(c_rows_start_A);
	cudaFree(c_rows_end_A);
	cudaFree(c_col_index_A);
	cudaFree(c_col_index_A2);
	cudaFree(c_values_A);
	cudaFree(c_values_A2);
	cudaFree(results);
}