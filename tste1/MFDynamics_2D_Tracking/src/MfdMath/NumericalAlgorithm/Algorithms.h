#pragma once

#include <assert.h>
#include <memory.h>
#include <iostream>
class Algorithms
{
public:
	Algorithms(void);
	~Algorithms(void);

public:
	static void RadixSort(int* in_ids, int* out_rids, int in_size, int* out_beginOfLists, int* out_endOfLists, int in_bucket);

	static void Accumulation(int* in_arr, int in_size);

	static void KMinimum(float* in_arr, int in_size, int* out_arrIndex, int in_out_size);
};

