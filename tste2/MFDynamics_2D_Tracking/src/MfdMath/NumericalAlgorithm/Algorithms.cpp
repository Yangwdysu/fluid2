#include "Algorithms.h"

Algorithms::Algorithms(void)
{
}


Algorithms::~Algorithms(void)
{
}

void Algorithms::RadixSort(int* in_ids, int* out_rids, int in_size, int* out_beginOfLists, int* out_endOfLists, int in_bucket)
{
	if (in_bucket <= 0)
	{
		std::cout << "radix sort failed !!!" << std::endl;
		exit(0);
	}

	memset(out_beginOfLists, 0, sizeof(int)*in_bucket);
	memset(out_endOfLists, 0, sizeof(int)*in_bucket);

	for (int i = 0; i < in_size; i++)
	{
		if (in_ids[i] >= 0 && in_ids[i] < in_bucket)
		{
			out_endOfLists[in_ids[i]]++;
		}
	}

	Algorithms::Accumulation(out_endOfLists, in_bucket);

	out_beginOfLists[0] = 0;
	for (int i = 1; i < in_bucket; i++)
	{
		out_beginOfLists[i] = out_endOfLists[i-1];
	}

	int* index = new int[in_bucket];
	memcpy(index, out_beginOfLists, sizeof(int)*in_bucket);
	int totalNum = out_endOfLists[in_bucket-1];
	for (int i = 0; i < in_size; i++)
	{
		if (in_ids[i] >= 0)
		{
			out_rids[index[in_ids[i]]] = i;
			index[in_ids[i]]++;
		}
		else
		{
			out_rids[totalNum] = i;
			totalNum++;
		}
	}

	delete[] index;
}

void Algorithms::Accumulation( int* in_arr, int in_size )
{
	if(in_size <= 1)
	{
		std::cout << "Accumulation failed !!!" << std::endl;
		exit(0);
	}

	for (int i = 1; i < in_size; i++)
	{
		in_arr[i] += in_arr[i-1];
	}
}

void Algorithms::KMinimum( float* in_arr, int in_size, int* out_arrIndex, int in_out_size )
{
	if(in_out_size > in_size)
	{
		std::cout << "The required size is larger than the input size !!!" << std::endl;
		exit(0);
	}

	bool* checked = new bool[in_size];
	memset(checked, 0, in_size*sizeof(bool));
	//find k minimum value
	for (int i = 0; i < in_out_size; i++)
	{
		float minV = FLT_MAX;
		int index = -1;
		//find the minimum value
		for (int j = 0; j < in_size; j++)
		{
			if (!checked[j])
			{
				if (in_arr[j] < minV)
				{
					minV = in_arr[j];
					index = j;
				}
			}
		}

		checked[index] = true;
		out_arrIndex[i] = index;
	}
	delete[] checked;
}
