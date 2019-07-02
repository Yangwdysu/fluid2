#ifndef MFD_ARRAY_H
#define MFD_ARRAY_H

#include <list>
#include <map>
#include <algorithm>
#include <iostream>
#include <assert.h>
#include "Vec.h"

namespace mfd{

	class IArrayObject
	{
	public:
		IArrayObject() {};
		~IArrayObject() {};
		virtual void Reordering( int* ids, int size) = 0;
	};

	template<typename T>
	class Array : public IArrayObject
	{
	public:
		Array() { data = NULL; elementCount = 0; }
		~Array() { Release(); } 

		inline T* DataPtr() { return data; }

		void SetSpace( const int count ) { elementCount = count; Allocate(); }

		void Reset( const int count ) { Release(); SetSpace(count); }

		void Zero() { memset((void*)data, 0, elementCount*sizeof(T)); }

		//void SetScalar(unsigned int index, T val) { data[index] = val; }

		unsigned int ElementCount() { return elementCount; }

		virtual void Reordering( int* ids, int size)
		{
			if (size != elementCount)
			{
				std::cout << "array size do not match!" << std::endl;
				exit(0);
			}
			T* tp = new T[elementCount];
#pragma omp parallel for
			for (int i = 0; i < (int)elementCount; i++)
			{
				tp[i] = data[ids[i]];
			}

			memcpy(data, tp, elementCount*sizeof(T));

			delete[] tp;
		}

		inline T& operator [] (unsigned int id)
		{
			assert(id >= 0 && id < elementCount);
			return data[id];
		}

	protected:
		void Allocate() { data = new T[elementCount]; }
		void Release() { if( data != NULL ) delete[] data; }

	private:
		unsigned int elementCount;
		T* data;
	};

	template class Array<unsigned>;
	template class Array<float>;
	template class Array<double>;
	template class Array<Vector3f>;
	template class Array<Vector3d>;

	class ArrayManager
	{
	public:
		ArrayManager(){};
		~ArrayManager(){};

		void AddArray(std::string name, IArrayObject* arr) { arrs.insert(std::map<string,IArrayObject*>::value_type(name, arr)); }

		IArrayObject* GetArray( std::string key ) { return arrs[key]; }

		void Reordering( int* ids, int size)
		{
			std::map<std::string, IArrayObject*>::iterator iter;
			for (iter = arrs.begin(); iter != arrs.end(); iter++)
			{
				iter->second->Reordering(ids, size);
			}
		}

	private:
		std::map<std::string, IArrayObject*> arrs;
	};

}




#endif