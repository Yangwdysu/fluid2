#pragma once
#include <vector>
#include <map>
//#include <hash_map>
using namespace std;

#define REAL	double

#define   EPSILON   1e-6
#define PRECONDITIONING

//#define KEYMAPPING

class NeighborTemp{
public:
	NeighborTemp(){coff = 1.0f;}
	NeighborTemp(REAL c){coff = c;}
	~NeighborTemp(){};

	REAL coff;
	REAL pcon;
};

//typedef stdext::hash_map<int, int> KeyMap;

typedef map<int, int> KeyMap;
typedef map<int, NeighborTemp> KeyMapping;


class AnEquation
{
public:
	
	AnEquation(){ b=0.0f; x=0.0f; r=0.0f; p=0.0f; coff = 0.0f;}
	~AnEquation(){};

	class NeighborCoff{
	public:
		NeighborCoff(int id, REAL c){neighorid = id; coff = c; pcon = 0.0f;}
		~NeighborCoff(){};

		int neighorid;
		REAL coff;
		REAL pcon;
	};

	
	REAL b;
	REAL x;


	REAL r;
	REAL p;
	REAL z;
#ifdef KEYMAPPING
	KeyMap index_row;
#endif
	//map<int, int> index_col;
	//map<int, Neighbor> neib;
	REAL coff;
	vector<NeighborCoff> row;
	//KeyMapping onerow;
	//vector<NeighborCoff> col;
};

//class for solving Ax = b;
class LinearSystem
{
public:
	enum PreconditionerType
	{
		INCOMPLETE_CHOLESKY,
		JACOBI,
		SSOR
	};

	LinearSystem(){m_maxIteration = 50; m_maxError = 0.000001f; m_bPrecondition = true; m_limitIteration = 500; pretype = SSOR;}
	~LinearSystem(){
		m_equations.clear();
	}

	//conjugate gradient method
	bool SolvorCG(vector<REAL>& x0);

	bool SolvorPCG(vector<REAL>& x0);

	bool SolvorSOR(vector<REAL>& x0);


	void pushAnEquation(AnEquation& equ);

	int iterationNumber();
	REAL residualSquared();

	void writeToFile(string filename);
	void readFromFile( string filename );

	bool symmetry();

	int getEquationNumber();

private:
	void initialize(vector<REAL>& x0);
	void computeResidual();

	REAL computeRZ();

	REAL computeAlpha();

	void increX(REAL alpha);
	void increR(REAL alpha);

	void increP(REAL beta);

	void constructPreconditioner();
	void applyPreconditioner();

	REAL getThreshold();

private:
	//void constructColumns();
	//incomplete cholesky preconditioner
#ifdef KEYMAPPING
	void constructIndexs();
	void constructIncompleteCholeskyPreconditioner();
	void complementPreconditioner();
	void applyIncompleteCholeskyPreconditioner();
#endif	

	//jacobi preconditioner
	void constructJocobiPreconditioner();
	void applyJocobiPreconditioner();

	//SSOR preconditioner
	void constructSSORPreconditioner();
	void applySSORPreconditioner();

#ifdef KEYMAPPING
	void constructModifiedPreconditioner();		//modified incomplete cholesky preconditioner
	void constructVModifiedPreconditioner();	// a variant of modified incomplete cholesky preconditioner
#endif

public:
	int m_niter;
	int m_maxIteration;
	int m_limitIteration;
	REAL m_maxError;
	bool m_bPrecondition;
	vector<AnEquation> m_equations;
	PreconditionerType pretype;

	REAL scale;	//normalization cofficient for smoothing function
					//standing for the volume in a control
	//vector<AnEquation> m_eqs;
};

