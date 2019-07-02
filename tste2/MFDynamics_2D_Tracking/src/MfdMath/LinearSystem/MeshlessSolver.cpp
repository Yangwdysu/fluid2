#include <assert.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include "MeshlessSolver.h"



// r=Ax-b
void LinearSystem::computeResidual()
{
	const int nequals = m_equations.size();
	for (int i = 0; i < nequals; i++)
	{
		REAL ri = m_equations[i].b;

		vector<AnEquation::NeighborCoff>& neighbors = m_equations[i].row;
		const int nneighbors = m_equations[i].row.size();
		for (int j = 0; j < nneighbors; j++)
		{
			ri -= neighbors[j].coff*m_equations[neighbors[j].neighorid].x;
		}
		m_equations[i].r = ri;
	}
}

//r0=Ax0-b, p0=-r0
void LinearSystem::initialize(vector<REAL>& x0)
{
	assert(x0.size() == m_equations.size());
	//constructColumns();

	const int nequals = m_equations.size();
	for (int i = 0; i < nequals; i++)
	{
		m_equations[i].x = x0[i];
	}
	computeResidual();

	if (m_bPrecondition)
	{
		constructPreconditioner();
		
		applyPreconditioner();
	}

	for (int i = 0; i < nequals; i++)
	{
		if (m_bPrecondition)
		{
			m_equations[i].p = m_equations[i].z;
		}
		else
		{
			m_equations[i].p = m_equations[i].r;
		}
	}
}

//alpha = r(k)*r(k)/p(k)Ap(k);
REAL LinearSystem::computeAlpha()
{
	REAL rr = 0.0f;
	REAL pAp = 0.0f;

	const int neqals = m_equations.size();
	for (int i = 0; i < neqals; i++)
	{
		if (m_bPrecondition)
			rr += m_equations[i].r*m_equations[i].z;
		else
			rr += m_equations[i].r*m_equations[i].r;
		
		REAL Ap = 0.0f;

		vector<AnEquation::NeighborCoff>& neighbors = m_equations[i].row;
		const int nneighbors = m_equations[i].row.size();
		for (int j = 0; j < nneighbors; j++)
		{
			Ap += neighbors[j].coff*m_equations[neighbors[j].neighorid].p;
		}

		pAp += m_equations[i].p*Ap;
	}
	/*if (pAp > 0.0f)
	{
		cout << "number:" << m_niter << " no positive" << endl;
		exit(0);
	}*/
	return rr/pAp;
}

//x(k+1) += x(k)+alpha*p(k);
void LinearSystem::increX( REAL alpha )
{
	const int neqals = m_equations.size();
	for (int i = 0; i < neqals; i++)
	{
		m_equations[i].x += alpha*m_equations[i].p;
	}
}

//r(k+1) += r(k) alpha*Ap(k)
void LinearSystem::increR( REAL alpha )
{
	const int neqals = m_equations.size();
	for (int i = 0; i < neqals; i++)
	{
		REAL Ap = 0.0f;

		vector<AnEquation::NeighborCoff>& neighbors = m_equations[i].row;
		const int nneighbors = m_equations[i].row.size();
		for (int j = 0; j < nneighbors; j++)
		{
			Ap += neighbors[j].coff*m_equations[neighbors[j].neighorid].p;
		}
		m_equations[i].r -= alpha*Ap;
	}
}

//p(k+1) = -r(k+1)+beta*p(k);
void LinearSystem::increP( REAL beta )
{
	const int nequals = m_equations.size();
	for (int i = 0; i < nequals; i++)
	{
		m_equations[i].p = beta*m_equations[i].p;
		if (m_bPrecondition)
			m_equations[i].p += m_equations[i].z;
		else
			m_equations[i].p += m_equations[i].r;
	}
}

REAL LinearSystem::residualSquared()
{
	REAL rr = 0.0f;
	const int nequals = m_equations.size();
	for (int i = 0; i < nequals; i++)
	{
		rr += m_equations[i].r*m_equations[i].r;
	}
	return rr;
}


bool LinearSystem::SolvorCG(vector<REAL>& x0)
{
	REAL threshold = FLT_MAX;
	m_bPrecondition = false;

	assert(x0.size() == m_equations.size());

	initialize(x0);

	REAL residual = residualSquared();
	REAL initRes = residual;
	if (residual < EPSILON)
	{
		threshold = 0.0f;
	}
	m_niter = 0;
	REAL maxerrorsquared = m_maxError*m_maxError;
	while (threshold > m_maxError && m_niter < m_maxIteration && m_niter < m_limitIteration)
	{
		REAL alpha = computeAlpha();
		increX(alpha);
		increR(alpha);
		//computeResidual();
		REAL rrnew = residualSquared();
		REAL beta = rrnew/residual;
		increP(beta);

		//cout << "alpha: " << "\t" << alpha << "\t" << " residual: " << "\t" << rrnew << endl;

		residual = rrnew;
		m_niter++;

		threshold = getThreshold();
		cout << "(SolvorCG) iteration " << m_niter << " threshold: " << threshold << endl;
	}
// 	cout << "(SolvorCG) final residual: " << sqrt(residual) << " initial residual: " << sqrt(initRes) << " acuracy: " << sqrt(residual/initRes) << endl;
// 	cout << "(SolvorCG)time of initialization: " << t_initialize._end-t_initialize._begin << endl;
// 	cout << "(SolvorCG)time of each iteration: " << t_iter._end-t_iter._begin << endl;

	if (m_niter >= m_maxIteration)
		return false;

	return true;
}


bool LinearSystem::SolvorPCG(vector<REAL>& x0)
{
	REAL threshold = FLT_MAX;
	m_bPrecondition = true;

	assert(x0.size() == m_equations.size());

	initialize(x0);

	REAL residual = residualSquared();
	REAL initRes = residual;
	REAL initThresh = getThreshold();
	cout << "(SolvorPCG) Initial threshold: " << initThresh << endl;
	if (residual < EPSILON)
	{
		threshold = 0.0f;
	}
	m_niter = 0;
	REAL maxerrorsquared = m_maxError*m_maxError;
	while (threshold > m_maxError && m_niter < m_maxIteration && m_niter < m_limitIteration)
	{
		REAL alpha = computeAlpha();
		
		REAL rzold = computeRZ();
		increX(alpha);
		computeResidual();
		//increR(alpha);
		
		applyPreconditioner();

		REAL rznew = computeRZ();
		REAL beta = rznew/rzold;
		increP(beta);

		//cout << "alpha: " << "\t" << alpha << "\t" << " residual: " << "\t" << rrnew << endl;

		residual = residualSquared();
		m_niter++;

		threshold = getThreshold();
//		threshold /= initThresh;
 		cout << "(SolvorPCG) iteration " << m_niter << " threshold: " << threshold << endl;
// 
// 		getchar();
	}

	cout << "(SolvorCG) final threshhold: " << threshold << " initial threshhold: " << initThresh << " acuracy: " << initThresh/threshold << endl;
// 	cout << "(SolvorCG) final residual: " << sqrt(residual) << " initial residual: " << sqrt(initRes) << " acuracy: " << sqrt(residual/initRes) << endl;
// //	cout << "(SolvorPCG) time of initialization: " << t_initialize._end-t_initialize._begin << endl;
// 	cout << "(SolvorPCG) time of each iteration: " << t_iter._end-t_iter._begin << endl;
// 	cout << "(SolvorPCG) time of each apply preconditioner: " << t_applyconditioner._end-t_applyconditioner._begin << endl;
	cout << "(SolvorCG) Total iteration: " << m_niter << endl;

	if (m_niter >= m_maxIteration)
		return false;

	return true;
}

// see http://en.wikipedia.org/wiki/Successive_over-relaxation for details
bool LinearSystem::SolvorSOR(vector<REAL>& x0)
{
	REAL omiga = 1.8f;
	REAL threshold = FLT_MAX;

	const int neqals = m_equations.size();
	for (int i = 0; i < neqals; i++)
	{
		m_equations[i].x = x0[i];
	}
	computeResidual();
	REAL initialResidual = residualSquared();
	cout << "(SolvorSOR) initial residual: " << initialResidual << endl;
	if (abs(initialResidual) < EPSILON)
	{
		threshold = 0.0f;
	}
	m_niter = 0;
	
	REAL sigma = 0.0f;
	REAL residual = 0.0f;
	while (threshold > m_maxError && m_niter < m_maxIteration && m_niter < m_limitIteration) 
	{
		for (int i = 0; i < neqals; i++)
		{
			AnEquation& eqi = m_equations[i];
			sigma = 0.0f;

			vector<AnEquation::NeighborCoff>& neighbors = eqi.row;
			const int nneighbors = eqi.row.size();
			for (int ne = 0; ne < nneighbors; ne++)
			{
				int j = neighbors[ne].neighorid;
				if (j != i)
				{
					sigma += neighbors[ne].coff*m_equations[j].x;
				}
			}
			sigma = ((eqi.b-sigma)/eqi.coff - eqi.x)*omiga+eqi.x;
			eqi.x = sigma;
		}	


		computeResidual();
		residual = residualSquared();

		threshold = getThreshold();
		m_niter++;

		

		//cout << "(SolverSOR) iteration " << m_niter << " threshold: " << threshold << endl;
		//getchar();
	};
	computeResidual();

	return true;
}

int LinearSystem::iterationNumber()
{
	return m_niter;
}

void LinearSystem::pushAnEquation( AnEquation& equ )
{
	m_equations.push_back(equ);
}

int LinearSystem::getEquationNumber()
{
	return m_equations.size();
}


void LinearSystem::writeToFile(string filename)
{
	ofstream output(filename.c_str(), ios::out|ios::binary);
	const int neqals = m_equations.size();
	output.write((char*)&neqals, sizeof(int));
	for (int i = 0; i < neqals; i++)
	{
		AnEquation& eq = m_equations[i];
		output.write((char*)&eq.b, sizeof(REAL));

		vector<AnEquation::NeighborCoff>& neighbors = m_equations[i].row;
		const int nneighbors = m_equations[i].row.size();

		output.write((char*)&nneighbors, sizeof(int));
		for (int j = 0; j < nneighbors; j++)
		{
			AnEquation::NeighborCoff& neighbour = neighbors[j];
			output.write((char*)&neighbour.neighorid, sizeof(int));
			output.write((char*)&neighbour.coff, sizeof(REAL));
		}

	}
	output.close();
	cout << "Written " <<neqals << " equations to " << filename << endl;
}

void LinearSystem::readFromFile( string filename )
{
	if (m_equations.size() > 0)
	{
		m_equations.clear();
	}

	ifstream input(filename.c_str(), ios::in|ios::binary);
	int neqals;
	input.read((char*)&neqals, sizeof(int));
	for (int i = 0; i < neqals; i++)
	{
		AnEquation eqi;
		input.read((char*)&eqi.b, sizeof(REAL));

		int nneighbors;
		input.read((char*)&nneighbors, sizeof(int));
		for (int j = 0; j < nneighbors; j++)
		{
			int index;
			REAL coff;
			input.read((char*)&index, sizeof(int));
			input.read((char*)&coff, sizeof(REAL));
			eqi.row.push_back(AnEquation::NeighborCoff(index, coff));
		}
		m_equations.push_back(eqi);
	}
	input.close();
}

bool LinearSystem::symmetry()
{
	const int neqals = m_equations.size();
	for (int i = 0; i < neqals; i++)
	{
		vector<AnEquation::NeighborCoff>& neighbors = m_equations[i].row;
		const int nneighbors = m_equations[i].row.size();
		for (int j = 0; j < nneighbors; j++)
		{
			int neigbhourid = neighbors[j].neighorid;
			AnEquation& eqj = m_equations[neigbhourid];
			vector<AnEquation::NeighborCoff>& neighborsj = eqj.row;
			const int nneighborsj = neighborsj.size();
			int bb = 0;
			for (int k = 0; k < nneighborsj; k++)
			{
				if (neighborsj[k].neighorid == i)
				{
					if (neighborsj[k].coff-neighbors[j].coff > 0.00001f || neighborsj[k].coff-neighbors[j].coff < -0.00001f)
					{
						return false;
					}
					else
						bb ++;
				}
			}
			if( bb >= nneighborsj) return false;
		}
	}
	return true;
}

#ifdef KEYMAPPING
void LinearSystem::constructIndexs()
{
	const int nequals = m_equations.size();
	for (int i = 0; i < nequals; i++)
	{
		vector<AnEquation::NeighborCoff>& neighbors = m_equations[i].row;
		KeyMap& indexs = m_equations[i].index_row;
		const int nneighbors = neighbors.size();
		for (int ne = 0; ne < nneighbors; ne++)
		{
			indexs[neighbors[ne].neighorid] = ne;
		}
	}
}


/*void LinearSystem::constructColumns()
{
	const int nequals = m_equations.size();
	for (int i = 0; i < nequals; i++)
	{
		vector<AnEquation::NeighborCoff>& rows = m_equations[i].row;
		const int nneighbors = rows.size();
		for (int j = 0; j < nneighbors; j++)
		{
			m_equations[rows[j].neighorid].col.push_back(AnEquation::NeighborCoff(i, rows[j].coff));
		}
	}
}*/

//A = LL^t
void LinearSystem::constructIncompleteCholeskyPreconditioner()
{
//	Timer t;
//	t.start();
	const int nequals = m_equations.size();
	for (int a = 0; a < nequals; a++)
	{
		AnEquation& eqa = m_equations[a];
		vector<AnEquation::NeighborCoff>& rowa = eqa.row;		// because of symmetry
		const int nrow = rowa.size();

		REAL Maa = 0.0f;//eqa.coff;
		for (int ne = 0; ne < nrow; ne++)
		{
			if (rowa[ne].neighorid < a)
			{
				Maa -= rowa[ne].pcon*rowa[ne].pcon;
			}
			else if (rowa[ne].neighorid == a)
			{
				Maa += rowa[ne].coff;
			}
		}
		Maa = 1.0f/sqrt(Maa);
		eqa.coff = Maa;
		
		KeyMap& keya = eqa.index_row;
		for (int ne = 0; ne < nrow; ne++)
		{
			if (rowa[ne].neighorid > a)
			{
				REAL Mai = rowa[ne].coff;
				vector<AnEquation::NeighborCoff>& rown = m_equations[rowa[ne].neighorid].row;
				KeyMap& keyn = m_equations[rowa[ne].neighorid].index_row;
				KeyMap::iterator& a_itor = keya.begin();
				KeyMap::iterator& n_itor = keyn.begin();
				int aj, nj;
				while (a_itor != keya.end() && n_itor != keyn.end())
				{
					aj = a_itor->first;
					nj = n_itor->first;
					if (aj >= a || nj >= a)
					{
						break;
					}
					else
					{
						if (aj < nj)
						{
							a_itor++;
						}
						else if (aj > nj)
						{
							n_itor++;
						}
						else
						{
							Mai -= rowa[a_itor->second].pcon*rown[n_itor->second].pcon;
							a_itor++;
							n_itor++;
						}
					}
				}
				//REAL Mai = rowa[ne].coff;
/*				vector<AnEquation::NeighborCoff>& rowi = m_equations[rowa[ne].neighorid].row;
				KeyMap& index = m_equations[rowa[ne].neighorid].index_row;
				for (int j = 0; j < nrow; j++)
				{
					if (rowa[j].neighorid < a)
					{
						if (index.find(rowa[j].neighorid) != index.end())
						{
							Mai -= rowa[j].pcon*rowi[index[rowa[j].neighorid]].pcon;
						}
					}
				}
*/
				rowa[ne].pcon = Mai*Maa;
				//make the matrix symmetry
				int k = rowa[ne].neighorid;
				m_equations[k].row[m_equations[k].index_row[a]].pcon = rowa[ne].pcon;
				//neighbors[ne].pcon = m_equations[j].row[m_equations[j].index_row[i]].pcon;
			}
		}
	}
//	t.stop();
//	cout << "(constructPreconditioner) time of each construction: " << t._end-t._begin << endl;
//	exit(0);
	//complementPreconditioner();
}
#endif

void LinearSystem::constructJocobiPreconditioner()
{

}

void LinearSystem::applyJocobiPreconditioner()
{
	const int nequals = m_equations.size();
	for (int i = 0; i < nequals; i++)
	{
		AnEquation& eqi = m_equations[i];
		eqi.z = eqi.r/eqi.coff;
	}
}

#ifdef KEYMAPPING
void LinearSystem::constructModifiedPreconditioner()
{
	const int nequals = m_equations.size();
	for (int a = 0; a < nequals; a++)
	{
		AnEquation& eqa = m_equations[a];
		vector<AnEquation::NeighborCoff>& rowa = eqa.row;		// because of symmetry
		const int nrow = rowa.size();

		REAL Maa = rowa[eqa.index_row[a]].coff;
		for (int ne = 0; ne < nrow; ne++)
		{
			if (rowa[ne].neighorid < a)
			{
				Maa -= rowa[ne].pcon*rowa[ne].pcon;
			}
		}
		Maa = sqrt(Maa);
		rowa[eqa.index_row[a]].pcon = Maa;

		for (int ne = 0; ne < nrow; ne++)
		{
			if (rowa[ne].neighorid > a)
			{
				REAL Mai = rowa[ne].coff;
				vector<AnEquation::NeighborCoff>& rowi = m_equations[rowa[ne].neighorid].row;
				KeyMap& index = m_equations[rowa[ne].neighorid].index_row;
				for (int j = 0; j < nrow; j++)
				{
					if (rowa[j].neighorid < a)
					{
						if (index.find(rowa[j].neighorid) != index.end())
						{
							Mai -= rowa[j].pcon*rowi[index[rowa[j].neighorid]].pcon;
						}
					}
				}
				rowa[ne].pcon = Mai/Maa;
				//make the matrix symmetry
				int k = rowa[ne].neighorid;
				m_equations[k].row[m_equations[k].index_row[a]].pcon = rowa[ne].pcon;
				//neighbors[ne].pcon = m_equations[j].row[m_equations[j].index_row[i]].pcon;
			}
		}
	}
}


void LinearSystem::constructVModifiedPreconditioner()
{
	
}


void LinearSystem::applyIncompleteCholeskyPreconditioner()
{
	const int nequals = m_equations.size();
	for (int i = 0; i < nequals; i++)
	{
		AnEquation& eqi = m_equations[i];
		REAL z = eqi.r;
		vector<AnEquation::NeighborCoff>& rowa = eqi.row;
		const int nrow = rowa.size();
		REAL Mii = eqi.coff;// = rowa[eqi.index_row[i]].pcon;
		for (int ne = 0; ne < nrow; ne++)
		{
			int j = rowa[ne].neighorid;
			if (j < i)
			{
				z -= rowa[ne].pcon*m_equations[j].z;
			}
		}
		eqi.z = z*Mii;
	}

	for (int i = nequals-1; i >= 0; i--)
	{
		AnEquation& eqi = m_equations[i];
		REAL z = eqi.z;
		vector<AnEquation::NeighborCoff>& rowa = eqi.row;
		const int nrow = rowa.size();
		REAL Mii = eqi.coff;// = rowa[eqi.index_row[i]].pcon;
		for (int ne = 0; ne < nrow; ne++)
		{
			int j = rowa[ne].neighorid;
			if (j > i)
			{
				z -= rowa[ne].pcon*m_equations[j].z;
			}
		}
		eqi.z = z*Mii;
	}
}
#endif


void LinearSystem::constructSSORPreconditioner()
{
	REAL omiga = 1.7f;
	REAL inv_omiga = 1.0f/omiga;
	REAL coe = omiga/(2.0f-omiga);
	coe = sqrt(coe);

	const int nequals = m_equations.size();
	vector<REAL> diag;
	diag.resize(nequals);
	for (int i = 0; i < nequals; i++)
	{
		diag[i] = coe*sqrt(1.0f/m_equations[i].coff);
	}

	for (int i = 0; i < nequals; i++)
	{
		AnEquation& eqi = m_equations[i];
		vector<AnEquation::NeighborCoff>& rowi = eqi.row;
		int nrow = rowi.size();
		for (int ne = 0; ne < nrow; ne++)
		{
			int j = rowi[ne].neighorid;
			if (j > i)
			{
				rowi[ne].pcon = diag[i]*rowi[ne].coff;
			}
			else if (j < i)
			{
				rowi[ne].pcon = diag[j]*rowi[ne].coff;
			}
			else
			{
				rowi[ne].pcon = inv_omiga*diag[i]*rowi[ne].coff;
				eqi.coff = 1.0f/rowi[ne].pcon;	//save the inverse
			}
		}
	}
}

void LinearSystem::applySSORPreconditioner()
{
	const int nequals = m_equations.size();

	for (int i = nequals-1; i >= 0; i--)
	{
		AnEquation& eqi = m_equations[i];
		REAL z = eqi.r;
		vector<AnEquation::NeighborCoff>& rowa = eqi.row;
		const int nrow = rowa.size();
		REAL Mii = eqi.coff;// = rowa[eqi.index_row[i]].pcon;
		for (int ne = 0; ne < nrow; ne++)
		{
			int j = rowa[ne].neighorid;
			if (j > i)
			{
				z -= rowa[ne].pcon*m_equations[j].z;
			}
		}
		eqi.z = z*Mii;
	}

	for (int i = 0; i < nequals; i++)
	{
		AnEquation& eqi = m_equations[i];
		REAL z = eqi.z;
		vector<AnEquation::NeighborCoff>& rowa = eqi.row;
		const int nrow = rowa.size();
		REAL Mii = eqi.coff;// = rowa[eqi.index_row[i]].pcon;
		for (int ne = 0; ne < nrow; ne++)
		{
			int j = rowa[ne].neighorid;
			if (j < i)
			{
				z -= rowa[ne].pcon*m_equations[j].z;
			}
		}
		eqi.z = z*Mii;
	}
}

#ifdef KEYMAPPING
//feed the upper triangular matrix
void LinearSystem::complementPreconditioner()
{
	const int nequals = m_equations.size();
	for (int i = 0; i < nequals; i++)
	{
		vector<AnEquation::NeighborCoff>& neighbors = m_equations[i].row;
		KeyMap& indexs = m_equations[i].index_row;
		const int nneighbors = neighbors.size();
		for (int ne = 0; ne < nneighbors; ne++)
		{
			if (neighbors[ne].neighorid < i)
			{
				int j = neighbors[ne].neighorid;
				neighbors[ne].pcon = m_equations[j].row[m_equations[j].index_row[i]].pcon;
			}
		}
	}
}
#endif

REAL LinearSystem::computeRZ()
{
	REAL rz = 0.0f;
	const int nequals = m_equations.size();
	for (int i = 0; i < nequals; i++)
	{
		rz += m_equations[i].r*m_equations[i].z;
	}
	return rz;
}


void LinearSystem::constructPreconditioner()
{
	switch(pretype)
	{
	case INCOMPLETE_CHOLESKY:
#ifdef KEYMAPPING
		constructIndexs();
		constructIncompleteCholeskyPreconditioner();
#endif
		break;
	case JACOBI:
		break;
	case SSOR:
		constructSSORPreconditioner();
	default :
		break;
	}
}

void LinearSystem::applyPreconditioner()
{
	switch(pretype)
	{
	case INCOMPLETE_CHOLESKY:
#ifdef KEYMAPPING
		applyIncompleteCholeskyPreconditioner();
#endif
		break;
	case JACOBI:
		applyJocobiPreconditioner();
		break;
	case SSOR:
		applySSORPreconditioner();
	default :
		break;
	}
	
}

REAL LinearSystem::getThreshold()
{
  	REAL thresh = 0.0f;
	const int nequals = m_equations.size();
	for (int i = 0; i < nequals; i++)
	{
		thresh += abs(m_equations[i].r);
	}
	thresh /= (nequals);
	thresh *= scale;

// 
// 	const int neqals = m_equations.size();
// 	for (int i = 0; i < neqals; i++)
// 	{
// 		REAL Ap = 0.0f;
// 
// 		vector<AnEquation::NeighborCoff>& neighbors = m_equations[i].row;
// 		const int nneighbors = m_equations[i].row.size();
// 		for (int j = 0; j < nneighbors; j++)
// 		{
// 			Ap += neighbors[j].coff*m_equations[neighbors[j].neighorid].r;
// 		}
// 
// 		thresh += m_equations[i].r*Ap;
// 	}

 	return thresh;
}




