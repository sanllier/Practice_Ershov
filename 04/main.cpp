#include <fstream>
#include <ctime>
#include <vector>
#include <cmath>

#include "helpers.h"
#include "parparser.h"

//---------------------------------------------------------------------------------

float a = -100.0;
float b = 100.0;

float mutationProbability = 0.5;
float mutationFrac = 0.3;

float crossoverProbability = 0.5;

float battleProbability = 0.5;
float winProbability = 0.9;

float migratesIndividsPart = 0.4;
int migrationFreq = 15;

int islandSize = 256;
int problemSize = 10;

int iterationsThreshold = 3000;

//---------------------------------------------------------------------------------

float spherical(const vector<float>& genotype) {
	float temp = 0.0;
	for (int i = 0; i < genotype.size(); ++i) {
		temp += genotype[i] * genotype[i];
	}

	return  sqrt(temp);
}

float rozenbrock(const vector<float>& genotype) {
	float temp = 0.0;
	for (int i = 0; i < genotype.size() - 1; ++i) {
		const float cur = genotype[i];
		const float curSqrt = cur * cur;
		const float next = genotype[i + 1];
		temp += 100.0 * (curSqrt - next) * (curSqrt - next) + (cur - 1.0) * (cur - 1.0);
	}

	return temp;
}

float rastrigin(const vector<float>& genotype) {
	float temp = 0.0;
	for (int i = 0; i < genotype.size(); ++i) {
		const float cur = genotype[i];
		const float cosVal = cos(6.28 * cur);
		temp += cur * cur + 10.0 * cosVal + 10.0;
	}

	return temp;
}

float eval(const vector<float>& genotype) {
	//return spherical(genotype);
	//return rozenbrock(genotype);
	return rastrigin(genotype);
}

//---------------------------------------------------------------------------------

inline float frand() {
	return float(rand()) / float(RAND_MAX) * 0.9999;
}

struct Individ {
	float fitness;
	vector<float> genotype;

	void init(int size) {
		genotype.resize(size);
		for (int i = 0; i < size; ++i) {
			genotype[i] = a + (b - a) * frand();
		}
	}
};

void pointCrossover(bool onePoint, Individ& a, Individ& b) {
	const int crossoverAPoint = rand() % a.genotype.size();
	int crossoverBPoint;
	if (onePoint) {
		crossoverBPoint = a.genotype.size();
	} else {
		crossoverBPoint = rand() % (a.genotype.size() - crossoverAPoint) + crossoverAPoint + 1;
	}
	
	const int crossSize = crossoverBPoint - crossoverAPoint;
	vector<float> temp(crossSize);
	
	memcpy(&temp[0], &a.genotype[crossoverAPoint], crossSize * sizeof(float));
	memcpy(&a.genotype[crossoverAPoint], &b.genotype[crossoverAPoint], crossSize * sizeof(float));
	memcpy(&b.genotype[crossoverAPoint], &temp[0], crossSize * sizeof(float));
}

void crossoverPopulation(vector<Individ>& population) {
for (int i = 0; i <= population.size() / 2; ++i) {
		if (frand() >= crossoverProbability) continue;

		const int aIndex = rand() % population.size();
		const int bIndex = rand() % population.size();
		if (aIndex == bIndex) continue;

		pointCrossover(true, population[aIndex], population[bIndex]);
	}	
}

void mutateIndivid(Individ& ind) {
	for (int i = 0; i < ind.genotype.size(); ++i) {
		if (frand() < mutationProbability) {
			const float delta = frand() * mutationFrac;
			ind.genotype[i] += (rand() % 2 == 0) ? delta : -delta;
		}
	}
}

void mutatePopulation(vector<Individ>& population) {
	for (int i = 0; i < population.size(); ++i) {
		mutateIndivid(population[i]);
	}
}

void select(vector<Individ>& population) {
	for (int i = 0; i <= population.size() / 2; ++i) {
		if (frand() >= battleProbability) continue;

		const int aIndex = rand() % population.size();
		const int bIndex = rand() % population.size();
		if (aIndex == bIndex) continue;

		const float aF = eval(population[aIndex].genotype);
		const float bF = eval(population[bIndex].genotype);
		population[aIndex].fitness = aF;
		population[bIndex].fitness = bF;

		const float r = frand();
		const int indSize = population[aIndex].genotype.size();

		if ((aF < bF && r < winProbability) || (aF > bF && r > winProbability)) {
			memcpy(&population[bIndex].genotype[0], &population[aIndex].genotype[0], indSize * sizeof(float));
		} else {
			memcpy(&population[aIndex].genotype[0], &population[bIndex].genotype[0], indSize * sizeof(float));
		}
	}
}

void migrate(vector<Individ>& population, int next, int prev) {
	const int indSize = population[0].genotype.size();
	const int migrationSize = indSize * migratesIndividsPart;
	vector<float> buf(indSize, 0.0);

	MPI_Status status;

	for (int i = 0; i < migrationSize; ++i) {
		const int target = rand() % population.size();

		MPICHECK(MPI_Sendrecv(&population[target].genotype[0], indSize, MPI_FLOAT, next, 0, 
			&buf[0], indSize, MPI_FLOAT, prev, 0, MPI_COMM_WORLD, &status));

		memcpy(&population[target].genotype[0], &buf[0], indSize * sizeof(float));
	}
}

float getBestFitness(vector<Individ>& population, int rank, int commSize) {
	MPI_Status status;

	float localBest = 0.0;
	for (int i = 0; i < population.size(); ++i) {
		if (i == 0 || population[i].fitness < localBest) {
			localBest = population[i].fitness;
		}
	}

	float globalBest = localBest;
	if (rank == MASTER) {
		for (int i = 1; i < commSize; ++i) {
			float remoteBest;
			MPICHECK(MPI_Recv(&remoteBest, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status));
			if (remoteBest < globalBest) globalBest = remoteBest;
		}
	} else {
		MPICHECK(MPI_Send(&localBest, 1, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD));
	}

	return globalBest;
}

//---------------------------------------------------------------------------------

int main(int argc, char** argv) {

	//parparser parser(argc, argv);

	//long inititalPosition = parser.get("x").asLong();

	srand(time(0));

	//-----------------------------------------------------------------------------

	MPICHECK(MPI_Init(&argc, &argv));
	//-----------------------------------------------------------------------------

	int commSize = 0;
	int rank = 0;
	MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &commSize));
	MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

	const int nextProc = rank == commSize - 1 ? 0 : rank + 1;
	const int prevProc = rank == 0 ? commSize - 1 : rank - 1;

	vector<Individ> population(islandSize);
	for (int i = 0; i < islandSize; ++i) {
		population[i].init(problemSize);
	}

	for (int i = 1; i <= iterationsThreshold; ++i) {
		select(population);
		crossoverPopulation(population);
		mutatePopulation(population);

		if (i % migrationFreq == 0) {
			migrate(population, nextProc, prevProc);
		}

		cout << getBestFitness(population, rank, commSize) << "\n";
	}
	
	//-----------------------------------------------------------------------------


	//-----------------------------------------------------------------------------
	MPICHECK(MPI_Finalize());
	return 0;
}
