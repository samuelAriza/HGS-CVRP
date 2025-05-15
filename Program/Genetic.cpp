// Genetic.cpp
#include "Genetic.h"

Genetic::Genetic(Params& params, RuinAndRecreate& rnr)
    : params(params),
      split(params),
      localSearch(params),
      population(params, split, localSearch, rnr),
      offspring(params),
      rnr(rnr) {}

void Genetic::run() {
    population.generatePopulation();
    int nbIter;
    int nbIterNonProd = 1;
    if (params.verbose) std::cout << "----- STARTING GENETIC ALGORITHM" << std::endl;
    for (nbIter = 0; nbIterNonProd <= params.ap.nbIter &&
                    (params.ap.timeLimit == 0 || (double)(clock() - params.startTime) / (double)CLOCKS_PER_SEC < params.ap.timeLimit);
         nbIter++) {
        crossoverOX(offspring, population.getBinaryTournament(), population.getBinaryTournament());
        localSearch.run(offspring, params.penaltyCapacity, params.penaltyDuration);
        rnr.apply(offspring, std::floor(params.gamma * params.nbClients));
        bool isNewBest = population.addIndividual(offspring, true);
        if (!offspring.eval.isFeasible && params.ran() % 2 == 0) {
            localSearch.run(offspring, params.penaltyCapacity * 10., params.penaltyDuration * 10.);
            if (offspring.eval.isFeasible) isNewBest = (population.addIndividual(offspring, false) || isNewBest);
        }
        if (isNewBest) nbIterNonProd = 1;
        else nbIterNonProd++;
        if (nbIter % params.ap.nbIterPenaltyManagement == 0) population.managePenalties();
        if (nbIter % params.ap.nbIterTraces == 0) population.printState(nbIter, nbIterNonProd);
        if (params.ap.timeLimit != 0 && nbIterNonProd == params.ap.nbIter) {
            population.restart();
            nbIterNonProd = 1;
        }
    }
    if (params.verbose) std::cout << "----- GENETIC ALGORITHM FINISHED AFTER " << nbIter
                                 << " ITERATIONS. TIME SPENT: " << (double)(clock() - params.startTime) / (double)CLOCKS_PER_SEC
                                 << std::endl;
}

void Genetic::crossoverOX(Individual& result, const Individual& parent1, const Individual& parent2) {
    std::vector<bool> freqClient(params.nbClients + 1, false);
    std::uniform_int_distribution<> distr(0, params.nbClients - 1);
    int start = distr(params.ran);
    int end = distr(params.ran);
    while (end == start) end = distr(params.ran);
    int j = start;
    while (j % params.nbClients != (end + 1) % params.nbClients) {
        result.chromT[j % params.nbClients] = parent1.chromT[j % params.nbClients];
        freqClient[result.chromT[j % params.nbClients]] = true;
        j++;
    }
    for (int i = 1; i <= params.nbClients; i++) {
        int temp = parent2.chromT[(end + i) % params.nbClients];
        if (!freqClient[temp]) {
            result.chromT[j % params.nbClients] = temp;
            j++;
        }
    }
    split.generalSplit(result, parent1.eval.nbRoutes);
}