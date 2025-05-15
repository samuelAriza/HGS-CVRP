// RuinAndRecreate.h
#ifndef RUIN_AND_RECREATE_H
#define RUIN_AND_RECREATE_H

#include "Individual.h"
#include "Params.h"
#include <vector>
#include <random>

class RuinAndRecreate {
public:
    RuinAndRecreate(Params& params);
    void apply(Individual& solution, int num_iterations);

private:
    Params& params_;
    std::mt19937& rng_;
    double cooling_factor_;

    std::vector<int> adjacent_string_removal(const Individual& solution);
    void greedy_insertion_with_blinks(Individual& solution, const std::vector<int>& removed);
};

#endif