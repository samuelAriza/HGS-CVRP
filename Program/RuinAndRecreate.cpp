// RuinAndRecreate.cpp
#include "RuinAndRecreate.h"
#include <cmath>
#include <algorithm>
#include <numeric>

// Constructor: Inicializa los parámetros y calcula el factor de enfriamiento para el simulated annealing
RuinAndRecreate::RuinAndRecreate(Params& params)
    : params_(params), rng_(params.ran) {
    // Calcula el factor de enfriamiento geométrico basado en T_0, T_f, gamma y número de clientes
    cooling_factor_ = std::pow(params_.final_temperature / params_.initial_temperature,
                              1.0 / (params_.gamma * params_.nbClients));
}

// Método principal: Aplica Ruin-and-Recreate con simulated annealing (Algoritmo 1 del paper)
void RuinAndRecreate::apply(Individual& solution, int num_iterations) {
    Individual best_solution = solution;       // Mejor solución encontrada
    Individual current_solution = solution;    // Solución actual en el proceso
    double temperature = params_.initial_temperature; // Temperatura inicial para SA
    std::uniform_real_distribution<> dist(0.0, 1.0); // Distribución para decisiones aleatorias

    for (int i = 0; i < num_iterations; ++i) {
        // Paso 1: Remover clientes usando Adjacent String Removal
        auto removed_customers = adjacent_string_removal(current_solution);
        if (removed_customers.empty()) continue; // Si no se removió nada, pasa a la siguiente iteración

        // Paso 2: Crear una nueva solución a partir de la actual y reconstruir rutas
        Individual new_solution = current_solution;
        greedy_insertion_with_blinks(new_solution, removed_customers);

        // Paso 3: Evaluar la nueva solución
        new_solution.evaluateCompleteCost(params_);

        // Paso 4: Simulated Annealing - Decidir si aceptar la nueva solución
        double delta_z = current_solution.eval.penalizedCost - new_solution.eval.penalizedCost;
        if (delta_z > 0 || std::log(dist(rng_)) < delta_z / temperature) {
            current_solution = new_solution;
            // Actualizar la mejor solución si la nueva es mejor
            if (new_solution.eval.penalizedCost < best_solution.eval.penalizedCost) {
                best_solution = new_solution;
            }
        }

        // Paso 5: Reducir la temperatura
        temperature *= cooling_factor_;
    }

    // Actualizar la solución original con la mejor encontrada
    solution = best_solution;
}

// Método para remover clientes usando Adjacent String Removal
std::vector<int> RuinAndRecreate::adjacent_string_removal(const Individual& solution) {
    std::vector<int> removed_customers;
    int target_remove = std::round(params_.avg_customers_to_remove * params_.nbClients);
    std::vector<bool> is_removed(params_.nbClients + 1, false);

    // Crear una lista de clientes disponibles una sola vez para mejorar eficiencia
    std::vector<int> available_customers;
    available_customers.reserve(params_.nbClients);
    for (int i = 1; i <= params_.nbClients; ++i) {
        available_customers.push_back(i);
    }

    while (removed_customers.size() < target_remove) {
        if (available_customers.empty()) break;

        // Seleccionar un cliente semilla aleatoriamente
        std::uniform_int_distribution<> dist(0, available_customers.size() - 1);
        int seed_idx = dist(rng_);
        int seed = available_customers[seed_idx];
        int string_length = std::min(params_.max_string_length, target_remove - static_cast<int>(removed_customers.size()));
        std::vector<int> string = {seed};
        is_removed[seed] = true;
        available_customers.erase(available_customers.begin() + seed_idx);

        // Buscar clientes cercanos para formar una cadena
        for (int i = 1; i < string_length && !available_customers.empty(); ++i) {
            int closest = -1;
            double min_dist = std::numeric_limits<double>::max();
            int closest_idx = -1;

            for (int j = 0; j < static_cast<int>(available_customers.size()); ++j) {
                int c = available_customers[j];
                double dist = params_.timeCost[seed][c];
                if (dist < min_dist) {
                    min_dist = dist;
                    closest = c;
                    closest_idx = j;
                }
            }

            if (closest == -1) break;
            string.push_back(closest);
            is_removed[closest] = true;
            available_customers.erase(available_customers.begin() + closest_idx);
        }

        removed_customers.insert(removed_customers.end(), string.begin(), string.end());
    }

    return removed_customers;
}

// Método para reconstruir rutas usando Greedy Insertion with Blinks
void RuinAndRecreate::greedy_insertion_with_blinks(Individual& solution, const std::vector<int>& removed) {
    // Eliminar clientes de las rutas
    for (int customer : removed) {
        for (auto& route : solution.chromR) {
            auto it = std::find(route.begin(), route.end(), customer);
            if (it != route.end()) {
                route.erase(it);
                break;
            }
        }
    }

    // Ordenar clientes removidos (Random, Demand, Close, Far)
    std::vector<int> sorted_removed = removed;
    std::uniform_int_distribution<> dist(0, 3);
    int sort_method = dist(rng_);
    if (sort_method == 0) {
        std::shuffle(sorted_removed.begin(), sorted_removed.end(), rng_);
    } else if (sort_method == 1) {
        std::sort(sorted_removed.begin(), sorted_removed.end(), [&](int a, int b) {
            return params_.cli[a].demand > params_.cli[b].demand;
        });
    } else if (sort_method == 2) {
        std::sort(sorted_removed.begin(), sorted_removed.end(), [&](int a, int b) {
            double dist_a = params_.timeCost[0][a];
            double dist_b = params_.timeCost[0][b];
            return dist_a < dist_b;
        });
    } else {
        std::sort(sorted_removed.begin(), sorted_removed.end(), [&](int a, int b) {
            double dist_a = params_.timeCost[0][a];
            double dist_b = params_.timeCost[0][b];
            return dist_a > dist_b;
        });
    }

    // Calcular la carga actual de cada ruta para evitar cálculos repetitivos
    std::vector<double> route_loads(params_.nbVehicles, 0.0);
    for (int r = 0; r < params_.nbVehicles; ++r) {
        for (int c : solution.chromR[r]) {
            route_loads[r] += params_.cli[c].demand;
        }
    }

    // Insertar clientes con blinks
    for (int customer : sorted_removed) {
        struct InsertionPoint {
            int route;
            int pos;
            double cost;
        };
        std::vector<InsertionPoint> insertion_points;

        for (int r = 0; r < params_.nbVehicles; ++r) {
            auto& route = solution.chromR[r];
            double new_load = route_loads[r] + params_.cli[customer].demand;
            if (new_load > params_.vehicleCapacity) continue; // Saltar si excede la capacidad

            for (int pos = 0; pos <= static_cast<int>(route.size()); ++pos) {
                double cost = 0.0;
                if (route.empty() && pos == 0) {
                    cost = params_.timeCost[0][customer] + params_.timeCost[customer][0];
                } else {
                    int prev = (pos == 0) ? 0 : route[pos - 1];
                    int next = (pos == static_cast<int>(route.size())) ? 0 : route[pos];
                    cost = params_.timeCost[prev][customer] + params_.timeCost[customer][next];
                    if (pos > 0) cost -= params_.timeCost[prev][next];
                }
                insertion_points.push_back({r, pos, cost});
            }
        }

        if (insertion_points.empty()) {
            for (int r = 0; r < params_.nbVehicles; ++r) {
                if (solution.chromR[r].empty() && params_.cli[customer].demand <= params_.vehicleCapacity) {
                    solution.chromR[r].push_back(customer);
                    route_loads[r] += params_.cli[customer].demand;
                    break;
                }
            }
            continue;
        }

        std::sort(insertion_points.begin(), insertion_points.end(),
                  [](const auto& a, const auto& b) { return a.cost < b.cost; });

        // Seleccionar punto de inserción con probabilidad de "blink"
        std::vector<double> probabilities;
        double prob_sum = 0.0;
        for (int r = 1; r <= static_cast<int>(insertion_points.size()); ++r) {
            double p = (1.0 - params_.blink_probability) * std::pow(params_.blink_probability, r - 1);
            probabilities.push_back(p);
            prob_sum += p;
        }
        std::uniform_real_distribution<> dist_prob(0.0, prob_sum);
        double rand = dist_prob(rng_);
        int selected = 0;
        double cumsum = 0.0;
        for (int i = 0; i < static_cast<int>(probabilities.size()); ++i) {
            cumsum += probabilities[i];
            if (rand <= cumsum) {
                selected = i;
                break;
            }
        }

        int route = insertion_points[selected].route;
        int pos = insertion_points[selected].pos;
        solution.chromR[route].insert(solution.chromR[route].begin() + pos, customer);
        route_loads[route] += params_.cli[customer].demand;
    }

    // Actualizar chromT y reevaluar la solución
    solution.chromT.clear();
    for (const auto& route : solution.chromR) {
        solution.chromT.insert(solution.chromT.end(), route.begin(), route.end());
    }
    solution.evaluateCompleteCost(params_);
}