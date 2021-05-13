import numpy as np
from sklearn.svm import SVR


class BGA():
    """
    Hybradized genetic algorithm.
    User Guide:
    >> test = GA(pop_shape=(10, 10), method=np.sum)
    >> solution, fitness = test.run()
    """

    def __init__(self, train, pop_shape, p_c=0.8, p_m=0.2, max_round=100, early_stop_rounds=None, verbose=None, maximum=False):
        if early_stop_rounds != None:
            assert (max_round > early_stop_rounds)
        self.pop_shape = pop_shape
        self.pop = np.zeros(pop_shape)
        self.fitness = np.zeros(pop_shape[0])
        self.p_c = p_c
        self.p_m = p_m
        self.max_round = max_round
        self.early_stop_rounds = early_stop_rounds
        self.verbose = verbose
        self.maximum = maximum

        self.train = train

    def evaluation(self, pop):
        fitness = []
        for i in range(len(pop)):
            individual = pop[i]
            cols = [index for index in range(16) if individual[index] <= 0.0]
            X_trainParsed = self.train.drop(self.train.columns[cols], axis=1)
            svr_rbf = SVR(kernel='rbf', gamma=individual[-1], C=individual[-3], epsilon=individual[-2])
            score = -1.0 * np.mean(
                cross_val_score(svr_rbf, X_trainParsed, y_train, scoring='neg_mean_squared_error', cv=10))
            fitness.append(score)
        return fitness

    def initialization(self):
        """
        Initalizing the population which shape is self.pop_shape(0-1 matrix).
        """
        self.pop = np.array(init_svr)
        self.fitness = self.evaluation(self.pop)

    def crossover(self, ind_0, ind_1):
        """
        Single point crossover.
        """
        assert (len(ind_0) == len(ind_1))
        point = np.random.randint(len(ind_0))
        new_0 = np.hstack((ind_0[:point], ind_1[point:]))
        new_1 = np.hstack((ind_1[:point], ind_0[point:]))
        assert (len(new_0) == len(ind_0))
        return new_0, new_1

    def mutation(self, indi):
        """
        Simple mutation.
        """
        point = np.random.randint(0, 18)
        if point < 16:
            indi[point] = +5
        elif point == 16:
            indi[point] = 100
        elif point == 17:
            indi[point] = 0.01
        elif point == 18:
            indi[point] = 0.01
        #        indi[point] = 1 - indi[point]
        return indi

    def rws(self, size, fitness):
        """
        Roulette Wheel Selection.
        Args:
            size: the size of individuals you want to select according to their fitness.
            fitness: the fitness of population you want to apply rws to.
        """
        if self.maximum:
            fitness_ = fitness
        else:
            fitness_ = fitness
        idx = np.random.choice(np.arange(len(fitness_)), size=size, replace=True,
                               p=fitness_ / np.sum(fitness_))
        return idx

    def run(self):
        """
        Run the genetic algorithm.
        """
        self.initialization()
        best_index = np.argsort(self.fitness)[0]
        global_best_fitness = self.fitness[best_index]
        global_best_ind = self.pop[best_index, :]
        count = 0

        for it in range(self.max_round):
            next_gene = []

            for n in range(int(self.pop_shape[0] / 2)):
                i, j = self.rws(2, self.fitness)
                # choosing 2 individuals with rws.
                indi_0, indi_1 = self.pop[i, :].copy(), self.pop[j, :].copy()
                if np.random.rand() < self.p_c:
                    indi_0, indi_1 = self.crossover(indi_0, indi_1)

                if np.random.rand() < self.p_m:
                    indi_0 = self.mutation(indi_0)
                    indi_1 = self.mutation(indi_1)

                next_gene.append(indi_0)
                next_gene.append(indi_1)

            self.pop = np.array(next_gene)
            self.fitness = self.evaluation(self.pop)

            if self.maximum:
                if np.max(self.fitness) > global_best_fitness:
                    best_index = np.argsort(self.fitness)[-1]
                    global_best_fitness = self.fitness[best_index]
                    global_best_ind = self.pop[best_index, :]
                    count = 0
                else:
                    count += 1
                worst_index = np.argsort(self.fitness)[-1]
                self.pop[worst_index, :] = global_best_ind
                self.fitness[worst_index] = global_best_fitness

            else:
                if np.min(self.fitness) < global_best_fitness:
                    best_index = np.argsort(self.fitness)[0]
                    global_best_fitness = self.fitness[best_index]
                    global_best_ind = self.pop[best_index, :]
                    count = 0
                else:
                    count += 1

                worst_index = np.argsort(self.fitness)[-1]
                self.pop[worst_index, :] = global_best_ind
                self.fitness[worst_index] = global_best_fitness

            print(it, '   ', global_best_fitness, global_best_ind)
            if self.verbose != None and 0 == (it % self.verbose):
                print('Gene {}:'.format(it))
                print('Global best fitness:', global_best_fitness)

            if self.early_stop_rounds != None and count > self.early_stop_rounds:
                print('Did not improved within {} rounds. Break.'.format(self.early_stop_rounds))
                break

        print('\n Solution: {} \n Fitness: {}'.format(global_best_ind, global_best_fitness))
        return global_best_ind, global_best_fitness