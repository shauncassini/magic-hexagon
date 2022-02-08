''' 
Author: Shaun Cassini
Date: 8.2.2022
'''

import numpy as np
import time

class MagicHexSolver():
    
    global ROW_MAP, M

    M = 38
    ROW_MAP = np.array([
        [1, 2, 3, 0, 0],
        [4, 5, 6, 7, 0],
        [8, 9, 10, 11, 12],
        [13, 14, 15, 16, 0],
        [17, 18, 19, 0, 0],

        [1, 4, 8, 0, 0],
        [2, 5, 9, 13, 0],
        [3, 6, 10, 14, 17],
        [7, 11, 15, 18, 0],
        [12, 16, 19, 0, 0],

        [3, 7, 12, 0, 0],
        [2, 6, 11, 16, 0],
        [1, 5, 10, 15, 19],
        [4, 9, 14, 18, 0],
        [8, 13, 17, 0, 0]
    ])


    def __init__(self,
        p_size: int,
        p_cand: float
    ):
        self.p_size = p_size
        self.n_candidates = self.get_n_candidates(p_size, p_cand)

        self.r = np.arange(self.p_size) # modular arithmetic
        self.p_indices = np.tile(np.arange(1,20), self.p_size)

    
    def get_population(self):

        population = np.array([
            np.random.permutation(19) + 1 for _ in range(self.p_size)
        ])
        zero_padding = np.zeros(self.p_size, dtype=np.uint8)[:,None]
        population = np.hstack((zero_padding, population))

        return population


    def get_n_candidates(self, p_size, p_cand):
        cs = np.array([c for c in range(1, p_size+1) if p_size % c == 0]) # factors of p_size
        return cs[np.abs(cs - int(p_size * p_cand)).argmin()] # choose closest factor as n_candidates


    def get_cost(self, population):

        row_sums = np.sum(population[:, ROW_MAP], axis=2)
        row_diff = np.abs(M - row_sums)

        return np.sum(row_diff, axis=1)


    def select_best(self, population):
        # how many times to clone each candidate
        cloning_factor = self.p_size // self.n_candidates
        costs = self.get_cost(population)

        fittest_indices = costs.argsort()[:self.n_candidates]
        fittest = np.repeat(population[fittest_indices], cloning_factor, axis=0)

        return fittest, costs.min() 

    
    def mutate(self, candidates):
        # Don't swap the 0th index!
        swaps = np.random.randint(1, 20, size=(2, self.p_size)) 

        indices = range(self.p_size) # needed to index each gene - different to ':'
        temp = candidates[indices, swaps[0]]
        candidates[indices, swaps[0]] = candidates[indices, swaps[1]]
        candidates[indices, swaps[1]] = temp

        return candidates

    def display_board_mele(self, board):
        # convert numbers to strings (with spaces in front of numbers < 10)
        board_str = ['{}'.format(t) if t > 9 else '0{}'.format(t) for t in board[1:]]
        template = '''
                    /)        
                   (/         
            ##**##//######*   
          ####*############## 
         ##### {} {} {} ######
        #*### {} {} {} {} ####
        ### {} {} {} {} {} ##*
        ##### {} {} {} {} ####
         ##*## {} {} {} ###*# 
          #**###############  
            #####*###*###      
        '''
        return print(template.format(*board_str))


    def display_board(self, board, i, time):
        # convert numbers to strings (with 0s in front of numbers < 10)
        board_str = ['{}'.format(t) if t > 9 else '0{}'.format(t) for t in board[1:]]
        template = '''
            Solution found!

               {} {} {} 
              {} {} {} {} 
            {} {} {} {} {} 
              {} {} {} {}
               {} {} {}

           {:.2e} iterations
           {:.2e} seconds
        '''
        print(template.format(*board_str, i, time))


    def search(self):

        population = self.get_population()
        candidates, cost = self.select_best(population)
        
        # Keep track of the time and no. of iterations
        time_start = time.time() 
        i = 0 
        while cost > 0: 
            i += 1
            population = self.mutate(candidates)
            candidates, cost = self.select_best(population) # 'cost' is the evaluation

        time_elapsed = time.time() - time_start

        costs = self.get_cost(candidates)
        best_candidate = candidates[costs.argsort()[0]]
        self.display_board(best_candidate, i, time_elapsed)