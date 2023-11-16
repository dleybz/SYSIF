from src.prompt.machine_prompt import EvoMachinePrompt, init_char_crossover, init_char_mutate, init_lama_fitness
from src.data.lama_dataset import LAMAset

#load LAMA
lamaset = LAMAset()

# evolution params
mutate_function = init_char_mutate(p_rand=0.01, p_suppr=0.01, p_dupp=0.01)
crossover_function = init_char_crossover()
fitness_function = init_lama_fitness(lamaset)

# start evolution
evoprompt = EvoMachinePrompt(mutate_function, crossover_function, fitness_function)
evoprompt.evolution()