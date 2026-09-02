from prettyNEAT.neat import Neat
from prettyNEAT._speciate import Species


class Parent:
    fitness = 1.0
    rank = 0

    def createChild(self, p, innov, gen, mate=None):
        return Child(), innov


class Child:
    def express(self):
        return True


def test_single_species_offspring_allocation_controls_recombination_size():
    pop_size = 7
    neat = Neat(
        {
            "popSize": pop_size,
            "select_cullRatio": 0,
            "select_eliteRatio": 0,
            "select_tournSize": 1,
            "prob_crossover": 0,
        }
    )
    parent = Parent()
    species = Species(parent)

    assert species.nOffspring == 0

    allocated_species = neat.assignOffspring([species], [parent], neat.p)
    children, _ = neat.recombine(allocated_species[0], innov=[], gen=0)

    assert allocated_species[0].nOffspring == pop_size
    assert len(children) == pop_size
