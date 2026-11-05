from tests.test_verification import test_verification
from experiments.train_make_moons import task5_make_moons
from experiments.eager_vs_compiled import task6_compilation

if __name__ == "__main__":
    test_verification()
    task5_make_moons()
    task6_compilation()