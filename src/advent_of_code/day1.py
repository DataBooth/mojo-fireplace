INITIAL_POSITION = 50
N_POSITION = 100
DEBUG = False

INPUT_FILE = "day1_input.txt"
INPUT_TEST_FILE = "day1_input_test.txt"


def load_rotations(filename):
    rotations = []
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith("L"):
                rotations.append(-int(line[1:]))
            else:
                rotations.append(int(line[1:]))
    return rotations


def update_position(current_position, rotation, n_position):
    # Modular arithmetic using % handles wrapping for both positive and negative values
    return (current_position + rotation) % n_position


def main():
    input_file = INPUT_TEST_FILE if DEBUG else INPUT_FILE
    rotations = load_rotations(input_file)

    print(f"Info - number of rotations: {len(rotations)}")
    print(f"Info - max rotations: {max(rotations)}; min rotations: {min(rotations)}")

    current_position = INITIAL_POSITION
    count_zeros = 0

    print(f"The dial starts by pointing at {INITIAL_POSITION}.")

    for rotation in rotations:
        current_position = update_position(current_position, rotation, N_POSITION)
        if current_position == 0:
            count_zeros += 1

        if DEBUG:
            direction = "L" if rotation < 0 else "R"
            print(
                f"The dial is rotated {direction}{abs(rotation)} to point at {current_position}."
            )

    print(f"Password: {count_zeros}")


if __name__ == "__main__":
    main()
