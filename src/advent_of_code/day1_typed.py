"""Day 1 - Rotation/Dial Problem"""

from typing import List

INITIAL_POSITION: int = 50
N_POSITION: int = 100

INPUT_FILE: str = "day1_input.txt"
INPUT_TEST_FILE: str = "day1_input_test.txt"


def load_rotations(filename: str) -> List[int]:
    """Load rotations from file. L prefix = negative, R prefix = positive."""
    rotations: List[int] = []
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


def update_position(
    current_position: int, rotation: int, n_position: int
) -> int:
    """Update position with modular arithmetic to handle wrapping."""
    return (current_position + rotation) % n_position


def main() -> None:
    """Run the dial rotation calculation and print summary information."""
    debug: bool = True  # Toggle this flag to enable verbose debugging output and use the test input file.

    input_file: str = INPUT_TEST_FILE if debug else INPUT_FILE
    rotations: List[int] = load_rotations(input_file)

    if debug:
        print(
            f"\n**Debug mode**: Using input data ({input_file}) - Password (count of zeros) should be 3."
        )
        print()

    print(f"Info - number of rotations: {len(rotations)}")

    # Calculate min/max rotations
    max_rotation: int = rotations[0]
    min_rotation: int = rotations[0]
    for rotation in rotations:
        if rotation > max_rotation:
            max_rotation = rotation
        if rotation < min_rotation:
            min_rotation = rotation

    print(
        f"Info - max rotations: {max_rotation}; min rotations: {min_rotation}"
    )

    current_position: int = INITIAL_POSITION
    count_zeros: int = 0

    print(f"The dial starts by pointing at {INITIAL_POSITION}")

    for rotation in rotations:
        current_position = update_position(current_position, rotation, N_POSITION)
        if current_position == 0:
            count_zeros += 1

        if debug:
            direction: str = "L" if rotation < 0 else "R"
            print(
                f"The dial is rotated {direction}{abs(rotation)} to point at {current_position}"
            )

    print(f"Password: {count_zeros}")


if __name__ == "__main__":
    main()
