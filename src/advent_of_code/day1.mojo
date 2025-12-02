"""
Day 1 - Rotation/Dial Problem
"""

alias INITIAL_POSITION: Int = 50
alias N_POSITION: Int = 100

alias INPUT_FILE: String = "day1_input.txt"
alias INPUT_TEST_FILE: String = "day1_input_test.txt"


fn load_rotations(filename: String) raises -> List[Int]:
    """Load rotations from file. L prefix = negative, R prefix = positive."""
    var rotations = List[Int]()

    with open(filename, "r") as file:
        var lines = file.read().split("\n")
        for line in lines:
            var line_stripped = line.strip()
            if len(line_stripped) == 0:
                continue

            if line_stripped.startswith("L"):
                # L rotation: negative value
                rotations.append(-Int(line_stripped[1:]))
            else:
                # R rotation: positive value (strip 'R' prefix)
                rotations.append(Int(line_stripped[1:]))

    return rotations.copy()


fn update_position(
    current_position: Int, rotation: Int, n_position: Int
) -> Int:
    """Update position with modular arithmetic to handle wrapping."""
    return (current_position + rotation) % n_position


fn main() raises:
    var debug = False  # Toggle this flag to enable verbose debugging output and use the test input file.

    var input_file = INPUT_TEST_FILE if debug else INPUT_FILE
    var rotations = load_rotations(input_file)

    print("Info - number of rotations:", len(rotations))

    # Calculate min/max rotations
    var max_rotation = rotations[0]
    var min_rotation = rotations[0]
    for rotation in rotations:
        if rotation > max_rotation:
            max_rotation = rotation
        if rotation < min_rotation:
            min_rotation = rotation

    print(
        "Info - max rotations:", max_rotation, "; min rotations:", min_rotation
    )

    var current_position = INITIAL_POSITION
    var count_zeros = 0

    print("The dial starts by pointing at", INITIAL_POSITION, ".")

    for rotation in rotations:
        current_position = update_position(
            current_position, rotation, N_POSITION
        )

        if current_position == 0:
            count_zeros += 1

        if debug:
            var direction = "L" if rotation < 0 else "R"
            var abs_rotation = -rotation if rotation < 0 else rotation
            print(
                "The dial is rotated",
                direction + String(abs_rotation),
                "to point at",
                current_position,
                ".",
            )

    print("Password:", count_zeros)
