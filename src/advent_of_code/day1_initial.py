INITIAL_POSITION = 50
N_POSITION = 100
DEBUG = True

INPUT_FILE = "day1_input.txt"
INPUT_TEST_FILE = "day1_input_test.txt"

rotations = []

if DEBUG:
    input_file = INPUT_TEST_FILE
else:
    input_file = INPUT_FILE

with open(input_file, "r") as file:
    for line in file:
        line = line.strip()  # Remove any trailing newline or spaces
        if line.startswith("L"):
            rotations.append(-int(line[1:]))
        else:
            rotations.append(int(line[1:]))

print(f"Info - number of rotations: {len(rotations)}")
print(f"Info - max rotations: {max(rotations)}; min rotations: {min(rotations)}")

current_position = INITIAL_POSITION
count_zeros = 0
count_all_zeros = 0

print(f"The dial starts by pointing at {INITIAL_POSITION}.")

for rotation in rotations:
    n_revolution = abs(rotation // N_POSITION)
    current_position = current_position + rotation % N_POSITION
    if current_position >= N_POSITION:
        current_position = current_position - N_POSITION
    if current_position < 0:
        current_position = N_POSITION + current_position
    if current_position == 0:
        count_zeros += 1
    count_all_zeros = count_all_zeros + n_revolution
    if True:
        if rotation < 0:
            print(
                f"The dial is rotated L{abs(rotation)} to point at {current_position} - with {n_revolution} revolutions."
            )
        else:
            print(
                f"The dial is rotated R{rotation} to point at {current_position} - with {n_revolution} revolutions."
            )

count_all_zeros = count_all_zeros + count_zeros

print(f"Password [old]: {count_zeros} - [new]: {count_all_zeros}")
