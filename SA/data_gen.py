import os
import random
import argparse

def generate_math_combinations_to_file(n, output_dir, split=False):
    """
    Generates all unique addition and subtraction combinations
    where numbers are within the range 0 to 50, and results are positive.
    Then writes them to the specified file.

    Args:
        output_dir (str): The directory to save the generated files.
        split (bool): Whether to split the dataset into train and test sets.
    """
    combinations = set()
    
    if n > 300:
        random.seed(42)
        while len(combinations) < 90500:
            x = random.randint(0, n)
            y = random.randint(0, n)
            p = random.choice(['+', '-'])
            if p == '+':
                result = x + y
            else:
                result = x - y
            if result >= 0:
                combination_str = f"{x}{p}{y}={result}"
                combinations.add(combination_str)
    else:
        for num1 in range(n + 1):
            for num2 in range(n + 1):
                # Addition
                sum_result = num1 + num2
                if 0 <= sum_result:
                    combination_str = f"{num1}+{num2}={sum_result}"
                    combinations.add(combination_str)

                # Subtraction
                diff_result = num1 - num2
                if 0 <= diff_result:
                    combination_str = f"{num1}-{num2}={diff_result}"
                    combinations.add(combination_str)

                # # Xor
                # xor_result = num1 ^ num2
                # combination_str = f"{num1}^{num2}={xor_result}"
                # combinations.add(combination_str)

    sorted_combinations = sorted(list(combinations), key=lambda x: (int(x.split('+')[0].split('-')[0].split('^')[0]), x))

    # Ensure the directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if split:
        indices = list(range(len(sorted_combinations)))
        random.seed(19)
        random.shuffle(indices)
        test_size = 2000
        test_data = [sorted_combinations[i] for i in indices[:test_size]]
        train_data = [sorted_combinations[i] for i in indices[test_size:]]
        # Write train combinations to the file
        train_output_filepath = os.path.join(output_dir, f"arith_lt{n}_train.txt")
        with open(train_output_filepath, 'w') as f:
            for combo in train_data:
                f.write(combo + '\n') # Add a newline character for each combination
        print(f"Generated train combinations for n={n}: {train_output_filepath}")
        # Write test combinations to the file
        test_output_filepath = os.path.join(output_dir, f"arith_lt{n}_test.txt")
        with open(test_output_filepath, 'w') as f:
            for combo in test_data:
                f.write(combo + '\n') # Add a newline character for each combination
        print(f"Generated test combinations for n={n}: {test_output_filepath}")
    else:
        output_filepath = os.path.join(output_dir, f"arith_lt{n}.txt")
        # Write combinations to the file
        with open(output_filepath, 'w') as f:
            for combo in sorted_combinations:
                f.write(combo + '\n') # Add a newline character for each combination
        print(f"Generated combinations for n={n}: {output_filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate math combinations for synthetic demo.")
    parser.add_argument("--n", type=int, default=50,
                        help="The upper limit for numbers in combinations (inclusive)")
    parser.add_argument("--output_dir", type=str, default='data/',
                        help="Path to save the generated combinations file")
    parser.add_argument("--split", action='store_true',
                        help="Whether to split the dataset into train and test sets")
    args = parser.parse_args()

    # Define the output file path
    n = args.n

    # Call the function to generate and write the combinations to the file
    generate_math_combinations_to_file(n, args.output_dir, args.split)
