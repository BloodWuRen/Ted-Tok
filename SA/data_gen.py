import os
import random

def generate_math_combinations_to_file(n, output_filepath):
    """
    Generates all unique addition and subtraction combinations
    where numbers are within the range 0 to 50, and results are positive.
    Then writes them to the specified file.

    Args:
        output_filepath (str): The full path to the output file.
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
    output_dir = os.path.dirname(output_filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write combinations to the file
    with open(output_filepath, 'w') as f:
        for combo in sorted_combinations:
            f.write(combo + '\n') # Add a newline character for each combination

    print(f"Write the demo to file: {output_filepath}")

# Define the output file path
n = 9999

# file_path = f"/data/zhangzhi/streaming_demo/arith_lt{n}.txt"
file_path = f"/data/zhangzhi/streaming_demo/arith_lt{n}.txt"

# Call the function to generate and write the combinations to the file
generate_math_combinations_to_file(n, file_path)

