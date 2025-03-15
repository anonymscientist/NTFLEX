import os
import argparse
from sklearn.model_selection import train_test_split

class Preprocess:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.all_data = []
        self.all_files = os.listdir(input_path)
        if len(self.all_files) == 0:
            raise FileNotFoundError(f"No files found in {input_path}")
        else:
            for file in self.all_files:
                content = self.read_file(os.path.join(input_path, file))
                self.all_data.extend(content)
                print(f"Read {len(content)} quintuples from {file}")
        print(f"-> Read {len(self.all_data)} quintuples in total")

    def read_file(self, file_path):
        try:
            with open(file_path, "r") as file:
                return file.readlines()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {file_path}")
        
    def write_file(self, file_path, data):
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        # make sure to overwrite the file if it already exists
        with open(file_path, "w") as output:
            for line in data:
                output.write("\t".join([str(x) for x in line]) + "\n")

    def quintuple_to_quadruple(self, lines: list) -> list:
        '''
        This function reads a file with quintuples in format (s, p, o, since, until) and converts them to quadruples
        The function returns a list of quadruples in format (s, p, o, year) where year could be any possible year between since and until (including both)
        Note: duplicates will be removed!
        '''
        quadruples = set()
        for line in lines:
            s, p, o, _, since, _, until = line.strip().split('\t')
            since = int(since) if since != "None" else None
            until = int(until) if until != "None" else None
            try:
                # (s, p, o, since, until)
                if since and until:
                    while since <= until:
                        quadruples.add((s, p, o, since))
                        since += 1
                # (s, p, o, None, until)
                elif until:
                    quadruples.add((s, p, o, until))
                # (s, p, o, since, None)
                elif since:
                    quadruples.add((s, p, o, since))
                # (s, p, o, None, None)
                else:
                    pass
            except IndexError as e:
                print(f"IndexError: {e}, line: {line}")
        return quadruples

    def cleaning(self, data):
        cleaned_data = []
        for line in data:
            s, p, o, year = line
            if o.startswith("Q") or o.startswith("+") or o.startswith("-"):
                cleaned_data.append(line)
            else:
                # print(f"Removed line: {line}")
                pass
        return cleaned_data
    
    def split_data(self, train_size, valid_size, test_size, seed):
        data_list = list(self.all_data)
        train_data, test_data = train_test_split(data_list, test_size=test_size + valid_size, random_state=seed)
        valid_data, test_data = train_test_split(test_data, test_size=test_size / (test_size + valid_size), random_state=seed)
        print(f"Split data into {len(train_data)} training, {len(valid_data)} validation, and {len(test_data)} test quadruples")
        return train_data, valid_data, test_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Preprocess data for NTFLEX')
    parser.add_argument('--input_path', type=str, default='./data/raw', help='Path to the raw data')
    parser.add_argument('--output_path', type=str, default='./data/WIKI', help='Path to the preprocessed data')
    parser.add_argument('--train_file', type=str, default='train', help='Name of the training data file')
    parser.add_argument('--valid_file', type=str, default='valid', help='Name of the validation data file')
    parser.add_argument('--test_file', type=str, default='test', help='Name of the test data file')
    parser.add_argument('--train_size', type=float, default=0.8, help='Size of the training data')
    parser.add_argument('--valid_size', type=float, default=0.1, help='Size of the validation data')
    parser.add_argument('--test_size', type=float, default=0.1, help='Size of the test data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data splitting')
    args = parser.parse_args()
    
    preprocessor = Preprocess(input_path=args.input_path, output_path=args.output_path)
    preprocessor.all_data = preprocessor.quintuple_to_quadruple(preprocessor.all_data)
    print(f"Converted quintuples to {len(preprocessor.all_data)} quadruples")

    preprocessor.all_data = preprocessor.cleaning(preprocessor.all_data)
    print(f"The cleaned data contains {len(preprocessor.all_data)} quadruples")

    train_data, valid_data, test_data = preprocessor.split_data(train_size=args.train_size, valid_size=args.valid_size, test_size=args.test_size, seed=args.seed)
    preprocessor.write_file(os.path.join(args.output_path, args.train_file), train_data)
    preprocessor.write_file(os.path.join(args.output_path, args.valid_file), valid_data)
    preprocessor.write_file(os.path.join(args.output_path, args.test_file), test_data)
    print(f"Saved training, validation, and test data in {args.output_path}")
