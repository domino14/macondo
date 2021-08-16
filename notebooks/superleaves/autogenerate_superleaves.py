import csv
import argparse
import subprocess
from shutil import copyfile
from datetime import date
import os.path
import pickle as pkl
from string import ascii_uppercase

# Constants

MAXIMUM_SUPERLEAVE_LENGTH = 6
TILE_DISTRIBUTION = {'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 2, 'I': 9, 'J': 1, 'K': 1, 'L': 4, 'M': 2, 'N': 6, 'O': 8, 'P': 2, 'Q': 1, 'R': 6, 'S': 4, 'T': 6, 'U': 4, 'V': 2, 'W': 2, 'X': 1, 'Y': 2, 'Z': 1, '?': 2}
TILES = [x for x in ascii_uppercase] + ['?']
ALPHABETICAL_KEY = '?ABCDEFGHIJKLMNOPQRSTUVWXYZ'
SORT_FUNC = lambda x: ALPHABETICAL_KEY.index(x)

LEAVES_STRUCTURE_FILENAME = 'data.idx'
OLV_STRUCTURE_FILENAME = 'data.olv'
ALL_LEAVES_FILENAME = 'all_leaves.p'
PATH_TO_DEFAULT_ENGLISH = '../../data/strategy/default_english/'
TILE_LIMIT = 1
CURRENT_DATE = date.today().strftime("%Y%m%d")

BINGO_COUNT_INDEX = 0
COUNT_INDEX = 1
EQUITY_INDEX = 2
POINTS_INDEX = 3
MEAN_SCORE_INDEX = 4
MEAN_EQUITY_INDEX = 5
BINGO_PCT_INDEX = 6
PCT_INDEX = 7
ADJUSTED_MEAN_SCORE_INDEX = 8
EV_INDEX = 9
SMOOTHED_EV_INDEX = 10

def increment_pointers(pointers, number_of_tiles):
    minimum_pointer_to_move = len(pointers) - 1
    endpoint_for_pointer = number_of_tiles - 1
    while pointers[minimum_pointer_to_move] == endpoint_for_pointer:
        minimum_pointer_to_move -= 1
        endpoint_for_pointer -= 1
        if minimum_pointer_to_move < 0:
            return None

    old_minimum_pointer_to_move_value = pointers[minimum_pointer_to_move]
    for i in range (minimum_pointer_to_move, len(pointers)):
        # Move pointer to break point instead
        old_minimum_pointer_to_move_value += 1
        pointers[i] = old_minimum_pointer_to_move_value

    return pointers

def generate_leaves_for_length(tile_bag, leave_length, allow_duplicates):
    pointers = list(range(0, leave_length))
    tile_bag_size = len(tile_bag)
    leaves_hash = {}
    leaves = []
    while True:
        leave = ""
        for pointer in pointers:
            leave += tile_bag[pointer]
        if allow_duplicates:
            leaves.append(leave)
        else:
            leaves_hash[leave] = True
        pointers = increment_pointers(pointers, tile_bag_size)
        if pointers is None:
            break
    
    if not allow_duplicates:
        leaves = list(leaves_hash.keys())

    return leaves

def generate_leaves_to_from_length(tile_bag, start_length, end_length, allow_duplicates):
    leaves = []
    for i in range(start_length,end_length+1):
        leaves.extend(generate_leaves_for_length(tile_bag, i, allow_duplicates))
    return leaves

def smoothable(key, dp):
    return (len(key) == 5 and dp[COUNT_INDEX] < 828) or (len(key) == 6 and dp[COUNT_INDEX] < 234)

def get_neighboring_leaves(leave):
    neighbors = []
    for i in range(len(leave)):
        if leave[i] == '?':
            continue    
        for letter in ALPHABETICAL_KEY[1:]:
            neighbors.append(''.join(sorted(leave[:i] + letter + leave[i+1:], key=SORT_FUNC)))
    return neighbors

def calculate_smoothed_superleave(ev_data, root_leave):
    neighbor_leaves = get_neighboring_leaves(root_leave)

    neighboring_equity = 0
    neighboring_count = 0
    
    for neighbor_leave in neighbor_leaves:
        if neighbor_leave in ev_data:
            neighboring_equity += ev_data[neighbor_leave][EQUITY_INDEX]
            neighboring_count += ev_data[neighbor_leave][COUNT_INDEX]
    
    if neighboring_count == 0:
        print ("No neighbors found for: {}".format(root_leave))
        return None

    return neighboring_equity/neighboring_count

def calculate_superleaves(move_logfile, leaves_csv_file):
    leaves = {}

    if not os.path.isfile(ALL_LEAVES_FILENAME):
        for i in range(1,MAXIMUM_SUPERLEAVE_LENGTH+1):
            truncated_tile_distribution = {}
            for tile in TILE_DISTRIBUTION:
                truncated_tile_distribution[tile] = min(i, TILE_DISTRIBUTION[tile])
            truncated_tile_bag = []
            for tile in truncated_tile_distribution:
                truncated_tile_bag.extend([tile]*truncated_tile_distribution[tile])
            truncated_tile_bag = sorted(truncated_tile_bag, key=SORT_FUNC)
            leaves[i] = generate_leaves_for_length(truncated_tile_bag, i, False)

        pkl.dump(leaves,open(ALL_LEAVES_FILENAME,'wb'))
    else:
        leaves = pkl.load(open(ALL_LEAVES_FILENAME,'rb'))

    all_leaves = []

    for i in leaves:
        all_leaves.extend(leaves[i])

    # Expected Value
    ev_data = {}
    for x in all_leaves:
        ev_data[x] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    row_count = 0
    total_equity = 0
    total_points = 0

    with open(move_logfile,'r') as f:
        moveReader = csv.reader(f)
        next(moveReader)
        
        for i,row in enumerate(moveReader):
            tiles_remaining = row[10]
            if int(tiles_remaining) >= TILE_LIMIT:
                row_rack = sorted(row[3], key=SORT_FUNC)
                row_score = int(row[5])
                row_tiles_played = row[7]
                row_equity = float(row[9])

                total_equity += row_equity
                total_points += row_score
                row_count += 1
                subleaves = generate_leaves_to_from_length(row_rack, 1, MAXIMUM_SUPERLEAVE_LENGTH, False)
                for subleave in subleaves:
                    if row_tiles_played == 7:
                        ev_data[subleave][BINGO_COUNT_INDEX]  += 1
                    ev_data[subleave][COUNT_INDEX] += 1
                    ev_data[subleave][EQUITY_INDEX] += row_equity
                    ev_data[subleave][POINTS_INDEX] += row_score

    mean_score = total_points/row_count
    mean_equity = total_equity/row_count

    for key in ev_data:
        dp = ev_data[key]
        if dp[COUNT_INDEX] > 0:
            dp[MEAN_SCORE_INDEX] = dp[POINTS_INDEX] / dp[COUNT_INDEX]
            dp[MEAN_EQUITY_INDEX] = dp[EQUITY_INDEX] / dp[COUNT_INDEX]
            dp[BINGO_PCT_INDEX] = 100 * dp[BINGO_COUNT_INDEX] / dp[COUNT_INDEX]
            dp[PCT_INDEX] = 100 * dp[COUNT_INDEX] / len(ev_data)
            dp[ADJUSTED_MEAN_SCORE_INDEX] = dp[MEAN_SCORE_INDEX] - mean_score
            dp[EV_INDEX] = dp[MEAN_EQUITY_INDEX] - mean_equity

    # Fill Null Values

    for leave in all_leaves:
        if ev_data[leave][COUNT_INDEX] == 0:
            leave_minus_one = len(leave)-1
            subleaves = generate_leaves_for_length(leave, leave_minus_one, True)
            sub_evs = [ev_data[subleave][EV_INDEX] for subleave in subleaves]
            signs = sum([x/abs(x) for x in sub_evs])
            if signs==0:
                ev_data[leave][EV_INDEX] = sum(sub_evs)/len(sub_evs)
            if signs>0:
                ev_data[leave][EV_INDEX] = max(sub_evs)
            if signs<0:
                ev_data[leave][EV_INDEX] = min(sub_evs)

    # Smoothing
    for key in ev_data:
        dp = ev_data[key]
        if smoothable(key, dp):
            smoothed_leave = calculate_smoothed_superleave(ev_data, key)
            if smoothed_leave is None:
                smoothed_leave = dp[EV_INDEX]
            else:
                smoothed_leave -= mean_equity
        else:
            smoothed_leave = dp[EV_INDEX]
        
        dp[SMOOTHED_EV_INDEX] = smoothed_leave

    with open(leaves_csv_file, 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key in ev_data:
            writer.writerow([key, ev_data[key][SMOOTHED_EV_INDEX]])

def main():
    parser = argparse.ArgumentParser(description='Generate superleaves with a single command. Must be executed from the superleaves directory.')
    parser.add_argument('-i', '--iterations', type=int, default=100, help='number of iterations to run')
    parser.add_argument('-g', '--games', type=int, default=200000, help='number of autoplay games per iteration')
    parser.add_argument('-l', '--lexicon', type=str, default='CSW19', help='lexicon used to generate superleaves')
    parser.add_argument('-o', '--output', type=str, default='autogenerate_{}'.format(CURRENT_DATE), help='prefix for outputs')
    args = parser.parse_args()

    iterations = args.iterations
    games = args.games
    lexicon = args.lexicon
    output = args.output

    for x in range(iterations):
        autoplay_logfile = output + '_autoplay_{}.csv'.format(x + 1)
        leaves_csv_file = output + '_leaves_{}.csv'.format(x + 1)
        leaves_idx_file = output + '_leaves_{}.idx'.format(x + 1)

        # Autoplay
        autoplay_cmd = "../../bin/shell autoplay -numgames {} -lexicon {} -logfile {} -block true".format(games, lexicon, autoplay_logfile)
        return_code = subprocess.call(autoplay_cmd, shell=True)  
        if return_code != 0:
            print ("Autoplay exited with return code {}".format(return_code))

        # Calculate superleaves
        calculate_superleaves(autoplay_logfile, leaves_csv_file)

        # Create superleave structure
        make_leaves_structure_cmd = "../../bin/make_leaves_structure -filename {}".format(leaves_csv_file)
        return_code = subprocess.call(make_leaves_structure_cmd, shell=True)  
        if return_code != 0:
            print ("Make leaves structure exited with return code {}".format(return_code))

        # Copy files to the right places
        copyfile(LEAVES_STRUCTURE_FILENAME, leaves_idx_file)
        copyfile(LEAVES_STRUCTURE_FILENAME, PATH_TO_DEFAULT_ENGLISH + 'leaves.idx')
        copyfile(OLV_STRUCTURE_FILENAME, PATH_TO_DEFAULT_ENGLISH + 'leaves.olv')


if __name__ == "__main__":
    main()
