################################ Execute all Scripts ################################
import input

start = input.start_end_file(file=__file__.split("/")[-1])
################################ Start

# initialize json file
import json

json.dump([], open(input.json_file, "w"))

# run all scripts
import s1_data

# import s21_snorkel
import s22_da

# import s23_tl
import s24_comb
import s3_prep
import s4_train
import s5_eval
import sfinal_test

################################ End
input.start_end_file("end", start, file=__file__.split("/")[-1])
