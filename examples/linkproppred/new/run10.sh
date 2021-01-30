python run.py --do_train --cuda --do_test \
  --dataset ogbl-tail1 --test_random_sample \
  --model F -n 128 -b 512 -d 20 -g 0.0 -a 1.0 -adv --double_relation_embedding \
  -lr 0.0001 --max_steps 50000 --cpu_num 2 --test_batch_size 32 --print_on_screen

#--add_random_fraction 0.5 --seed 125 \
