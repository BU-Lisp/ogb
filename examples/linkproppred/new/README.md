# Read new graphs and convert to OGB datasets


python convertkg.py <new-dataset> -m <mode> -f <file> <options ...>

Creates a directory dataset_<name> based on the data from <file> .

To use the dataset,
python run.py --dataset <name> ...

Modes:
-m read_triples   -- <file> is text lines each with head,relation,tail 
-m read_two_files -- <file> is a directory containing two files,
   		     edge.csv and edge_reltype.csv .  This is the form of
		     the OGB raw data.
-m random_gnp	  -- generate a random graph using parameters
   		     --n_vertices N --edge_probability p --n_relations R

## Modifications to apply to the graph

--subsample <fraction>
--test_upto N     -- testset will be sampled only from links 0 to N-1.
--shuffle_edge_types <fraction> 
--collapse_edge_types N -- apply (modulo N) to the relation types

--map_node_file <dictfile>
--map_relation_file <dictfile>
		    These files consist of lines index,name
		    converts the string "name" in a triple to the number "index".

## Testing and printing functions

-t  	   read in a dataset and print some or all of it
	   --print_relations prints a list of all test edge types in order
	   --select_head h print all triples with head index h and/or tail t.
	   --select_tail t