from sys import argv
from qltsp import qltsp


def main(args):
	# to run the software three parameters are required:
	#
	# (i) the first specifies the real machine where to execute the algorithm
	#     choose the solver from the available ones in the client_config.conf file
	#
	# (ii) the second specifies the kind of execution that we want to test
	#      choose the mode by looking in the qltsp interface
	#
	# (iii) the third specifies the name of the experiment to be executed
	#       choose the name depending on the one present in the expriments.json file
    
    qltsp(solver=str(args[1]), mode=int(args[2]), exp_name=str(args[3]))


if __name__ == '__main__':
    main(argv)
