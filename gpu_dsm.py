import argparse 
from core.main import FSM_LINEAR

def gpu_dsm():

	parser = argparse.ArgumentParser(description='Run pyDSM with specified arguments.')

	parser.add_argument('ID', type=int, nargs='?',default=0,
                    help='An integer for the simulation ID.')
	parser.add_argument('-d', metavar='device_num', type=int, nargs='?',default=0,
                    help='An integer for the device ID.')
	parser.add_argument('-c', type=str, nargs='?',default='MuNCH',choices=['otf','munch'],
                    help='Specify which correlator to use (otf or munch).')
	parser.add_argument('-o', metavar='path/to/output/',type=str, nargs='?',default='./DSM_results',
					help='Specify output directory.')
	parser.add_argument("-f", "--fit", action="store_true", help='A flag to turn on G(t) fit.')

	args = parser.parse_args()

	sim_ID = args.ID
	device_ID = args.d
	output_dir = args.o
	correlator = args.c 
	fit = args.fit 

	run_dsm = FSM_LINEAR(sim_ID,device_ID,output_dir,correlator,fit)
	run_dsm.run()

	return

if __name__ == "__main__":
	gpu_dsm()

