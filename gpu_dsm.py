import argparse 
from core.main import FSM_LINEAR

def gpu_dsm():

	parser = argparse.ArgumentParser(description='Run pyDSM with specified arguments.')

	parser.add_argument('ID', type=int, nargs='?',default=0,
                    help='An integer for the simulation ID.')
	parser.add_argument('-d', metavar='device_num', type=int, nargs='?',default=0,
                    help='An integer for the device ID.')
	parser.add_argument('-c', type=str, nargs='?',default='munch',choices=['rsvl','munch'],
                    help='Specify which correlator to use (otf or munch).')
	parser.add_argument('-o', metavar='path/to/output/',type=str, nargs='?',default='./DSM_results',
					help='Specify output directory.')
	parser.add_argument("--rawdata", action="store_true", 
					help='A flag to save raw results to file (storage files may become large).')
	parser.add_argument("--fit", action="store_true", 
					help='A flag to turn on G(t) fit.')
	parser.add_argument('--distr',action="store_true",
					help='Save initial and final distributions for Q, Lpp, and Z.')
	parser.add_argument("-l","--load",metavar="/path/to/loadfile",type=str,nargs='?',
					help='Load in checkpoint file.')
	parser.add_argument('-s','--save',metavar='filename',type=str,default='checkpoint.dat',
					help='Save simulation checkpoint to file.')

	args = parser.parse_args()

	if args.d == None:
		args.d = 0
	if args.c == None:
		args.c = 'munch'
	if args.o == None:
		args.o = './DSM_results'
	if args.load == None:
		args.load = './'

	run_dsm = FSM_LINEAR(args.ID,args.d,args.o,args.c,args.rawdata,args.fit,args.distr,args.load,args.save)
	run_dsm.run()

	return

if __name__ == "__main__":
	gpu_dsm()

