import sys
from core.main import FSM_LINEAR

def gpu_dsm(narg, argv):

	k = 1
	device_ID=0
	sim_ID = 0
	correlator = 'default'
	while k < narg:
		if k == 1:
			sim_ID = int(argv[k])

		if str(argv[k])=="-d" and k+1 < narg:
			device_ID = int(argv[k+1])
			k+=1

		if str(argv[k])=="-c" and k+1 < narg:
			correlator = str(argv[k+1])
			k+=1

		k+=1

	run_dsm = FSM_LINEAR(sim_ID,device_ID,correlator)
	run_dsm.run()

	return

if __name__ == "__main__":
	gpu_dsm(len(sys.argv), sys.argv)

