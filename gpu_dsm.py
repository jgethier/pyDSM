import sys
from core.main import FSM_LINEAR

def gpu_dsm(narg, argv):

	run_flag = True
	k = 1
	device_ID=0
	sim_ID = 0
	while k < narg:
		if k == 1:
			sim_ID = int(sys.argv[k])

		if str(sys.argv[k])=="-d" and k+1 < narg:
			device_ID = int(sys.argv[k+1])
			k+=1

		k+=1

	run_dsm = FSM_LINEAR(sim_ID,device_ID)
	run_dsm.main()

	return

if __name__ == "__main__":
	gpu_dsm(len(sys.argv), sys.argv)

