import sys
# sys.path.insert(0, '..')
import resiliencyTool as rt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--simulation", type=str, default='',help="Simulation name")
parser.add_argument("-p", "--prerun", type=bool, default=False, help="launch pre run")
parser.add_argument("-n", "--number_simulations", type=int, default=1,help="number of Monte Carlo simulations")

args = parser.parse_args()
if args.simulation == '':
	print ('Error: no input file name')
	sys.exit()

print(f'Pre-run: {args.prerun}')
print(f'Number of iterations: {args.number_simulations}')

network = rt.network.Network(args.simulation);
simulation = rt.simulation.Sim(args.simulation);
n = args.number_simulations
if args.prerun:
	simulation.pre_run(network, n)
simulation.run(network, iterationSet = range(n), run_type = 'pm_ac_opf', delta = 1e-16, saveOutput = True)

breakpoint()