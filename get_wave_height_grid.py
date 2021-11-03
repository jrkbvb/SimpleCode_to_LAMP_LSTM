import numpy as np
import matplotlib.pyplot as plt

def get_wave_height_grid(x_center, y_center, t, x_length, y_length, dx, dy, seaway, ramp_up_time):
	'''Returns a 2-D numpy array (matrix) of wave heights centered at (y_center,x_center).
	Note that the order of (y,x) follows the numpy array of (rows, columns).
	So when viewing the matrix, y increases going down and x increases going right.

	t = time
	x_length = # of columns (must be odd)
	y_length = # of rows (must be odd)
	d = distance in meters between grid points (both directions)
	sea_file = the .sea file which has "Frequency 	Phase (deg)		Amplitude	Heading (deg)"

	Calculation performed using equation (89) from LAMP manual volume XII; assumes deep water
	'''
	if x_length%2==0 or y_length%2==0:
		print("ERROR: Must enter wave height grid dimensions that are odd")
		return

	#First read in the .sea file to get all the wave components
	x0 = x_center - dx*(x_length-1)/2
	y0 = y_center - dy*(y_length-1)/2
	x = np.linspace(x0, x0+x_length*dx, num=x_length, endpoint=False)
	y = np.linspace(y0, y0+y_length*dy, num=y_length, endpoint=False)
	i = np.arange(0,seaway.shape[0])
	xx, yy, ii = np.meshgrid(x, y, i, sparse=True)
	w = seaway[:,0] #frequency
	th = seaway[:,1]#phase angle
	A = seaway[:,2] #amplitude
	B = seaway[:,3] #heading angle
	wave_grid = A[ii]*np.cos((w[ii]**2 / 9.807)*(xx*np.cos(np.radians(B[ii])) + yy*np.sin(np.radians(B[ii]))) - w[ii]*t + np.radians(th[ii]))
	wave_grid = wave_grid.sum(axis=2)
	if t<ramp_up_time:
		wave_grid *= t/ramp_up_time
	return wave_grid

def generate_wave_grid_file(lamp_file, x_length, y_length, dx, dy, ramp_up_time):
	'''Saves a new text file with the name "lamp_file.wave_grid".
	This file will have one row for every time segment. Each row is a 
	flattened array of the wave grid. The lamp_file.mot is used to get the x
	and y coordinates of the center of gravity (grid center point).
	x_length = number of points to include in the x-direction.
	y_length = number of points to include in the y-direction.
	d = distance in meters between grid points'''
	xy_coord = np.loadtxt(lamp_file+".mot",skiprows=3)[:,1:3] #all the rows, 2nd and 3rd cols
	seaway = np.loadtxt(lamp_file+".sea", skiprows=6)
	num_time_steps = xy_coord.shape[0]
	all_wave_grids = np.zeros((num_time_steps, x_length*y_length))
	for t in range(num_time_steps):
		time = t/10
		wave_grid = get_wave_height_grid(xy_coord[t,0], xy_coord[t,1], time, x_length, y_length, dx, dy, seaway, ramp_up_time)
		all_wave_grids[t,:] = wave_grid.flatten()

	np.savetxt(lamp_file+".wave_grid", all_wave_grids, delimiter=' ')



###### TEST #####
lamp_file_base = "D:/LAMP/SJ_version/output_files/polar_set1_115H_164T/L2_115H_164T_"
x_length = 3
y_length = 3
dx = 39
dy = 5
ramp_up_time = 10
for speed in range(0,21,5):
	print("speed = ", speed)
	for angle in range(0,360,15):
		print("angle = ", angle)
		for seed in range(1,12):
			print("seed = ", seed)
			lamp_file = lamp_file_base + str(speed) + "kt_" + str(angle) + "deg_0" + str(seed)
			generate_wave_grid_file(lamp_file, x_length, y_length, dx, dy, ramp_up_time)

'''
x_center = 0
y_center = 0
x_length = 1
y_length = 1
d = 1
sea_file = ("C:/Users/ASUS/Documents/MIT/Thesis/python code/L2_10kt_ss8_r0001.sea")
seaway = np.loadtxt(sea_file, skiprows=6)
print("Loaded seaway file. There are ", seaway.shape[0], " wave components")
max_time = 1800
ramp_up_time = 10
my_wave_grid = np.zeros((max_time*10,1))
times = np.linspace(0,max_time,max_time*10,endpoint=False)
for t in times:
	x_center = 5.1444 * t -2.53688
	my_wave_grid[int(t*10)] = get_wave_height_grid(x_center, y_center, t, x_length, y_length, d, seaway, ramp_up_time)

correct_answers = np.loadtxt("C:/Users/ASUS/Documents/MIT/Thesis/python code/L2_10kt_ss8_r0001.wav", skiprows=3)[:,1:2]

errors = correct_answers[:max_time*10] - my_wave_grid
print("my_wave_grid:", my_wave_grid)
print("\n correct_answers:", correct_answers)
print("\n errors:",errors)
plt.plot(errors)

plt.figure()
plt.plot(correct_answers)
plt.plot(my_wave_grid)
plt.show()
'''