import numpy as np
import matplotlib.pyplot as plt

def get_wave_elevation(A, B, w, th, x, y, t, ramp_up_time):
	'''returns a single wave elevation using equation 89 from volume XII of LAMP manual, assuming deep water'''
	k = w**2 / 9.807 #g=9.807
	wave_elevation = A*np.cos(k*(x*np.cos(np.radians(B)) + y*np.sin(np.radians(B))) - w*t + np.radians(th))
	return wave_elevation

def get_wave_height_grid(x_center, y_center, t, x_length, y_length, d, sea_file, ramp_up_time):
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
	wave_grid = np.zeros((y_length, x_length))
	x0 = x_center - d*(x_length-1)/2
	y0 = y_center - d*(y_length-1)/2

	for i in range(y_length):
		y = y0 + i*d
		for j in range(x_length):
			x = x0 + j*d
			height = 0
			for n in range(seaway.shape[0]):
				w = seaway[n,0] #frequency
				th = seaway[n,1]#phase angle
				A = seaway[n,2] #amplitude
				B = seaway[n,3] #heading angle
				height += get_wave_elevation(A, B, w, th, x, y, t, ramp_up_time)
			if t<ramp_up_time:
				height *= t/ramp_up_time
			wave_grid[i,j] = height

	return wave_grid

###### TEST #####
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