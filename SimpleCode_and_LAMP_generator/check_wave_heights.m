clc; clear all; close all;

SC_file = "SimpleCode_files\SC_h11.5_p16.4_a90_hh0_pp0_aa0_s0-0000001.wav";
LAMP_file = "LAMP_files\LAMP_h11.5_p16.4_a90_hh0_pp0_aa0_s0-0000001.wav";

time = importdata(SC_file).data(:,1);
SC_data = importdata(SC_file).data(:,2);
LAMP_data = importdata(LAMP_file).data(:,2);


plot(time, SC_data)
hold on
plot(time, LAMP_data)

figure(2)
plot(time,(SC_data-LAMP_data))