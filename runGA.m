close all; 
clear all;
clc;
warning off
tic;
       
%% USAGE INSTRUCTIONS:
% To use this framework, simply open the Objective.m file, and modify the 
% trigger function to suit your needs.  This function should be the function 
% to be minimized.  
% 
% Examples are a simulation of a car doing a quarter mile, 
% which can be in a subfunction, and the Objective.trigger function simulates 
% the quarter mile, then calculates and returns the mean squared error between
% that simulation and an experimental result.  
% 
% Remember not to use any input
% or output file writting or the parallelization will fail.
% 
% Before using, type in terminal:
%     >> matlabpool
% to configure the Parallel Computing Toolbox (required).

%% Defining solution space and parameters
mins=[0.01,0.01,.1,0.01,0.01,0.01,0.1,0.01];
maxes=[0.95,0.99,13,0.2,0.999999,0.77,100,0.667];
rate=0.08;
kurt=10;
num_neighborhoods=4;
num_generations=2500;
num_members=8;
crossover_type=0;
%% Creating the Globe
% Parameter list:
%       num_gens, population size, mins, maxes 

%NOTE: population size should be divisible by number of populations, or
%else the merge of populations will fail.

gen_sample1=GAGlobe(@fitnessFunction,num_neighborhoods, num_generations, num_members,mins,maxes,kurt,rate,crossover_type);

%% Selection the selection type
% denoted by a string of 3 numbers, with 1 for on, 0 for off.  If more than
% one is selected, then the selections will be split among the selected
% methods.
%
%           selection_string=[elitism roulette tournament]
selection_string=[1 1 0];
for j=1:num_neighborhoods
    gen_sample1.setSelectionString(j,selection_string);
end

%% Initial Evolution
% evolves one epoch in each of the still segregated populations
gen_sample2=gen_sample1.evolveCommunities;

%% Combine populations
% takes the best 25% from each community and combines them into one
% population
gen_sample3=gen_sample2.mergeCommunities;
gen_sample3.setSelectionString(num_neighborhoods+1,selection_string);

%% Secondary Evolution
% evolves the combined population for one epoch.
gen_sample4=gen_sample3.evolveGlobe;



%% Printing
popfitness=gen_sample4.nations.popfitness;
[empty,ix]=sort(popfitness);
gene=gen_sample4.nations.pop{1,ix(1)}.mis.gene;
command=sprintf('SRM_serve_print.exe %20.10fd0 %20.10fd0 %20.10fd0 %20.10fd0 %20.10fd0 %20.10fd0 %20.10fd0 %20.10fd0',gene);
[~,result]=system(command);
plot_data=sscanf(result,'%f\t');

figure;
plot(plot_data);
ylabel('Thrust');
title('Neutral Thrust Result')
toc;