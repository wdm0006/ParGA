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

%% Initialization 
% Before using, type in terminal:
%     >> matlabpool
% to configure the Parallel Computing Toolbox (required).
% 
% When the process is completed, type:
%     >> matlabpool close
% to complete the parallel job.

%% Objects
% The framework is made up of 4 main components.  This file is the run
% script, it is where the optimization is characterized and started.  
% 
% The Globe.m object contains the various neighborhoods ('nations'), and
% handles the evolution and merging of those.
%
% The Population.m file is a neighborhood object, it contains the various
% members, and handles breeding and mutation.
%
% The Member.m function is the member, it contains the fitness function and
% any dependencies to calculate that.  It also contains the gene itself for
% the memeber and the min and max bounds for gene parameters.  This is
% where you will enter in the fitness function to be minimized.

%% Defining solution space and parameters
mins=rand(1,10);
maxes=rand(1,10)+ones(1,10);
rate=0.15;
kurt=10;

%% Creating the Globe
% Parameter list:
%       num_gens, population size, mins, maxes 

%NOTE: population size should be divisible by number of populations, or
%else the merge of populations will fail.

gen_sample1=Globe(4,1000,64,mins,maxes,rate,kurt);

%% Initial Evolution
% evolves one epoch in each of the still segregated populations
gen_sample2=gen_sample1.evolveCommunities;

%% Combine populations
% takes the best 25% from each community and combines them into one
% population
gen_sample3=gen_sample2.mergeCommunities;

%% Secondary Evolution
% evolves the combined population for one epoch.
gen_sample4=gen_sample3.evolveGlobe;

toc;







