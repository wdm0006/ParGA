function out=fitnessFunction(gene)
command=sprintf('SRM_serve.exe %20.10fd0 %20.10fd0 %20.10fd0 %20.10fd0 %20.10fd0 %20.10fd0 %20.10fd0 %20.10fd0', gene);
[~,result]=system(command);
out=str2double(result);