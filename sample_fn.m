function out=sample_fn(gene)
command=sprintf('SRM_serve.exe %20.10fd0 %20.10fd0 %20.10fd0 %20.10fd0 %20.10fd0 %20.10fd0 %20.10fd0 %20.10fd0', gene);
[~,result]=system(command);
%fprintf(result)
out=str2double(result);