classdef Population < handle
    properties
        numMembers
        pop
        popfitness=[];
        mi
        ma
        rate
        kurt
    end
    
    methods
        function obj=Population(ps,mins,maxes,k,r)
            %constructor
            obj.mi=mins;
            obj.ma=maxes;
            obj.numMembers=ps;
            obj.rate=r;
            obj.kurt=k;
            for j=1:ps
               obj.pop{1,j}=Member(mins,maxes);
               obj.popfitness(1,j)=obj.pop{1,j}.trigger;
            end
        end
        
        function [obj,out]=breed(obj)
            %sorting fitnesses
            [temp,ix]=sort(obj.popfitness);             
            fprintf('Fitness: %20.10f',obj.popfitness(ix(1)));
            fprintf('\n');
            newpop=cell(1,obj.numMembers);
            %generating new population
            count=1;
            
            for j=1:obj.numMembers/2
            %allowing the best from previous generation to continue 
                newpop{1,j}=obj.pop{1, ix(j)};
                %mutation
                if rand<obj.rate
                   weights=obj.pop{1,ix(j)}.getWeights;
                   index=ceil(rand*length(weights));
                   weights(index)=weights(index)*(((randn-1)/obj.kurt)+1); 
                   newpop{1,j}.setWeights(weights);
                end
                count=count+1;
            end  
                
            for j=obj.numMembers/2:obj.numMembers
                
                %picking the two genes to breed, should favor more fit genes strongly.
                index=10000;
                index2=10000;
                while index>length(ix) || index2>length(ix) || index==0 || index2==0
                    index=round(abs(randn-1)*length(ix)/2);
                    index2=round(abs(randn-1)*length(ix)/2);
                end
                w1=obj.pop{1,ix(index)}.getWeights;
                w2=obj.pop{1,ix(index2)}.getWeights;
                
                %making a blank citizen for breeding
                newpop{1,j}=Member(obj.mi,obj.ma);
                
                %single crossover
                mod=zeros(1,length(w1));
                mod2=ones(1,length(w2));
                for k=1:round(rand(1,length(mod)))
                   mod(k)=1;
                   mod2(k)=0;
                end
                weights=(abs(mod).*w1)+(mod2.*w2);
                
                %mutation
                if rand<obj.rate
                   index=ceil(rand*length(weights));
                   weights(index)=weights(index)*(((randn-1)/obj.kurt)+1); 
                end
                
                
                newpop{1,j}.setWeights(weights);
                count=count+1;
                
                
            end
            %testing the new fitnesses
            
            
            %tic;
            
            obj.pop=newpop;
            
            %calculating new fitnesses (in parallel)
            temppop=obj.pop;
            temppopfitness=obj.popfitness;
            parfor j=1:obj.numMembers
               temp1=temppop{1,j}.trigger;
               temppopfitness(1,j)=temp1;
            end
            obj.popfitness=temppopfitness;
            %toc;
            
            out=temppopfitness;
            
            
            
        end
        
        function out=converganceCheck(obj)
           %returns the fitnesses of the most recent generation
           out=obj.popfitness(end,:); 
        end
        
        function out=totalfitness(obj)
            %returns fitnesses of all of the generations
            out=obj.popfitness;
        end
        
        function obj=setPop(obj,newpop)
            %sets new citizens for the population
            obj.pop=newpop;
        end
        
        function out=getPop(obj)
            %gets the cell array of citizens
            out=obj.pop;
        end
        
        
    end 
end